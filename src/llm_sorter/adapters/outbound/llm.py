"""Адаптеры моделей поверх OpenAI-совместимого API LM Studio."""

import json
import re
from typing import Any

from openai import APIStatusError, AsyncOpenAI
from pydantic import ValidationError

from llm_sorter.domain.discover import TopicGroup, TopicProposal
from llm_sorter.domain.traversal import Candidate
from llm_sorter.ports.models import InvalidModelOutput

_ID_RE = re.compile(r"^[a-z0-9_]+$")
_MAX_EXAMPLES = 3
_GENERATION_TEMPERATURE = 0.2
# статусы, которые не исправятся повтором для этого сообщения (длина контекста, схема).
# 404 и model_not_found - ошибка конфигурации (неверное имя модели): её ждут, как сбой,
# иначе всё молча уйдёт в other с model_error
_REJECTED = {400, 413, 422}


# LM Studio: model_not_found для chat, invalidModelIdentifier для /v1/systemone
_CONFIG_CODES = {"model_not_found", "invalidModelIdentifier"}


def _config_error(e: APIStatusError) -> bool:
    code = getattr(e, "code", None)
    return e.status_code == 404 or code in _CONFIG_CODES


def _rejected(e: APIStatusError) -> InvalidModelOutput | None:
    """Отказ, который повтор не исправит, - невалидный выход; иначе None (сбой, ждём)."""
    if e.status_code in _REJECTED and not _config_error(e):
        return InvalidModelOutput(f"model request rejected: {e.status_code} {e.message}")
    return None


# столько символов сообщения уходит в decide/guard: длинный текст не должен
# упираться в контекст модели, а начало разговора несёт тему
MAX_TEXT_CHARS = 12000
# длина варианта в systemone: варианты читаются после текста на каждом шаге спуска;
# на 13 темах полные описания (~5.8k символов) - 3.6 с на шаг,
# обрезка до 80 - 0.7 с при 28/30 тех же ответов
CRITERION_CHARS = 120
_LANG_NOTE = "Транскрипт может быть на любом языке; оценивай смысл независимо от языка."


def make_openai_client(url: str, api_key: str) -> AsyncOpenAI:
    # ретраи с backoff делает обработчик; max_retries=0, чтобы не удваивать ожидание
    return AsyncOpenAI(base_url=url, api_key=api_key, timeout=110, max_retries=0)


def _schema(name: str, schema: dict[str, Any]) -> dict[str, Any]:
    return {"type": "json_schema", "json_schema": {"name": name, "strict": True, "schema": schema}}


def _enum_schema(name: str, key: str, values: list[str]) -> dict[str, Any]:
    return _schema(
        name,
        {
            "type": "object",
            "properties": {key: {"type": "string", "enum": values}},
            "required": [key],
            "additionalProperties": False,
        },
    )


async def _chat(
    client: AsyncOpenAI,
    model: str,
    system: str,
    user: str,
    temperature: float,
    response_format: dict[str, Any] | None = None,
) -> str:
    kwargs: dict[str, Any] = {}
    if response_format is not None:
        kwargs["response_format"] = response_format
    try:
        resp = await client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            **kwargs,
        )
    except APIStatusError as e:
        # повтор даст тот же отказ: это невалидный выход, а не сбой связи
        if (bad := _rejected(e)) is not None:
            raise bad from e
        raise
    return resp.choices[0].message.content or ""


async def _chat_json(
    client: AsyncOpenAI,
    model: str,
    system: str,
    user: str,
    temperature: float,
    response_format: dict[str, Any],
) -> dict[str, Any]:
    """Вызов чата со строгой json-схемой; нарушения формата -> InvalidModelOutput."""
    content = await _chat(client, model, system, user, temperature, response_format)
    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        raise InvalidModelOutput(f"невалидный JSON: {content!r}") from e
    if not isinstance(data, dict):
        raise InvalidModelOutput(f"ожидался JSON-объект: {content!r}")
    return data


def _pick_enum(data: dict[str, Any], key: str, allowed: list[str]) -> str:
    value = data.get(key)
    if not isinstance(value, str) or value not in allowed:
        raise InvalidModelOutput(f"{key}={value!r} вне списка {allowed}")
    return value


def _format_candidate(c: Candidate) -> str:
    lines = [f"- id: {c.id}", f"  title: {c.title}", f"  description: {c.description}"]
    lines += [f"  пример: {e}" for e in c.examples[:_MAX_EXAMPLES]]
    return "\n".join(lines)


def _level_hint(level: int, max_depth: int) -> str:
    """Насколько общей должна быть тема на этом уровне дерева."""
    if level >= max_depth:
        return (
            "Это последний уровень: тема - конкретный сценарий внутри родителя "
            "(например «Рефинансирование ипотеки»)."
        )
    if level == 1:
        return (
            "Это первый уровень под корнем: тема - широкий домен, внутри которого позже "
            "появятся подтемы (например «Финансы», «Здоровье», «Путешествия», «Связь»). "
            "Не называй тему по отдельной услуге или жалобе."
        )
    return (
        "Это промежуточный уровень: тема - направление внутри родителя, шире отдельного "
        "сценария (например «Кредиты» внутри «Финансы»)."
    )


class LLMDecision:
    def __init__(self, client: AsyncOpenAI, model: str) -> None:
        self._client = client
        self._model = model

    async def decide(self, text: str, candidates: list[Candidate]) -> str:
        ids = [c.id for c in candidates]
        system = (
            "Ты классификатор разговоров. Выбери одну тему, к которой относится разговор; "
            "если ни одна не подходит — other. Ответь JSON с полем choice. " + _LANG_NOTE
        )
        user = "Темы:\n" + "\n".join(_format_candidate(c) for c in candidates)
        user += f"\n\nРазговор:\n{text[:MAX_TEXT_CHARS]}"
        data = await _chat_json(
            self._client, self._model, system, user, 0, _enum_schema("choice", "choice", ids)
        )
        return _pick_enum(data, "choice", ids)


class SystemOneDecision:
    """Decide через decision-модель (Liquid d1): POST /v1/systemone, один вопрос choice по всем
    кандидатам узла. Модель не генерирует текст, а возвращает вероятности вариантов."""

    def __init__(self, client: AsyncOpenAI, model: str) -> None:
        self._client = client
        self._model = model

    async def decide(self, text: str, candidates: list[Candidate]) -> str:
        ids = [c.id for c in candidates]
        criteria = {
            c.id: " ".join(f"{c.title}: {c.description}".split())[:CRITERION_CHARS]
            for c in candidates
        }
        body = {
            "model": self._model,
            "state": text[:MAX_TEXT_CHARS],
            "questions": {
                "topic": {
                    "type": "choice",
                    "instructions": "Which topic does this message belong to?",
                    "criteria": criteria,
                }
            },
        }
        try:
            data = await self._client.post("/systemone", cast_to=object, body=body)
        except APIStatusError as e:
            if (bad := _rejected(e)) is not None:
                raise bad from e
            raise
        answer = data.get("answers", {}).get("topic") if isinstance(data, dict) else None
        if not isinstance(answer, dict):
            raise InvalidModelOutput(f"systemone: нет ответа topic: {data!r}")
        return _pick_enum(answer, "choice", ids)


class LLMGuard:
    def __init__(self, client: AsyncOpenAI, model: str) -> None:
        self._client = client
        self._model = model

    async def check(self, text: str, question: str) -> bool:
        system = (
            "Ты проверяешь разговор. Ответь на вопрос строго yes или no в JSON с полем answer. "
            + _LANG_NOTE
        )
        user = f"Вопрос: {question}\n\nРазговор:\n{text[:MAX_TEXT_CHARS]}"
        data = await _chat_json(
            self._client,
            self._model,
            system,
            user,
            0,
            _enum_schema("answer", "answer", ["yes", "no"]),
        )
        return _pick_enum(data, "answer", ["yes", "no"]) == "yes"


def _proposal_schema(sibling_ids: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "covered_by": {"type": "string", "enum": ["", *sibling_ids]},
            "id": {"type": "string", "pattern": "^[a-z0-9_]+$"},
            "title": {"type": "string"},
            "description": {"type": "string"},
            "examples": {"type": "array", "items": {"type": "string", "minLength": 1}},
            "guard": {"type": "string"},
        },
        "required": ["covered_by", "id", "title", "description", "examples", "guard"],
        "additionalProperties": False,
    }


def _topic_rules(language: str) -> str:
    """Общие правила полей темы для propose_topic и group_clusters."""
    return (
        "В description перечисли подтемы, которые видны в образцах, - по нему другие модели "
        "будут относить разговоры к теме. "
        "id - 1-3 английских слова в snake_case по смыслу темы, полными словами, без "
        "сокращений и транслита (military_action, finance, middle_east, не boe и не fin). "
        "guard - сам вопрос, на который отвечают да или нет, проверяющий принадлежность "
        "разговора к теме; пиши вопрос, а не ответ (не «yes»), например: «Разговор о "
        "банковских картах?». examples - 2-3 короткие типичные фразы из разговоров этой темы. "
        f"Title, description, examples и guard пиши на языке: {language}. "
        "Не повторяй уже существующие темы. " + _LANG_NOTE
    )


def _parse_proposal[P: TopicProposal](cls: type[P], data: Any) -> P:
    try:
        proposal = cls.model_validate(data)
    except ValidationError as e:
        raise InvalidModelOutput(f"некорректное предложение темы: {e}") from e
    if not _ID_RE.match(proposal.id):
        raise InvalidModelOutput(f"id {proposal.id!r} не соответствует ^[a-z0-9_]+$")
    examples = [e.strip() for e in proposal.examples if e.strip()]
    return proposal.model_copy(update={"examples": examples})


def _context_block(parents: list[Candidate], siblings: list[Candidate]) -> str:
    neighbours = "\n".join(_format_candidate(c) for c in siblings) or "(нет)"
    return f"{_parent_block(parents)}\n\nСоседние темы (не повторять):\n{neighbours}"


def _parent_block(parents: list[Candidate]) -> str:
    if not parents:
        return "Родитель: корень дерева (новая тема - одна из тем первого уровня)."
    p = parents[-1]
    path = " > ".join(c.title for c in parents)
    return f"Родитель: {path}\nОписание родителя: {p.description}"


class LLMText:
    def __init__(self, client: AsyncOpenAI, model: str) -> None:
        self._client = client
        self._model = model

    async def propose_topic(
        self,
        samples: list[str],
        parents: list[Candidate],
        siblings: list[Candidate],
        language: str,
        level: int,
        max_depth: int,
    ) -> TopicProposal:
        system = (
            "Ты проектируешь дерево тем для классификации разговоров. По образцам из одного "
            "кластера предложи одну новую тему. "
            f"Новая тема встанет на уровень {level} из {max_depth}. "
            f"{_level_hint(level, max_depth)} "
            "Тема должна покрывать ВСЕ образцы, а не один из них: если образцы о разном, назови "
            "то, что их объединяет, а не частный случай. "
            "Если образцы уже покрывает одна из соседних тем, укажи её id в covered_by "
            "(тогда новая тема не нужна), иначе covered_by - пустая строка. "
            + _topic_rules(language)
        )
        numbered = "\n\n".join(f"[{i}] {s}" for i, s in enumerate(samples, 1))
        user = f"{_context_block(parents, siblings)}\n\nОбразцы разговоров:\n{numbered}"
        data = await _chat_json(
            self._client,
            self._model,
            system,
            user,
            _GENERATION_TEMPERATURE,
            _schema("topic_proposal", _proposal_schema([c.id for c in siblings])),
        )
        return _parse_proposal(TopicProposal, data)

    async def group_clusters(
        self,
        clusters: list[list[str]],
        parents: list[Candidate],
        siblings: list[Candidate],
        language: str,
        level: int,
        max_depth: int,
    ) -> list[TopicGroup]:
        system = (
            "Ты проектируешь дерево тем для классификации разговоров. Ниже пронумерованные "
            "кластеры похожих разговоров одного узла. Объедини кластеры в темы: "
            f"новые темы встанут на уровень {level} из {max_depth}. "
            f"{_level_hint(level, max_depth)} "
            "Кластеры об одном домене объединяй в одну тему, а не делай тему на каждый кластер, "
            "но и не сливай разные домены в одну тему: политика, экономика, война, технологии, "
            "общество - разные домены. "
            f"Сделай от {min(3, len(clusters))} до {min(12, len(clusters))} тем. "
            "Темы не должны пересекаться между собой и с соседними. Каждый кластер отнеси не "
            "более чем к одной теме; кластер, который не подходит ни к одной широкой теме, не "
            "включай никуда. Не создавай тем «общие новости», «разное», «прочее», «события» - "
            "для неподходящего есть other. Если кластеры уже покрывает соседняя тема, верни "
            "для неё элемент с её id в covered_by и этими кластерами, иначе covered_by - "
            "пустая строка. В clusters - номера кластеров темы. " + _topic_rules(language)
        )
        blocks = []
        for n, samples in enumerate(clusters, 1):
            lines = "\n".join(f"- {' '.join(s.split())}" for s in samples)
            blocks.append(f"Кластер {n}:\n{lines}")
        user = f"{_context_block(parents, siblings)}\n\n" + "\n\n".join(blocks)
        item = _proposal_schema([c.id for c in siblings])
        item["properties"]["clusters"] = {
            "type": "array",
            "items": {"type": "integer", "minimum": 1, "maximum": len(clusters)},
        }
        item["required"] = [*item["required"], "clusters"]
        schema = {
            "type": "object",
            "properties": {"topics": {"type": "array", "items": item}},
            "required": ["topics"],
            "additionalProperties": False,
        }
        data = await _chat_json(
            self._client,
            self._model,
            system,
            user,
            _GENERATION_TEMPERATURE,
            _schema("topic_groups", schema),
        )
        topics = data.get("topics")
        if not isinstance(topics, list):
            raise InvalidModelOutput(f"ожидался список topics: {data!r}")
        groups = []
        for raw in topics:
            g = _parse_proposal(TopicGroup, raw)
            if not g.clusters or any(not 1 <= n <= len(clusters) for n in g.clusters):
                raise InvalidModelOutput(f"номера кластеров вне 1..{len(clusters)}: {g.clusters}")
            groups.append(g.model_copy(update={"clusters": [n - 1 for n in g.clusters]}))
        return groups

    async def same_topic(self, a: Candidate, b: Candidate) -> bool:
        system = (
            "Определи, описывают ли две темы одно и то же по смыслу. "
            "Ответь JSON с полем answer: yes или no."
        )
        user = f"Тема A:\n{_format_candidate(a)}\n\nТема B:\n{_format_candidate(b)}"
        data = await _chat_json(
            self._client,
            self._model,
            system,
            user,
            0,
            _enum_schema("answer", "answer", ["yes", "no"]),
        )
        return _pick_enum(data, "answer", ["yes", "no"]) == "yes"


class LLMEmbedder:
    def __init__(
        self, client: AsyncOpenAI, model: str, max_chars: int = 1200, batch_size: int = 8
    ) -> None:
        if max_chars < 1 or batch_size < 1:
            raise ValueError("embedding limits must be positive")
        self._client = client
        self._model = model
        self._max_chars = max_chars
        self._batch_size = batch_size

    async def embed(self, texts: list[str]) -> list[list[float]]:
        result: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = [t[: self._max_chars] for t in texts[start : start + self._batch_size]]
            resp = await self._client.embeddings.create(model=self._model, input=batch)
            if sorted(d.index for d in resp.data) != list(range(len(batch))):
                raise InvalidModelOutput("embedding response indices do not match input")
            result.extend(list(d.embedding) for d in sorted(resp.data, key=lambda d: d.index))
        return result
