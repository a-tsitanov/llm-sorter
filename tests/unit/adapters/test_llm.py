import json

import httpx2
import pytest
from openai import AsyncOpenAI

from llm_sorter.adapters.outbound.llm import (
    LLMDecision,
    LLMEmbedder,
    LLMGuard,
    LLMText,
    make_openai_client,
)
from llm_sorter.domain.traversal import Candidate
from llm_sorter.ports.models import InvalidModelOutput

CHAT = "http://test/v1/chat/completions"
EMBED = "http://test/v1/embeddings"


class Route:
    """Мок HTTP-транспорта openai (он работает на httpx2, respx его не патчит)."""

    def __init__(self) -> None:
        self.requests: list[httpx2.Request] = []
        self.responses: dict[str, httpx2.Response] = {}

    def handler(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        return self.responses[str(request.url)]

    def mock(self, url: str, response: httpx2.Response) -> None:
        self.responses[url] = response

    @property
    def last_body(self) -> dict:
        return json.loads(self.requests[-1].content)


_route = Route()


@pytest.fixture(autouse=True)
def route() -> Route:
    global _route
    _route = Route()
    return _route


def _client() -> AsyncOpenAI:
    http = httpx2.AsyncClient(transport=httpx2.MockTransport(_route.handler))
    return AsyncOpenAI(base_url="http://test/v1", api_key="x", http_client=http, max_retries=0)


def _chat(content: str) -> httpx2.Response:
    return httpx2.Response(
        200,
        json={
            "id": "c1",
            "object": "chat.completion",
            "created": 0,
            "model": "m",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": content},
                }
            ],
        },
    )


def _cands() -> list[Candidate]:
    return [
        Candidate(id="crypto", title="Крипто", description="Криптовалюты", examples=["a"] * 5),
        Candidate(id="banks", title="Банки", description="Банковские услуги"),
        Candidate(id="other", title="Other", description="Ничего не подходит"),
    ]


async def test_decide_returns_choice(route: Route) -> None:
    route.mock(CHAT, _chat('{"choice":"crypto"}'))
    result = await LLMDecision(_client(), "m").decide("текст", _cands())
    assert result == "crypto"
    body = route.last_body
    assert body["temperature"] == 0
    fmt = body["response_format"]
    assert fmt["type"] == "json_schema"
    assert fmt["json_schema"]["strict"] is True
    assert fmt["json_schema"]["schema"]["properties"]["choice"]["enum"] == [
        "crypto",
        "banks",
        "other",
    ]
    user = body["messages"][-1]["content"]
    assert "crypto" in user and "текст" in user
    assert user.count("пример:") == 3  # не более 3 примеров


async def test_decide_out_of_enum_raises(route: Route) -> None:
    route.mock(CHAT, _chat('{"choice":"nft"}'))
    with pytest.raises(InvalidModelOutput):
        await LLMDecision(_client(), "m").decide("t", _cands())


async def test_decide_bad_json_raises(route: Route) -> None:
    route.mock(CHAT, _chat("crypto"))
    with pytest.raises(InvalidModelOutput):
        await LLMDecision(_client(), "m").decide("t", _cands())


async def test_guard_yes_no(route: Route) -> None:
    route.mock(CHAT, _chat('{"answer":"yes"}'))
    guard = LLMGuard(_client(), "m")
    assert await guard.check("t", "q?") is True
    body = route.last_body
    assert body["temperature"] == 0
    assert body["response_format"]["json_schema"]["schema"]["properties"]["answer"]["enum"] == [
        "yes",
        "no",
    ]
    route.mock(CHAT, _chat('{"answer":"no"}'))
    assert await guard.check("t", "q?") is False


async def test_guard_invalid_raises(route: Route) -> None:
    route.mock(CHAT, _chat('{"answer":"maybe"}'))
    with pytest.raises(InvalidModelOutput):
        await LLMGuard(_client(), "m").check("t", "q?")


async def test_propose_topic_parses(route: Route) -> None:
    payload = {
        "id": "defi_news",
        "title": "DeFi",
        "description": "Новости DeFi",
        "examples": ["e1", "e2"],
        "guard": "Обсуждается ли DeFi?",
    }
    route.mock(CHAT, _chat(json.dumps(payload)))
    proposal = await LLMText(_client(), "m").propose_topic(
        ["s1", "s2"], "root/crypto", _cands()[:2], "ru", level=2, max_depth=3
    )
    assert proposal.id == "defi_news"
    assert proposal.examples == ["e1", "e2"]
    body = route.last_body
    assert body["temperature"] == 0.2
    assert "root/crypto" in body["messages"][-1]["content"]


async def test_propose_topic_drops_blank_examples(route: Route) -> None:
    # модель вернула examples: [""] - в дерево пустые примеры не попадают
    payload = {
        "id": "defi",
        "title": "DeFi",
        "description": "d",
        "examples": ["", "  ", " фраза "],
        "guard": "g",
    }
    route.mock(CHAT, _chat(json.dumps(payload)))
    proposal = await LLMText(_client(), "m").propose_topic(
        ["s"], "root", [], "ru", level=1, max_depth=3
    )
    assert proposal.examples == ["фраза"]


@pytest.mark.parametrize(
    ("level", "max_depth", "hint"),
    [(1, 3, "широкий домен"), (2, 3, "направление"), (3, 3, "конкретный сценарий")],
)
async def test_propose_topic_prompt_has_level(
    route: Route, level: int, max_depth: int, hint: str
) -> None:
    payload = {"id": "x", "title": "t", "description": "d", "examples": ["e"], "guard": "g"}
    route.mock(CHAT, _chat(json.dumps(payload)))
    await LLMText(_client(), "m").propose_topic(
        ["s"], "root", [], "ru", level=level, max_depth=max_depth
    )
    system = route.last_body["messages"][0]["content"]
    assert f"уровень {level} из {max_depth}" in system
    assert hint in system
    assert "ВСЕ образцы" in system


async def test_propose_topic_bad_id_raises(route: Route) -> None:
    payload = {"id": "Bad Id", "title": "t", "description": "d", "examples": [], "guard": "g"}
    route.mock(CHAT, _chat(json.dumps(payload)))
    with pytest.raises(InvalidModelOutput):
        await LLMText(_client(), "m").propose_topic(["s"], "root", [], "ru", level=1, max_depth=3)


async def test_propose_topic_missing_field_raises(route: Route) -> None:
    route.mock(CHAT, _chat('{"id":"x"}'))
    with pytest.raises(InvalidModelOutput):
        await LLMText(_client(), "m").propose_topic(["s"], "root", [], "ru", level=1, max_depth=3)


async def test_same_topic(route: Route) -> None:
    route.mock(CHAT, _chat('{"answer":"yes"}'))
    a, b = _cands()[:2]
    assert await LLMText(_client(), "m").same_topic(a, b) is True
    assert route.last_body["temperature"] == 0


async def test_embed_order(route: Route) -> None:
    route.mock(
        EMBED,
        httpx2.Response(
            200,
            json={
                "object": "list",
                "model": "e",
                "usage": {"prompt_tokens": 1, "total_tokens": 1},
                "data": [
                    {"object": "embedding", "index": 1, "embedding": [0.0, 1.0]},
                    {"object": "embedding", "index": 0, "embedding": [1.0, 0.0]},
                ],
            },
        ),
    )
    vecs = await LLMEmbedder(_client(), "e").embed(["a", "b"])
    assert vecs == [[1.0, 0.0], [0.0, 1.0]]
    assert route.last_body["input"] == ["a", "b"]


def test_make_openai_client() -> None:
    client = make_openai_client("http://x:1/v1", "k")
    assert str(client.base_url).startswith("http://x:1/v1")
    assert client.api_key == "k"
    # ретраи с backoff делает обработчик; свой таймаут, чтобы зависший запрос не висел вечно
    assert client.max_retries == 0
    assert client.timeout == 110


def _error(status: int) -> httpx2.Response:
    return httpx2.Response(status, json={"error": {"message": "boom", "type": "x"}})


@pytest.mark.parametrize("status", [400, 413, 422])
async def test_deterministic_4xx_is_invalid_output(route: Route, status: int) -> None:
    """Запрос, который всегда будет отвергнут, не должен ретраиться вечно как сбой связи."""
    route.mock(CHAT, _error(status))
    with pytest.raises(InvalidModelOutput, match=str(status)):
        await LLMDecision(_client(), "m").decide("t", _cands())


@pytest.mark.parametrize("status", [404, 408, 429, 500, 503])
async def test_transient_errors_propagate(route: Route, status: int) -> None:
    from openai import APIStatusError

    route.mock(CHAT, _error(status))
    with pytest.raises(APIStatusError):
        await LLMGuard(_client(), "m").check("t", "q?")


async def test_long_text_truncated_in_prompt(route: Route) -> None:
    from llm_sorter.adapters.outbound.llm import MAX_TEXT_CHARS

    route.mock(CHAT, _chat('{"answer":"yes"}'))
    await LLMGuard(_client(), "m").check("я" * (MAX_TEXT_CHARS * 3), "q?")
    user = route.last_body["messages"][1]["content"]
    assert user.count("я") == MAX_TEXT_CHARS


async def test_model_not_found_is_config_error_not_bad_output(route: Route) -> None:
    """LM Studio отвечает 400 model_not_found на неверное имя: это конфиг, его ждут, а не фолбэк."""
    from openai import APIStatusError

    route.mock(
        CHAT,
        httpx2.Response(
            400,
            json={
                "error": {
                    "message": 'Invalid model identifier "qwen3-4b".',
                    "type": "invalid_request_error",
                    "param": "model",
                    "code": "model_not_found",
                }
            },
        ),
    )
    with pytest.raises(APIStatusError) as e:
        await LLMDecision(_client(), "qwen3-4b").decide("t", _cands())
    assert not isinstance(e.value, InvalidModelOutput)


async def test_propose_prompt_asks_for_question_not_answer(route: Route) -> None:
    route.mock(
        CHAT,
        _chat('{"id":"t","title":"T","description":"D","examples":["e"],"guard":"Разговор о T?"}'),
    )
    await LLMText(_client(), "m").propose_topic(["s"], "", [], "ru", level=1, max_depth=3)
    system = route.last_body["messages"][0]["content"]
    assert "не ответ" in system and "?" in system
