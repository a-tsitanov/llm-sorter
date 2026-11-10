# synthetic-v1

Синтетический набор транскриптов разговоров (~156 файлов) с известной эталонной разметкой для тестов и оценки llm-sorter. Состав, дерево тем и правила генерации описаны в `SPEC.md`.

Один файл = один разговор: `transcripts/<id>.json` с полями `id`, `text`, `metadata` (`gold_path`, `lang`, `kind`, опционально `note`).

## Как скормить пайплайну

Скопируйте файлы в каталог источника (`SOURCE_DIR`, по умолчанию `data/inbox`) и запустите приём:

```
cp datasets/synthetic-v1/transcripts/*.json data/inbox/
uv run llm-sorter feed dir
```

## Проверка набора

Из корня проекта:

```
uv run python datasets/synthetic-v1/validate.py     # формат, сводка, неполные группы; код выхода != 0 при ошибках
uv run python datasets/synthetic-v1/manifest.py     # пишет manifest.csv (id, gold_path, kind, lang, words)
```

## Эталон

`metadata.gold_path` — ground truth для оценки `decide`, `guard` и `discover`: путь из эталонного дерева либо `other` для шума (`noise`) и служебных текстов (`edge`). Для `near_miss` это истинная тема, а в `note` указано, какую соседнюю тему файл имитирует.

## Оценка прогона

`evaluate.py` относится к архитектуре v1: он ожидает PostgreSQL, таблицу `classifications` и `DATABASE_URL`, которых нет в текущей версии v2. Не используйте его для оценки текущего запуска. Результаты v2 публикуются в RabbitMQ; отдельного актуального скрипта оценки этого набора в репозитории пока нет.
