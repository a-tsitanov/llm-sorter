# Локальные LiteLLM и LM Studio

LiteLLM запускается на `http://127.0.0.1:4001/v1`, LM Studio — на `http://127.0.0.1:1234`. Порт 4000 занят другим сервисом. LiteLLM использует локальный ключ из `.env` (`LLM_API_KEY`); тот же ключ хранится в пользовательском LaunchAgent с правами доступа только для владельца.

| Алиас LiteLLM | Модель LM Studio | Назначение |
|---|---|---|
| `d1-3b` | `d1-3b` | Выбор темы через System One |
| `small` | `qwen3-vl-4b-instruct-mlx` | Проверка темы |
| `large` | `qwen/qwen3.5-9b` | Предложение новых тем |
| `bge-m3` | `text-embedding-bge-m3` | Эмбеддинги |

Конфигурация маршрутов — `deploy/litellm-local/config.yaml`. Решение `hosted_vllm` использует совместимый с LM Studio `/v1/systemone`.

## Первоначальная установка

Этот раздел нужен, если локальный LiteLLM ещё не установлен. Установите LM Studio и `lms`, затем скачайте перечисленные выше модели через интерфейс LM Studio или `lms get`. Убедитесь через `lms ls`, что их идентификаторы совпадают с `config.yaml`; при необходимости отредактируйте конфигурацию. Запустите LM Studio API и загрузите модели командами ниже.

Установите отдельное окружение LiteLLM и запустите его в отдельном терминале:

```sh
uv venv --python 3.12 "$HOME/.local/share/llm-sorter/litellm-venv"
uv pip install --python "$HOME/.local/share/llm-sorter/litellm-venv/bin/python" 'litellm[proxy]==1.106.4'
```

Создайте надёжный секретный ключ, например командой `openssl rand -hex 32` (добавьте перед результатом `sk-`), и укажите его в проектном `.env` как `LLM_API_KEY`. Не сохраняйте ключ в файлах репозитория. Для запуска в терминале задайте этот же ключ в `LITELLM_MASTER_KEY` без вывода на экран:

```sh
read -rs 'LITELLM_MASTER_KEY?Ключ LiteLLM: '
export LITELLM_MASTER_KEY
"$HOME/.local/share/llm-sorter/litellm-venv/bin/litellm" \
  --config deploy/litellm-local/config.yaml --host 127.0.0.1 --port 4001
```

Пример использует синтаксис `read` оболочки zsh. Держите ключ в секрете. Если порт 4001 занят, выберите свободный и измените `LLM_URL` в `.env`. Этот ручной запуск работает, пока открыт терминал; он сам по себе не создаёт LaunchAgent. Создавать LaunchAgent необязательно.

## Запуск и состояние

```sh
lms server start --port 1234 --bind 127.0.0.1
lms ps
launchctl kickstart -k gui/$(id -u)/local.llm-sorter.litellm
```

Если модели не загружены, их можно загрузить через API:

```sh
for model in d1-3b text-embedding-bge-m3 qwen3-vl-4b-instruct-mlx qwen/qwen3.5-9b; do
  curl -sS http://127.0.0.1:1234/api/v1/models/load \
    -H 'Content-Type: application/json' -d "{\"model\":\"$model\"}"
done
```

На уже настроенной машине LiteLLM 1.106.4 установлен в `~/.local/share/llm-sorter/litellm-venv`; LaunchAgent `~/Library/LaunchAgents/local.llm-sorter.litellm.plist` запускает его при входе пользователя. Журнал: `~/.local/share/llm-sorter/litellm.log`. LaunchAgent и журнал не создаются при ручной установке выше. Автозагрузка LM Studio и моделей этой настройкой не предусмотрена.

Для работы llm-sorter дополнительно нужны инфраструктура и процессы из [корневой инструкции](../../README.md): `make up`, затем отдельно `make process`, `make learn` и подходящий фидер. Одновременная загрузка моделей требует значительных ресурсов памяти.
