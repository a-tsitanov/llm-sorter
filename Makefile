.PHONY: help install lint fmt typecheck test test-int up down feed-dir feed-tg process learn
help: ## Показать цели
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "%-10s %s\n", $$1, $$2}'
install: ## Установить зависимости
	uv sync
lint: ## ruff check
	uv run ruff check src tests
fmt: ## ruff format + fix
	uv run ruff format src tests && uv run ruff check --fix src tests
typecheck: ## mypy
	uv run mypy src
test: ## Юнит-тесты
	uv run pytest -m "not integration"
test-int: ## Интеграционные тесты (нужен make up; RABBIT_URL при нестандартном порте)
	uv run pytest -m integration
up: ## Поднять инфраструктуру
	docker compose up -d
down: ## Остановить инфраструктуру
	docker compose down
feed-dir: ## Фидер каталога SOURCE_DIR -> inbox
	uv run llm-sorter feed dir
feed-tg: ## Фидер папок Telegram TG_FOLDERS -> inbox
	uv run llm-sorter feed tg
process: ## Обработчик inbox
	uv run llm-sorter process
learn: ## Дообучение (stream other)
	uv run llm-sorter learn
