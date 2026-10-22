.PHONY: help install lint fmt typecheck test test-int up down migrate worker ingest consume
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
test-int: ## Интеграционные тесты (БД TEST_DATABASE_URL, по умолчанию sorter_test; RABBIT_URL при нестандартном порте)
	uv run pytest -m integration
up: ## Поднять инфраструктуру
	docker compose up -d
down: ## Остановить инфраструктуру
	docker compose down
migrate: ## Применить миграции alembic
	uv run alembic upgrade head
worker: ## Temporal worker
	uv run llm-sorter worker
ingest: ## Поллер входящих файлов
	uv run llm-sorter ingest
consume: ## Consumer RabbitMQ
	uv run llm-sorter consume
