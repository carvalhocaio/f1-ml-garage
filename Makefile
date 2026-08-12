.PHONY: help lint lint-fix format format-check check test test-integration test-cov

help:
	@echo "Comandos disponíveis no Makefile:"
	@echo "  make lint             - Executa o linter (ruff check)"
	@echo "  make lint-fix         - Corrige automaticamente problemas do linter (ruff check --fix)"
	@echo "  make format           - Formata o código (ruff format)"
	@echo "  make format-check     - Verifica se o código está formatado (ruff format --check)"
	@echo "  make check            - Executa o linter e valida a formatação"
	@echo "  make test             - Roda os testes rápidos (unit), sem rede"
	@echo "  make test-integration - Roda também os testes de integração (rede real do FastF1)"
	@echo "  make test-cov         - Roda os testes com relatório de cobertura"

lint:
	uv run ruff check .

lint-fix:
	uv run ruff check --fix .

format:
	uv run ruff format .

format-check:
	uv run ruff format --check .

check: lint format-check

test:
	uv run pytest -m "not integration"

test-integration:
	F1_ML_GARAGE_RUN_INTEGRATION=1 uv run pytest -m integration

test-cov:
	uv run pytest -m "not integration" --cov=f1_ml_garage --cov-report=term-missing
