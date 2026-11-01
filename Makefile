.PHONY: install test lint smoke replay api frontend demo

install:
	python3.12 -m venv .venv
	.venv/bin/pip install -e '.[dev]'
	cd frontend && npm install

lint:
	.venv/bin/ruff check .
	cd frontend && npm run lint

test:
	.venv/bin/pytest
	cd frontend && npm run build

smoke:
	.venv/bin/portfolio-agent data validate --dataset examples/smoke

replay:
	.venv/bin/portfolio-agent replay --dataset examples/smoke --mode mock --adaptive --rebalance monthly --iterations 3 --out runs/smoke

api:
	PORTFOLIO_AGENT_RUNS=$$(pwd)/runs .venv/bin/uvicorn portfolio_agent.api.app:app --host 127.0.0.1 --port 8000

frontend:
	cd frontend && npm run dev

demo:
	@echo "Run 'make api' and 'make frontend' in two terminals, then open http://127.0.0.1:3000"
