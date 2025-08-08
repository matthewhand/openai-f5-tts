.PHONY: install dev test docs

install:
	pip install -r requirements.txt

dev:
	openai-f5-tts --host 0.0.0.0 --port 9090 --debug

test:
	pytest -q

docs:
	@echo "Installing package in editable mode..."
	@python3 -m pip install -e .
	@echo "Generating OpenAPI spec..."
	@nohup python3 -m app.cli serve --host 127.0.0.1 --port 9090 > /dev/null 2>&1 & \
		PROC=$$! && \
		sleep 3 && \
		curl -s http://127.0.0.1:9090/apidocs/swagger.json -o openapi.json && \
		echo "OpenAPI spec saved to openapi.json" && \
		kill $$PROC || true
