.PHONY: deps lock test clean

deps:
	@echo "📦 Installing VAIO dependencies..."
	pip install -r requirements.txt
	pip install -e .

lock:
	@echo "🔒 Freezing dependency hashes..."
	pip-compile requirements.txt --generate-hashes -o vaio.lock

test:
	@echo "🧪 Running basic CLI sanity tests..."
	vaio -h
	vaio debug
	python -m vaio -h

clean:
	@echo "🧹 Cleaning build artifacts..."
	rm -rf build dist *.egg-info __pycache__
