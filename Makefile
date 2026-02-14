# Use python3 by default, but you can change this to python if needed.
PYTHON = python

# Directories to lint
SRC_DIR = src

# Define the list of files to exclude from linting
EXCLUDE = venv

# flake8 configuration options (you can customize these)
FLAKE8_OPTIONS = --ignore=E501,W291,W503,W605 --exclude=$(EXCLUDE)

.PHONY: lint

lint:
	@echo "Running flake8..."
	@$(PYTHON) -m flake8 $(FLAKE8_OPTIONS) $(SRC_DIR)
	@echo "Linting completed successfully."

run-app:
	@echo "Running Streamlit app..."
	@$(PYTHON) -m streamlit run $(SRC_DIR)/app.py
	@echo "Streamlit app running successfully."

run-eval:
	@echo "Running evaluation..."
	@$(PYTHON) -m src.evals.runner
	@echo "Evaluation completed successfully."

run-test:
	@echo "Running tests..."
	@$(PYTHON) -m pytest -q
	@echo "Tests completed successfully."
