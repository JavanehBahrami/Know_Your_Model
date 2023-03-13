# Don't change
SRC_DIR := .


RED=\033[0;31m
GREEN=\033[0;32m
YELLOW=\033[1;33m
BLUE=\033[0;34m
CYAN=\033[0;36m
NC=\033[0m

help:  ## 💬 This help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

pre-commit:   ## 🔎 Run pre-commit hooks
	@poetry run pre-commit run --all-files

flake-lint-test:     ## 👀 Lint code via flake8
	@poetry run flake8 .

black-lint-test:     ## 👀 Check code format via black
	@poetry run black . --check

black-lint-fix:     ## 🧼 Format code via black
	@poetry run black . --config pyproject.toml

test:    ## 🎯 Unit tests
	@poetry run pytest

cover-test: 	 ## 📊 Unit tests with HTML report
	@poetry run coverage run -m pytest --disable-pytest-warnings
	@poetry run coverage html

create-docs: ## 📜 Generate configs for Documentation using mkdocs
	@poetry run mkdocs new .

build-docs: ## 📜 Generate Documentation using mkdocs
	@poetry run mkdocs build

serve-docs: ## 📜 Host Documentation on local machine
	@poetry run mkdocs serve

clean:  ## 🧹 Clean up project
	rm -rf $(SRC_DIR)/.venv
	rm -rf $(SRC_DIR)/venv
	rm -f $(SRC_DIR)/.coverage
	find $(SRC_DIR) -name "__pycache__" -type d | xargs rm -rf
	find $(SRC_DIR) -name ".pytest_cache" -type d | xargs rm -rf

.PHONY: install build pre-commit
