# Test KYM Dashboard
In this part we want to test kym functions and methods using pytest.

we separated the kym tests into two different parts
1. test utils; for testing functionality of helper functions in utils.py module
2. test callbacks; for testing functionality of callback functions of dash app


## How to run test
run this command in terminal:

```
poetry run pytest -v -m unit

```

```
├── tests
    ├── __init__.py
    ├── README.md
    ├── test_callbacks.py
    └── test_utils.py

```
