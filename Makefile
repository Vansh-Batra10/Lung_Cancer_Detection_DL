VENV=venv
PYTHON=$(VENV)/bin/python
PIP=$(VENV)/bin/pip
FLASK=$(PYTHON) -m flask

setup:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

run:
	$(FLASK) run

show:
	$(PIP) show tensorflow
	
test:
	$(PYTHON) -m unittest discover -s tests -p "*.py"

lint:
	$(VENV)/bin/pylint app tests

infer:
	$(PYTHON) inference.py

clean:
	rm -rf $(VENV)
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type f -name "*.pyc" -exec rm -r {} +
	find . -type f -name "*.pyo" -exec rm -r {} +
