# Makefile — Bangalore Gridlock Intelligence

.PHONY: help install test train run clean

help:
	@echo "Bangalore Gridlock Intelligence Commands:"
	@echo "  make install  - Install Python dependencies"
	@echo "  make test     - Run unit test suite"
	@echo "  make train    - Run end-to-end preprocessing & XGBoost training pipeline"
	@echo "  make run      - Launch Streamlit interactive dashboard"
	@echo "  make clean    - Remove cached pyc and temp files"

install:
	pip install --upgrade pip
	pip install -r requirements.txt

test:
	python3 -m unittest discover tests -v

train:
	python3 preprocess_and_train.py

run:
	streamlit run app.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
