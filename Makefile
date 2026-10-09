.PHONY: setup train-text train-cv train-crisis test health docker-up docker-down clean

setup:
	python scripts/setup.py

train-text:
	python scripts/train_text.py --epochs 2

train-cv:
	python scripts/train_cv.py --epochs 20

train-crisis:
	python scripts/train_crisis.py

evaluate:
	python scripts/evaluate.py

test:
	cd backend && pytest -v

health:
	python scripts/health_check.py

docker-up:
	docker-compose up --build -d

docker-down:
	docker-compose down

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
