# Smart Mental Wellness Assistant — Deployment & Docker Guide

## Deployment Architectures

The platform supports two deployment methods:
1. **Local Native Development** (FastAPI + React Vite + MongoDB)
2. **Containerized Production Deployment** (Docker Compose with isolated services)

---

## 1. Native Local Deployment

### Step A: Prerequisites
- Python 3.10+
- Node.js 18+
- MongoDB 6.0+

### Step B: Backend Setup & Training
```bash
# Clone repository
cd backend

# Install Python dependencies
pip install -r requirements.txt

# Environment Setup
cp ../.env.example .env

# Train ML Models (Optional if pre-trained artifacts exist)
python ../scripts/train_text.py --epochs 2
python ../scripts/train_cv.py --epochs 20
python ../scripts/train_crisis.py

# Start Backend API Server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Step C: Frontend Setup
```bash
cd frontend
npm install
npm run dev
```
- Open application at `http://localhost:3000` (or `http://localhost:5173`).

---

## 2. Docker Deployment

### Services Overview
- `backend`: FastAPI app running on port `8000`.
- `frontend`: Nginx web server serving Vite production bundle on port `3000`.
- `mongodb`: MongoDB database on port `27017`.

```bash
# Build and launch all services in detached mode
docker-compose up --build -d

# Check service logs
docker-compose logs -f

# Verify status
python scripts/health_check.py
```
