# Smart Mental Wellness Assistant — Comprehensive Execution Guide

This guide provides complete instructions for starting, verifying, and developing the **Smart Mental Wellness Assistant** platform across native local environments and containerized Docker setups.

---

## 1. System Requirements & Prerequisites

| Requirement | Supported Version | Purpose |
| :--- | :--- | :--- |
| **Python** | 3.10+ | FastAPI backend server, PyTorch neural networks, HF Transformers |
| **Node.js & npm** | 18+ (npm 9+) | Vite dev server, TypeScript compiler, React frontend |
| **MongoDB** | 6.0+ | Persistent storage (mood history, user preferences, feedback loops) |
| **Hardware** | CPU / CUDA GPU | Auto-detects NVIDIA CUDA GPU if available; defaults gracefully to CPU |

---

## 2. Initial Setup & Environment Configuration

### Step A: Repository Setup
Clone and enter the workspace root:
```bash
git clone https://github.com/saipraveen-k/Smart-Mental-Wellness-Assistant.git
cd "Smart Mental Wellness Assistant"
```

### Step B: Initialize Environment Configuration
Generate your local `.env` configuration file from the template:
```bash
python scripts/setup.py
```

### Do You Need to Update `.env`?
* **For basic local runs: NO.** The pre-configured defaults in `.env` are ready out-of-the-box (`mongodb://localhost:27017`, `http://localhost:5173`, pre-trained model paths).
* **When you SHOULD update `.env`:**

| Variable | Default | When to Change |
| :--- | :--- | :--- |
| `JWT_SECRET` | `CHANGE_ME` | **Recommended:** Replace `CHANGE_ME` with any static string (e.g. `JWT_SECRET=my-dev-secret-12345`). Without this, an ephemeral key is generated on boot, which invalidates logins whenever the server restarts. |
| `MONGODB_URI` | `mongodb://localhost:27017` | Change only if using **MongoDB Atlas (cloud)** or a password-authenticated instance (e.g. `mongodb+srv://user:pass@cluster.mongodb.net`). |
| `CRISIS_RESOURCES_COUNTRY` | `US` | Change to your 2-letter ISO country code (e.g. `IN` for India, `GB` for UK) to return localized helpline contacts instead of US 988. |
| `MODEL_DEVICE` | `auto` | By default auto-detects CUDA GPU vs CPU. Set explicitly to `cpu` or `cuda` if needed. |

---

## 3. Running the Application (Local Development)

### Step 1: Ensure MongoDB is Active
MongoDB must be reachable on port `27017`.

**Windows Service:**
```powershell
net start MongoDB
```

**Docker Container Alternative:**
```powershell
docker run -d --name smwa-mongodb -p 27017:27017 -v mongodb_data:/data/db mongo:7.0
```

**Linux / macOS:**
```bash
brew services start mongodb-community
# or: sudo systemctl start mongod
```

---

### Step 2: Start Backend API Server
Open a terminal in the project root:

```bash
cd backend
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

* **API Base URL**: `http://127.0.0.1:8000`
* **Interactive Swagger UI**: `http://127.0.0.1:8000/docs`
* **Alternative Redoc Documentation**: `http://127.0.0.1:8000/redoc`
* **Health Check Endpoint**: `http://127.0.0.1:8000/health`

---

### Step 3: Start React Frontend
Open a separate terminal in the project root:

```bash
cd frontend
npm install
npm run dev
```

* **Frontend Application Interface**: `http://localhost:5173`
* The Vite dev server will connect with the backend API running at `http://127.0.0.1:8000`.

---

## 4. Single-Command Deployment via Docker Compose

To run MongoDB, the FastAPI backend, and the production-built React frontend in isolated containers:

```bash
# Build and launch all services in detached mode
docker-compose up --build -d

# Check container status
docker-compose ps

# Follow container logs
docker-compose logs -f

# Stop and remove containers
docker-compose down
```

### Docker Service Endpoints:
* **Frontend Web Application**: `http://localhost:3000`
* **Backend REST API**: `http://localhost:8000`
* **Interactive Swagger Docs**: `http://localhost:8000/docs`
* **MongoDB Database**: `localhost:27017`

---

## 5. Automated Verification & Quality Assurance

Run the automated verification suite to validate database connectivity, model pipelines, and API routes:

### 1. End-to-End System Health Check
Verifies MongoDB connectivity, Hugging Face dataset provenance, DistilBERT text inference, CV face detection, crisis screening, and route health:
```bash
python scripts/health_check.py
```
*Expected Result: `SUMMARY: Passed 9/9 checks. ALL E2E CHECKS PASSED SUCCESSFULLY! ✓`*

### 2. Comprehensive Automated Test Suite (PyTest)
Executes all unit tests, security audits, router validations, and end-to-end multi-step user workflows:
```bash
cd backend
pytest -v
```
*Expected Result: `24 passed in backend/tests`*

### 3. Model Benchmark & Evaluation Summary
Evaluates trained models across accuracy, macro F1, and recall metrics:
```bash
python scripts/evaluate.py
```

### 4. Frontend Production Build Check
Validates TypeScript types and builds the production bundle:
```bash
cd frontend
npm run build
```

---

## 6. Model Retraining Workflows (Optional)

Pre-trained weights and configurations are already included in `ml/artifacts/`. If you want to retrain models on latest data:

### Train Text Emotion Model (DistilBERT on GoEmotions):
```bash
python scripts/train_text.py --epochs 2 --batch-size 32
```

### Train Facial Emotion Model (FacialEmotionCNN on FER-2013):
```bash
python scripts/train_cv.py --epochs 20 --batch-size 128
```

### Train Crisis Screening Classifier (C-SSRS):
```bash
python scripts/train_crisis.py
```

---

## 7. Web Application Sitemap & Pages

Once running at `http://localhost:5173` (or `http://localhost:3000` in Docker), access the following modules:

| Route | View Name | Capabilities |
| :--- | :--- | :--- |
| `/` | **Assistant** | Interactive conversational check-in with real-time text emotion detection, stressor attribution, and crisis screening override. |
| `/camera` | **Camera & Vision** | Real-time webcam / snapshot facial expression estimation with Grad-CAM visual heatmaps. |
| `/multimodal` | **Multimodal Fusion** | Weighted fusion ($W_{text} = 0.65, W_{vision} = 0.35$) of text and visual expression signals into normalized wellness categories. |
| `/mood` | **Mood Dashboard** | Longitudinal check-in history, valence trend charts, and 7-day time-series forecasting. |
| `/wellness` | **Wellness Center** | Evidence-based CBT reframing, 4-7-8 breathing exercises, and dynamic feedback adjustment loop. |
| `/privacy` | **Privacy & Consent** | Explicit camera and data consent controls, one-click JSON data export, and full history purge. |

---

## 8. Common Troubleshooting

| Issue | Root Cause | Solution |
| :--- | :--- | :--- |
| **MongoDB connection refused** | MongoDB service is not started | Run `net start MongoDB` or start container via `docker run -d -p 27017:27017 mongo:7.0`. |
| **CORS error in browser console** | Frontend origin not in CORS whitelist | Check `CORS_ORIGINS` in `.env`. Ensure `http://localhost:5173,http://localhost:3000` is present. |
| **`NO_FACE_DETECTED` error** | Input image has no front-facing face | Provide a clear, well-lit portrait photo facing directly toward the camera. |
| **Model loading slow on first run** | Transformers model cache initializing | DistilBERT and FER2013 load weights into memory on first inference. Subsequent calls run with low latency. |
