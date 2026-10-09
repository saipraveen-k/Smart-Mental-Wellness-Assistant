# Smart Mental Wellness Assistant — Test Strategy & Automated Verification

## Overview

The application features comprehensive automated testing covering unit logic, ML dataset loaders, CV image preprocessing, multimodal fusion, crisis safety triggers, API routers, security policies, and end-to-end integration workflows.

---

## Test Suites Summary

| Test Suite | File Location | Scope |
| :--- | :--- | :--- |
| **Unit Tests** | `backend/tests/test_unit.py` | Emotion taxonomy mapping, dataset validation, text cleaning, CV face crop, fusion math, stressor rules, forecasting bounds |
| **API Endpoint Tests** | `backend/tests/test_api.py` | Health endpoints, auth flow, text analysis, vision analysis, multimodal fusion, crisis triggers, mood history, feedback |
| **Security Tests** | `backend/tests/test_security.py` | Auth failures, token expiration, payload size limits, image MIME type validation, path traversal defense |
| **Integration Tests** | `backend/tests/test_integration.py` | Multi-step user workflow (Text/Image $\to$ Fusion $\to$ Crisis check $\to$ DB record $\to$ Intervention $\to$ Feedback) |
| **Health Check Script** | `scripts/health_check.py` | End-to-end system status check verifying Backend, DB, ML Models, Datasets, Auth, and Frontend readiness |

---

## Running Automated Tests

```bash
# Run pytest test suite
cd backend
pytest -v

# Run system health check script
python scripts/health_check.py
```
