# Smart Mental Wellness Assistant — REST API Reference

All API requests and responses adhere to standard JSON envelopes:

### Success Envelope (HTTP 200 / 201)
```json
{
  "success": true,
  "data": { ... },
  "error": null
}
```

### Error Envelope (HTTP 4xx / 5xx)
```json
{
  "success": false,
  "data": null,
  "error": {
    "code": "ERROR_CODE",
    "message": "User-friendly error message",
    "detail": "Additional diagnostic detail"
  }
}
```

---

## Endpoint Catalog

### System & Health
- `GET /health` — Health check endpoint reporting service, database, and loaded model readiness.
- `GET /api/models/status` — Model loading and inference device status.
- `GET /api/datasets/status` — Dataset provenance (ID, source, revision, splits).
- `POST /api/datasets/refresh` — Cache invalidation and dataset reloading.
- `GET /api/safety/resources` — Emergency hotline directory by country.

### Authentication
- `POST /api/auth/register` — User registration (`email`, `password`, `full_name`).
- `POST /api/auth/login` — Authenticate and retrieve JWT Bearer token.

### Emotion & Vision Analysis
- `POST /api/emotion/text` — Single text analysis for emotion probabilities, taxonomy, and word attributions.
- `POST /api/vision/analyze` — Image upload/base64 facial expression detection & confidence breakdown.
- `POST /api/multimodal/analyze` — Fused text + vision analysis with configurable modality weights.

### Crisis & Safety
- `POST /api/crisis/analyze` — Dedicated crisis screening returning `LOW`, `MODERATE`, or `HIGH` risk level.

### Stressors & Interventions
- `POST /api/stressor/analyze` — Detect life domain stressors (Academic, Work, Social, etc.).
- `GET /api/interventions` — Query personalized intervention recommendations based on emotion and stressor.
- `POST /api/interventions/feedback` — Submit feedback on intervention effectiveness (`Much better`, `A little better`, `No change`, `Worse`).

### Mood History & Forecasting
- `GET /api/mood/history` — Query authenticated user's longitudinal mood logs.
- `GET /api/mood/trends` — Aggregate emotion distribution and weekly trends.
- `GET /api/mood/forecast` — Time-series forecast for future mood trajectory (requires minimum historical observations).

### Privacy & Consent
- `GET /api/privacy/settings` — Get user privacy preferences.
- `POST /api/privacy/settings` — Update consent status (mood recording, camera usage).
- `POST /api/privacy/export` — Export user history in compliant JSON format.
- `DELETE /api/privacy/data` — Purge user records and history.
