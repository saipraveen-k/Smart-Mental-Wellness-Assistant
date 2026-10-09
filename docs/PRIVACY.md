# Smart Mental Wellness Assistant — Privacy & Data Protection Policy

## 1. Privacy Architecture

The Smart Mental Wellness Assistant adheres to strict privacy-first guidelines for handling user data, mental wellness check-ins, and facial expression imagery.

---

## 2. Key Privacy Guarantees

### A. Zero Permanent Storage of Raw Camera Frames
- Webcam images captured for facial expression analysis are processed strictly in-memory during inference.
- Once facial expression probabilities are computed, raw image buffers are immediately discarded from memory.
- No facial images are ever written to disk or sent to third-party APIs.

### B. User Consent Controls
- **Mood Tracking Opt-In:** Users must explicitly grant consent to persist check-in history to the database.
- **Camera Access Consent:** Camera access requires browser permission prompt and explicit user toggle.

### C. Right to Erasure & Export
- **Export Data (`POST /api/privacy/export`):** Users can download their complete history of check-ins, interventions, and feedback in standard JSON format at any time.
- **Purge Data (`DELETE /api/privacy/data`):** Users can permanently delete all associated mood logs and account records with one click.

---

## 3. Database Security & Retention

- Passwords are encrypted using `bcrypt` with salt rounds.
- Session authentication utilizes secure HTTP JWT tokens.
- Sensitive conversations are stored in MongoDB with indexed access restricted to the authenticated user ID.
