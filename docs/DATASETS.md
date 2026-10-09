# Smart Mental Wellness Assistant — Dataset Documentation & Provenance

## Dynamic Loading Policy

> [!IMPORTANT]
> **No Hardcoded CSV Dependencies:** The Smart Mental Wellness Assistant **never** requires manual download of dataset ZIP files or static CSV files. All datasets are dynamically fetched from the Hugging Face Hub using the `datasets` library or direct Hub API streams, verified upon ingestion, and cached locally for offline development.

---

## 1. Text Dataset: GoEmotions

- **Source Identifier:** `SetFit/go_emotions` (configurable via `TEXT_DATASET_ID` in `.env`)
- **Repository Link:** [Hugging Face GoEmotions](https://huggingface.co/datasets/SetFit/go_emotions)
- **Original Paper:** Demszky et al., *GoEmotions: A Dataset of Fine-Grained Emotions*, ACL 2020.
- **Data Description:** 58,009 Reddit comments annotated with 27 emotion categories plus `neutral`.
- **Predefined Splits:**
  - `train`: 43,410 samples
  - `validation`: 5,426 samples
  - `test`: 5,427 samples

### Mapping to Normalized Wellness Taxonomy

To ensure clinically grounded and consistent application logic, fine-grained GoEmotions labels are mapped to a 5-class wellness taxonomy:

| GoEmotions Labels | Normalized Wellness Category | Category Index |
| :--- | :--- | :---: |
| `joy`, `amusement`, `approval`, `caring`, `desire`, `excitement`, `gratitude`, `love`, `optimism`, `pride`, `relief` | **Joy** | 0 |
| `sadness`, `disappointment`, `grief`, `remorse` | **Sadness** | 1 |
| `anger`, `annoyance`, `disapproval` | **Anger** | 2 |
| `fear`, `nervousness`, `embarrassment` | **Anxiety** | 3 |
| `neutral`, `curiosity`, `realization`, `confusion`, `surprise` | **Neutral / Other** | 4 |

---

## 2. Computer Vision Dataset: FER-2013

- **Source Identifier:** `Jeneral/fer-2013` (configurable via `CV_DATASET_ID` in `.env`)
- **Repository Link:** [Hugging Face FER-2013](https://huggingface.co/datasets/Jeneral/fer-2013)
- **Data Description:** 35,887 48x48 pixel grayscale face images categorized into seven facial-expression classes.
- **Classes:** `angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, `surprise`.
- **Predefined Splits:**
  - `train`: 28,709 images
  - `test`: 7,178 images

### Vision Mapping to Normalized Wellness Taxonomy

| FER-2013 Class | Normalized Wellness Category | Notes |
| :--- | :--- | :--- |
| `happy` | **Joy** | High correlation with positive affect |
| `sad` | **Sadness** | Lowered facial muscle tone |
| `angry` | **Anger** | Brow furrowing, tightened lips |
| `fear` | **Anxiety** | Wide open eyes, elevated brows |
| `neutral` | **Neutral** | Resting facial position |
| `disgust`, `surprise` | **Neutral / Other** | Mapped to general/other bucket |

---

## 3. Crisis Detection Dataset: C-SSRS Suicide Watch Posts

- **Source Identifier:** `av9ash/CSSR-S_labelled_suicidewatch_posts_reddit` (configurable via `CRISIS_DATASET_ID`)
- **Data Description:** Reddit posts annotated using the Columbia Suicide Severity Rating Scale (C-SSRS) (severities 0 through 6).
- **Control Dataset:** Sampled everyday non-crisis text from `SetFit/go_emotions` (validation/test splits) to prevent over-prediction on standard sadness or anxiety.

### Severity Mapping

- `0` + Control everyday text $\rightarrow$ **LOW**
- `1`, `2`, `3` (Ideation without specific plan) $\rightarrow$ **MODERATE**
- `4`, `5`, `6` (Explicit plan, intent, or behaviour) $\rightarrow$ **HIGH**

---

## Dataset Manager Architecture

The dataset loader component (`app/ml/dataset_manager.py`) provides:

```python
dataset_manager.load_text_dataset()
dataset_manager.load_cv_dataset()
dataset_manager.validate_text_dataset(dataset)
dataset_manager.validate_cv_dataset(dataset)
dataset_manager.get_dataset_statistics()
dataset_manager.clear_dataset_cache()
```

### Health & Provenance Endpoint

Dataset provenance status can be queried at runtime via `/api/datasets/status`:

```json
{
  "success": true,
  "data": {
    "text": {
      "dataset_id": "SetFit/go_emotions",
      "source": "REMOTE",
      "revision": "b83b20869f3b5eff89c0a375d8d015a878826094"
    },
    "cv": {
      "dataset_id": "Jeneral/fer-2013",
      "source": "REMOTE",
      "revision": "e9a03b5161427ad6e1b7b719bc42bb271b86d9a1"
    }
  }
}
```
