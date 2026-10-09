"""Dataset Manager — dynamic Hugging Face dataset loading with validation.

Guarantees:
  * Dataset IDs come from configuration (``TEXT_DATASET_ID`` / ``CV_DATASET_ID``
    / ``CRISIS_DATASET_ID``), never from local CSV files.
  * Loading is dynamic (``datasets.load_dataset``) with retries and a clear
    error when the source is unreachable.
  * Structure and labels are validated before any consumer sees the data.
  * Provenance (dataset id, revision, REMOTE vs CACHED source) is recorded and
    exposed through ``/api/datasets/status``.
  * Statistics are computed from the loaded data — never hardcoded.
  * The manager NEVER silently substitutes a different dataset. A failed load
    raises :class:`DatasetUnavailableError`.
"""
from __future__ import annotations

import logging
import os
import pickle
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.config import settings
from app.ml.emotion_mapping import GOEMOTIONS_LABELS, FER2013_LABELS, EMOTION_MAPPING

logger = logging.getLogger("app.datasets")

RETRY_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = 2.0


class DatasetError(Exception):
    """Base class for dataset failures."""

    code = "DATASET_ERROR"

    def __init__(self, message: str, dataset_id: str = ""):
        super().__init__(message)
        self.message = message
        self.dataset_id = dataset_id


class DatasetUnavailableError(DatasetError):
    code = "DATASET_UNAVAILABLE"

    def __init__(self, dataset_id: str, detail: str = ""):
        msg = (
            f"Dataset unavailable: {dataset_id}. "
            "Please check network connectivity or the configured dataset revision."
        )
        if detail:
            msg = f"{msg} ({detail})"
        super().__init__(msg, dataset_id)


class DatasetValidationError(DatasetError):
    code = "DATASET_INVALID"

    def __init__(self, dataset_id: str, detail: str):
        super().__init__(f"Dataset validation failed for {dataset_id}: {detail}", dataset_id)


@dataclass
class DatasetProvenance:
    kind: str
    dataset_id: str
    revision: Optional[str]
    source: str  # "REMOTE" (downloaded now) or "CACHED" (local cache reused)
    cache_dir: str
    loaded_at: str
    loader: str  # "datasets.load_dataset" or "hub_file_loader"
    splits: Dict[str, int] = field(default_factory=dict)
    validation: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "dataset_id": self.dataset_id,
            "revision": self.revision,
            "source": self.source,
            "cache_dir": self.cache_dir,
            "loaded_at": self.loaded_at,
            "loader": self.loader,
            "splits": self.splits,
            "validation": self.validation,
        }


def _hub_cache_dir() -> Path:
    """Hub snapshot cache (raw files, .pt/.jsonl) — where scan_cache_dir looks."""
    if settings.hf_cache:
        return Path(settings.hf_cache)
    from huggingface_hub.constants import HF_HUB_CACHE

    return Path(HF_HUB_CACHE)


def _arrow_cache_dir() -> Path:
    """Arrow dataset cache produced by datasets.load_dataset()."""
    from datasets import config

    return Path(config.HF_DATASETS_CACHE)


def _cached_revisions(repo_id: str) -> List[str]:
    try:
        from huggingface_hub import scan_cache_dir

        out: List[str] = []
        for r in scan_cache_dir(_hub_cache_dir()).repos:
            if r.repo_id == repo_id:
                rev = getattr(r, "commit_hash", None) or getattr(r, "revision", None)
                if rev:
                    out.append(rev)
        return out
    except Exception:
        return []


def _is_cached(repo_id: str) -> bool:
    """True when a snapshot of this exact repo already exists locally."""
    return bool(_cached_revisions(repo_id))


def _cached_revision(repo_id: str) -> Optional[str]:
    revisions = _cached_revisions(repo_id)
    return revisions[-1] if revisions else None


def _remote_revision(repo_id: str) -> Optional[str]:
    """Best-effort revision lookup from the Hub API (None when offline)."""
    try:
        from huggingface_hub import HfApi

        return HfApi().dataset_info(repo_id).sha
    except Exception:
        return _cached_revision(repo_id)


def _looks_like_network_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    needles = (
        "connection", "timed out", "timeout", "temporary failure",
        "name resolution", "proxy", "ssl", "max retries", "unreachable",
        "http error 5", "http error 429",
    )
    return any(n in text for n in needles)


class DatasetManager:
    """Loads, validates, caches and reports on all datasets used by the app."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._loaded: Dict[str, Any] = {}
        self._provenance: Dict[str, DatasetProvenance] = {}

    # ------------------------------------------------------------------ #
    # Text dataset
    # ------------------------------------------------------------------ #
    def load_text_dataset(self, refresh: bool = False) -> Any:
        """Load the GoEmotions text dataset dynamically from the Hub."""
        return self._load("text", settings.text_dataset_id, self._load_text, refresh)

    def validate_text_dataset(self, dataset: Any) -> Dict[str, Any]:
        dataset_id = settings.text_dataset_id
        if not isinstance(dataset, dict) or not dataset:
            raise DatasetValidationError(dataset_id, "expected a mapping of splits")
        missing = [s for s in ("train",) if s not in dataset]
        if missing:
            raise DatasetValidationError(dataset_id, f"missing required split(s): {missing}")
        problems: List[str] = []
        rows: Dict[str, int] = {}
        for split, ds in dataset.items():
            rows[split] = len(ds)
            if len(ds) == 0:
                problems.append(f"split {split!r} is empty")
                continue
            cols = set(ds.column_names)
            if "text" not in cols:
                problems.append(f"split {split!r} has no 'text' column")
            missing_labels = [l for l in GOEMOTIONS_LABELS if l not in cols]
            if missing_labels:
                problems.append(f"split {split!r} missing emotion columns: {missing_labels[:5]}")
        if problems:
            raise DatasetValidationError(dataset_id, "; ".join(problems))
        return {
            "ok": True,
            "splits": rows,
            "required_columns": ["text"] + GOEMOTIONS_LABELS,
            "num_emotion_labels": len(GOEMOTIONS_LABELS),
        }

    # ------------------------------------------------------------------ #
    # CV dataset
    # ------------------------------------------------------------------ #
    def load_cv_dataset(self, refresh: bool = False) -> Dict[str, Any]:
        """Load the FER-2013 facial-expression dataset dynamically from the Hub.

        The configured repository (``Jeneral/fer-2013``) stores its data as
        pickled example files plus a legacy ``datasets`` loader script. Modern
        ``datasets`` releases no longer execute dataset scripts, so when
        ``load_dataset`` reports that, we fall back to a *hub file loader* that
        downloads the *same repository's* declared data files (``train.pt`` /
        ``test.pt``) and parses them exactly as the repository's loader script
        documents. The dataset ID never changes — if it cannot be loaded, we
        raise instead of substituting anything else.
        """
        return self._load("cv", settings.cv_dataset_id, self._load_cv, refresh)

    def validate_cv_dataset(self, dataset: Any) -> Dict[str, Any]:
        dataset_id = settings.cv_dataset_id
        if not isinstance(dataset, dict) or "train" not in dataset:
            raise DatasetValidationError(dataset_id, "expected splits including 'train'")
        problems: List[str] = []
        counts: Dict[str, int] = {}
        per_class: Dict[str, Dict[str, int]] = {}
        for split, payload in dataset.items():
            images = payload["images"]
            labels = payload["labels"]
            counts[split] = int(len(labels))
            if len(images) != len(labels):
                problems.append(f"split {split!r}: images/labels length mismatch")
            if len(labels) == 0:
                problems.append(f"split {split!r} is empty")
                continue
            if images.ndim != 3 or images.shape[1:] != (48, 48):
                problems.append(
                    f"split {split!r}: expected grayscale 48x48 images, got shape {images.shape}"
                )
            observed = sorted({int(x) for x in set(labels.tolist())})
            expected = list(range(len(FER2013_LABELS)))
            if observed != expected:
                problems.append(
                    f"split {split!r}: label ids {observed} do not match the 7 FER classes {expected}"
                )
            dist: Dict[str, int] = {}
            for idx in observed:
                dist[FER2013_LABELS[idx]] = int((labels == idx).sum())
            per_class[split] = dist
        if problems:
            raise DatasetValidationError(dataset_id, "; ".join(problems))
        return {
            "ok": True,
            "splits": counts,
            "classes": list(FER2013_LABELS),
            "image_size": [48, 48],
            "per_class": per_class,
        }

    # ------------------------------------------------------------------ #
    # Statistics
    # ------------------------------------------------------------------ #
    def get_dataset_statistics(self, kind: str) -> Dict[str, Any]:
        """Compute dataset statistics from the actual loaded data."""
        if kind == "text":
            ds = self.load_text_dataset()
            stats: Dict[str, Any] = {"kind": "text", "dataset_id": settings.text_dataset_id}
            label_counts = {l: 0 for l in GOEMOTIONS_LABELS}
            wellness_counts: Dict[str, int] = {}
            total = 0
            ambiguous = 0
            lengths: List[int] = []
            for split, part in ds.items():
                for row in part:
                    total += 1
                    text = row["text"] or ""
                    lengths.append(len(text))
                    active = [l for l in GOEMOTIONS_LABELS if row[l]]
                    for l in active:
                        label_counts[l] += 1
                    targets = {EMOTION_MAPPING[l] for l in active}
                    if len(targets) == 1:
                        cls = next(iter(targets))
                        wellness_counts[cls] = wellness_counts.get(cls, 0) + 1
                    elif len(targets) > 1:
                        ambiguous += 1
            lengths.sort()
            stats.update(
                {
                    "total_rows": total,
                    "splits": {k: len(v) for k, v in ds.items()},
                    "goemotions_label_counts": label_counts,
                    "wellness_class_counts": wellness_counts,
                    "ambiguous_rows": ambiguous,
                    "text_length": {
                        "min": lengths[0] if lengths else 0,
                        "max": lengths[-1] if lengths else 0,
                        "mean": round(sum(lengths) / len(lengths), 2) if lengths else 0,
                        "p95": lengths[int(0.95 * (len(lengths) - 1))] if lengths else 0,
                    },
                    "computed_at": datetime.now(timezone.utc).isoformat(),
                }
            )
            return stats

        if kind == "cv":
            ds = self.load_cv_dataset()
            stats = {"kind": "cv", "dataset_id": settings.cv_dataset_id}
            per_class: Dict[str, Dict[str, int]] = {}
            for split, payload in ds.items():
                labels = payload["labels"]
                per_class[split] = {
                    FER2013_LABELS[i]: int((labels == i).sum()) for i in range(len(FER2013_LABELS))
                }
            stats.update(
                {
                    "splits": {k: int(len(v["labels"])) for k, v in ds.items()},
                    "per_class": per_class,
                    "classes": list(FER2013_LABELS),
                    "computed_at": datetime.now(timezone.utc).isoformat(),
                }
            )
            return stats

        raise ValueError(f"Unknown dataset kind: {kind!r} (expected 'text' or 'cv')")

    # ------------------------------------------------------------------ #
    # Cache management
    # ------------------------------------------------------------------ #
    def cache_dataset(self, kind: str) -> Dict[str, Any]:
        """Ensure the dataset is present in the local cache (download if needed)."""
        if kind == "text":
            self.load_text_dataset()
            repo = settings.text_dataset_id
        elif kind == "cv":
            self.load_cv_dataset()
            repo = settings.cv_dataset_id
        else:
            raise ValueError(f"Unknown dataset kind: {kind!r}")
        return {"cached": True, "dataset_id": repo, "cache_dir": str(_hub_cache_dir())}

    def clear_dataset_cache(self, kind: str) -> Dict[str, Any]:
        """Delete the cached snapshot of this dataset (and any in-memory copy)."""
        if kind == "text":
            repo = settings.text_dataset_id
        elif kind == "cv":
            repo = settings.cv_dataset_id
        else:
            raise ValueError(f"Unknown dataset kind: {kind!r}")
        with self._lock:
            self._loaded.pop(kind, None)
            self._provenance.pop(kind, None)
        removed = 0
        try:
            from huggingface_hub import scan_cache_dir
            from huggingface_hub.delete_cache import delete_repo_cache

            for repo_info in scan_cache_dir(_hub_cache_dir()).repos:
                rev = getattr(repo_info, "commit_hash", None) or getattr(repo_info, "revision", None)
                if repo_info.repo_id == repo and rev:
                    delete_repo_cache(revision=rev)
                    removed += 1
        except Exception as exc:  # cache deletion is best-effort
            logger.warning("clear_dataset_cache(%s) partial failure: %s", kind, exc)
        # Also drop the arrow cache written by datasets.load_dataset().
        try:
            import shutil

            arrow_root = _arrow_cache_dir()
            for entry in arrow_root.glob(f"*--{repo.split('/')[0]}--{repo.split('/')[1]}*"):
                shutil.rmtree(entry, ignore_errors=True)
                removed += 1
        except Exception as exc:
            logger.warning("clear_dataset_cache(%s) arrow cache failure: %s", kind, exc)
        logger.info("dataset cache cleared kind=%s repo=%s removed_revisions=%d", kind, repo, removed)
        return {"cleared": True, "dataset_id": repo, "removed_revisions": removed}

    def get_status(self) -> Dict[str, Any]:
        """Provenance of every dataset this manager has loaded (lazy: no load)."""
        return {
            "cache_dir": str(_hub_cache_dir()),
            "datasets": {
                kind: prov.as_dict() for kind, prov in sorted(self._provenance.items())
            },
            "configured": {
                "text": settings.text_dataset_id,
                "cv": settings.cv_dataset_id,
                "crisis": settings.crisis_dataset_id,
                "crisis_control": settings.crisis_control_dataset_id,
                "revision": settings.hf_dataset_revision,
            },
        }

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _load(self, kind: str, repo_id: str, loader, refresh: bool) -> Any:
        with self._lock:
            if kind in self._loaded and not refresh:
                return self._loaded[kind]

        was_cached = _is_cached(repo_id)
        loader_name = "datasets.load_dataset"
        last_error: Optional[BaseException] = None

        for attempt in range(1, RETRY_ATTEMPTS + 1):
            try:
                data, loader_name = loader(repo_id, refresh)
                break
            except DatasetError:
                raise
            except Exception as exc:  # noqa: BLE001 - normalised below
                last_error = exc
                if _looks_like_network_error(exc) and attempt < RETRY_ATTEMPTS:
                    wait = RETRY_BACKOFF_SECONDS * attempt
                    logger.warning(
                        "dataset load attempt %d/%d failed (%s); retrying in %.0fs",
                        attempt, RETRY_ATTEMPTS, exc, wait,
                    )
                    time.sleep(wait)
                    continue
                if _looks_like_network_error(exc):
                    raise DatasetUnavailableError(repo_id, "network error") from exc
                raise DatasetUnavailableError(repo_id, str(exc)[:300]) from exc
        else:  # pragma: no cover - defensive
            raise DatasetUnavailableError(repo_id, str(last_error or ""))

        source = "CACHED" if was_cached and not refresh else "REMOTE"
        if refresh:
            source = "REMOTE (forced refresh)"

        revision = _remote_revision(repo_id)
        prov = DatasetProvenance(
            kind=kind,
            dataset_id=repo_id,
            revision=revision,
            source=source,
            cache_dir=str(_hub_cache_dir()),
            loaded_at=datetime.now(timezone.utc).isoformat(),
            loader=loader_name,
        )
        if kind == "text":
            prov.splits = {k: len(v) for k, v in data.items()}
            prov.validation = self.validate_text_dataset(data)
        elif kind == "cv":
            prov.splits = {k: int(len(v["labels"])) for k, v in data.items()}
            prov.validation = self.validate_cv_dataset(data)

        logger.info(
            "dataset loaded kind=%s id=%s revision=%s source=%s loader=%s splits=%s",
            kind, repo_id, revision, source, loader_name, prov.splits,
        )
        with self._lock:
            self._loaded[kind] = data
            self._provenance[kind] = prov
        return data

    # -- loaders -------------------------------------------------------- #
    def _load_text(self, repo_id: str, refresh: bool):
        from datasets import load_dataset

        if refresh:
            data = load_dataset(
                repo_id,
                revision=settings.hf_dataset_revision,
                download_mode="force_redownload",
                cache_dir=settings.hf_cache or None,
            )
        else:
            data = load_dataset(
                repo_id,
                revision=settings.hf_dataset_revision,
                cache_dir=settings.hf_cache or None,
            )
        if not isinstance(data, dict) and hasattr(data, "keys"):
            data = {k: data[k] for k in data.keys()}
        return data, "datasets.load_dataset"

    def _load_cv(self, repo_id: str, refresh: bool):
        # 1) Preferred path: the datasets library.
        try:
            from datasets import load_dataset

            data = load_dataset(
                repo_id,
                revision=settings.hf_dataset_revision,
                cache_dir=settings.hf_cache or None,
                download_mode="force_redownload" if refresh else "reuse_dataset_if_exists",
            )
            return self._normalize_cv_splits(data), "datasets.load_dataset"
        except DatasetError:
            raise
        except Exception as exc:  # noqa: BLE001
            msg = str(exc).lower()
            script_unsupported = (
                "dataset scripts are no longer supported" in msg
                or "pandas-rows scripts" in msg
                or "loading a dataset script" in msg
            )
            if not script_unsupported:
                # Genuine failure (network, missing repo, wrong id) — do NOT
                # substitute a different dataset.
                raise

        # 2) Fallback: same repository, its own documented data files.
        logger.info(
            "load_dataset cannot execute the dataset script for %s — "
            "using hub file loader for the SAME configured repository.", repo_id,
        )
        return self._load_cv_from_repo_files(repo_id), "hub_file_loader"

    def _load_cv_from_repo_files(self, repo_id: str) -> Dict[str, Any]:
        import numpy as np

        from app.cv.image_utils import decode_face_image

        files = {"train": "train.pt", "test": "test.pt"}
        splits: Dict[str, Any] = {}
        for split, filename in files.items():
            try:
                from huggingface_hub import hf_hub_download

                path = hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    repo_type="dataset",
                    revision=settings.hf_dataset_revision,
                )
            except Exception as exc:  # noqa: BLE001
                if _looks_like_network_error(exc):
                    raise DatasetUnavailableError(repo_id, "network error") from exc
                raise DatasetUnavailableError(repo_id, f"could not fetch {filename}") from exc

            # The repository's own loader script reads these files with pickle.load.
            with open(path, "rb") as fh:
                examples = pickle.load(fh)
            if not isinstance(examples, list) or not examples:
                raise DatasetValidationError(repo_id, f"{filename} did not contain a non-empty example list")

            images = np.zeros((len(examples), 48, 48), dtype=np.uint8)
            labels = np.zeros((len(examples),), dtype=np.int64)
            class_to_id = {name: i for i, name in enumerate(FER2013_LABELS)}
            for i, ex in enumerate(examples):
                if not isinstance(ex, dict) or "img_bytes" not in ex or "labels" not in ex:
                    raise DatasetValidationError(
                        repo_id, f"{filename}: example {i} does not match the documented schema"
                    )
                label = ex["labels"]
                if isinstance(label, (int, float)):
                    label = FER2013_LABELS[int(label)]
                if label not in class_to_id:
                    raise DatasetValidationError(repo_id, f"{filename}: unknown label {label!r} at index {i}")
                img = decode_face_image(ex["img_bytes"])
                if img is None:
                    raise DatasetValidationError(repo_id, f"{filename}: undecodable image at index {i}")
                images[i] = img
                labels[i] = class_to_id[label]
            splits[split] = {"images": images, "labels": labels}
        return splits

    @staticmethod
    def _normalize_cv_splits(data: Any) -> Dict[str, Any]:
        """Convert a datasets.DatasetDict of img_bytes/labels rows to arrays."""
        import numpy as np

        from app.cv.image_utils import decode_face_image

        out: Dict[str, Any] = {}
        class_to_id = {name: i for i, name in enumerate(FER2013_LABELS)}
        for split in ("train", "test", "validation"):
            if split not in data:
                continue
            ds = data[split]
            labels_raw = ds["labels"] if "labels" in ds.column_names else ds["label"]
            images = np.zeros((len(ds), 48, 48), dtype=np.uint8)
            labels = np.zeros((len(ds),), dtype=np.int64)
            img_col = ds["img_bytes"] if "img_bytes" in ds.column_names else ds["image"]
            for i, (b, lab) in enumerate(zip(img_col, labels_raw)):
                if isinstance(lab, int):
                    lab = FER2013_LABELS[lab]
                img = decode_face_image(b)
                if img is None:
                    raise DatasetValidationError(settings.cv_dataset_id, f"undecodable image at index {i}")
                images[i] = img
                labels[i] = class_to_id[lab]
            out[split] = {"images": images, "labels": labels}
        return out


# Module-level singleton used by the API and training scripts.
dataset_manager = DatasetManager()
