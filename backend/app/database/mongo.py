"""MongoDB access layer (pymongo, synchronous).

Every collection indexes user_id / timestamp / created_at as required.
Raw camera frames are never stored. Users can delete or export all their
data through the privacy endpoints.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pymongo import ASCENDING, DESCENDING, MongoClient
from pymongo.database import Database
from pymongo.errors import PyMongoError

from app.config import settings

logger = logging.getLogger("app.database")

COLLECTIONS = [
    "users", "mood_records", "interventions", "feedback",
    "crisis_events", "model_predictions", "dataset_runs", "audit_logs",
]


class DatabaseUnavailableError(Exception):
    code = "DATABASE_UNAVAILABLE"


class Database:
    def __init__(self) -> None:
        self._client: Optional[MongoClient] = None

    def client(self) -> MongoClient:
        if self._client is None:
            self._client = MongoClient(
                settings.mongodb_uri,
                serverSelectionTimeoutMS=settings.mongodb_timeout_ms,
            )
        return self._client

    def db(self) -> Database:
        return self.client()[settings.mongodb_database]

    def ping(self) -> bool:
        try:
            self.client().admin.command("ping")
            return True
        except PyMongoError as exc:
            logger.error("mongodb ping failed: %s", exc)
            return False

    def ensure_indexes(self) -> None:
        db = self.db()
        db["users"].create_index([("email", ASCENDING)], unique=True)
        db["mood_records"].create_index([("user_id", ASCENDING), ("timestamp", DESCENDING)])
        db["mood_records"].create_index([("created_at", DESCENDING)])
        db["feedback"].create_index([("user_id", ASCENDING), ("created_at", DESCENDING)])
        db["feedback"].create_index([("user_id", ASCENDING), ("intervention_id", ASCENDING)])
        db["interventions"].create_index([("user_id", ASCENDING), ("created_at", DESCENDING)])
        db["crisis_events"].create_index([("user_id", ASCENDING), ("created_at", DESCENDING)])
        db["model_predictions"].create_index([("user_id", ASCENDING), ("created_at", DESCENDING)])
        db["model_predictions"].create_index([("kind", ASCENDING), ("created_at", DESCENDING)])
        db["dataset_runs"].create_index([("kind", ASCENDING), ("loaded_at", DESCENDING)])
        db["audit_logs"].create_index([("request_id", ASCENDING)])
        db["audit_logs"].create_index([("created_at", DESCENDING)])
        logger.info("mongodb indexes ensured on database %s", settings.mongodb_database)

    # ------------------------------------------------------------------ #
    # Users
    # ------------------------------------------------------------------ #
    def create_user(self, email: str, password_hash: str, name: str) -> Dict[str, Any]:
        now = datetime.now(timezone.utc)
        doc = {
            "email": email.lower(),
            "password_hash": password_hash,
            "name": name,
            "created_at": now,
            "consent": {
                "save_mood_history": True,   # explicit opt-in at first check-in
                "camera_access": False,
                "store_images": False,
                "consent_updated_at": now,
            },
        }
        try:
            res = self.db()["users"].insert_one(doc)
        except PyMongoError as exc:
            if "duplicate key" in str(exc).lower():
                raise ValueError("EMAIL_ALREADY_REGISTERED") from exc
            raise DatabaseUnavailableError(str(exc)) from exc
        doc["id"] = str(res.inserted_id)
        return doc

    def find_user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        doc = self.db()["users"].find_one({"email": email.lower()})
        if doc:
            doc["id"] = str(doc["_id"])
        return doc

    def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        from bson import ObjectId
        from bson.errors import InvalidId

        try:
            doc = self.db()["users"].find_one({"_id": ObjectId(user_id)})
        except InvalidId:
            return None
        if doc:
            doc["id"] = str(doc["_id"])
        return doc

    def update_consent(self, user_id: str, consent: Dict[str, bool]) -> Optional[Dict[str, Any]]:
        from bson import ObjectId

        consent = dict(consent)
        consent["consent_updated_at"] = datetime.now(timezone.utc)
        self.db()["users"].update_one(
            {"_id": ObjectId(user_id)}, {"$set": {"consent": consent}}
        )
        user = self.get_user(user_id)
        return user["consent"] if user else None

    # ------------------------------------------------------------------ #
    # Mood records
    # ------------------------------------------------------------------ #
    def insert_mood_record(self, record: Dict[str, Any]) -> str:
        doc = dict(record)
        doc["created_at"] = datetime.now(timezone.utc)
        try:
            res = self.db()["mood_records"].insert_one(doc)
        except PyMongoError as exc:
            raise DatabaseUnavailableError(str(exc)) from exc
        return str(res.inserted_id)

    def mood_history(self, user_id: str, limit: int = 500) -> List[Dict[str, Any]]:
        cursor = (
            self.db()["mood_records"]
            .find({"user_id": user_id})
            .sort("timestamp", DESCENDING)
            .limit(limit)
        )
        out = []
        for doc in cursor:
            doc["id"] = str(doc.pop("_id"))
            doc["timestamp"] = doc["timestamp"].isoformat()
            doc["created_at"] = doc["created_at"].isoformat()
            out.append(doc)
        return out

    # ------------------------------------------------------------------ #
    # Feedback / interventions
    # ------------------------------------------------------------------ #
    def insert_feedback(self, feedback: Dict[str, Any]) -> str:
        doc = dict(feedback)
        doc["created_at"] = datetime.now(timezone.utc)
        res = self.db()["feedback"].insert_one(doc)
        return str(res.inserted_id)

    def feedback_stats(self, user_id: str) -> Dict[str, Dict[str, Any]]:
        """Aggregate per-intervention feedback -> real learning signal."""
        from app.services.intervention_service import FEEDBACK_VALUES

        stats: Dict[str, Dict[str, Any]] = {}
        cursor = self.db()["feedback"].find({"user_id": user_id})
        for doc in cursor:
            iid = doc.get("intervention_id")
            rating = doc.get("rating")
            value = FEEDBACK_VALUES.get(rating)
            if value is None:
                continue
            bucket = stats.setdefault(iid, {"total": 0, "sum": 0.0, "ratings": {}})
            bucket["total"] += 1
            bucket["sum"] += value
            bucket["ratings"][rating] = bucket["ratings"].get(rating, 0) + 1
        for iid, b in stats.items():
            b["mean_score"] = round(b["sum"] / b["total"], 4) if b["total"] else 0.0
        return stats

    def insert_intervention(self, doc: Dict[str, Any]) -> str:
        payload = dict(doc)
        payload["created_at"] = datetime.now(timezone.utc)
        res = self.db()["interventions"].insert_one(payload)
        return str(res.inserted_id)

    # ------------------------------------------------------------------ #
    # Crisis events / predictions / dataset runs / audit
    # ------------------------------------------------------------------ #
    def insert_crisis_event(self, doc: Dict[str, Any]) -> str:
        payload = dict(doc)
        payload["created_at"] = datetime.now(timezone.utc)
        res = self.db()["crisis_events"].insert_one(payload)
        return str(res.inserted_id)

    def insert_model_prediction(self, doc: Dict[str, Any]) -> str:
        payload = dict(doc)
        payload["created_at"] = datetime.now(timezone.utc)
        res = self.db()["model_predictions"].insert_one(payload)
        return str(res.inserted_id)

    def insert_dataset_run(self, doc: Dict[str, Any]) -> str:
        payload = dict(doc)
        payload.setdefault("created_at", datetime.now(timezone.utc))
        res = self.db()["dataset_runs"].insert_one(payload)
        return str(res.inserted_id)

    def insert_audit_log(self, doc: Dict[str, Any]) -> None:
        payload = dict(doc)
        payload.setdefault("created_at", datetime.now(timezone.utc))
        try:
            self.db()["audit_logs"].insert_one(payload)
        except PyMongoError as exc:  # audit failure must not break requests
            logger.warning("audit log insert failed: %s", exc)

    # ------------------------------------------------------------------ #
    # Privacy: delete / export
    # ------------------------------------------------------------------ #
    def delete_user_data(self, user_id: str) -> Dict[str, int]:
        """Delete all personal data; keep only an anonymized consent record."""
        db = self.db()
        counts: Dict[str, int] = {}
        for coll in ("mood_records", "feedback", "interventions", "crisis_events", "model_predictions"):
            res = db[coll].delete_many({"user_id": user_id})
            counts[coll] = res.deleted_count
        # Remove the account itself (deleting data must remove the identity).
        from bson import ObjectId

        res = db["users"].delete_one({"_id": ObjectId(user_id)})
        counts["users"] = res.deleted_count
        logger.info("user data deleted user_id=%s counts=%s", user_id, counts)
        return counts

    def export_user_data(self, user_id: str) -> Dict[str, Any]:
        db = self.db()
        user = self.get_user(user_id)
        if user:
            user.pop("password_hash", None)
            user.pop("_id", None)
        def _dump(coll: str) -> List[Dict[str, Any]]:
            out = []
            for doc in db[coll].find({"user_id": user_id}):
                doc["id"] = str(doc.pop("_id"))
                for k, v in list(doc.items()):
                    if isinstance(v, datetime):
                        doc[k] = v.isoformat()
                out.append(doc)
            return out

        return {
            "user": user,
            "mood_records": _dump("mood_records"),
            "feedback": _dump("feedback"),
            "interventions": _dump("interventions"),
            "crisis_events": _dump("crisis_events"),
            "model_predictions": _dump("model_predictions"),
            "exported_at": datetime.now(timezone.utc).isoformat(),
        }


database = Database()
