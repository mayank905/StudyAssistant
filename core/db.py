"""SQLite Persistence & Analytics Engine for StudyAssistant.

Handles persistent tracking of:
- Timed MCQ Mock Test attempts and question-level breakdown
- Oral Viva examiner session evaluations and rubric scores
- Flashcards and Spaced Repetition (SM-2 algorithm) mastery levels
- Study sessions and time-tracking metrics
- JSON backup export / import
- Downloadable Markdown / Text summary performance reports
"""

import json
import os
import sqlite3
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union


DEFAULT_DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "study_assistant.db",
)


class Database:
    """Thread-safe SQLite Database Manager for StudyAssistant."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or DEFAULT_DB_PATH
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Create a new SQLite connection with foreign keys and row factory."""
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def _init_db(self) -> None:
        """Create database tables and performance indexes if not present."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # 1. Quiz Attempts Table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS quiz_attempts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    subject TEXT NOT NULL,
                    chapter TEXT DEFAULT '',
                    score REAL NOT NULL,
                    total_questions INTEGER NOT NULL,
                    correct_answers INTEGER NOT NULL,
                    time_taken_seconds INTEGER DEFAULT 0,
                    difficulty TEXT DEFAULT 'medium',
                    questions_data TEXT DEFAULT '[]',
                    weak_topics TEXT DEFAULT '[]'
                )
                """
            )

            # 2. Oral Viva Evaluations Table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS viva_evaluations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    subject TEXT NOT NULL,
                    topic TEXT DEFAULT '',
                    question TEXT NOT NULL,
                    student_response TEXT NOT NULL,
                    score REAL NOT NULL,
                    accuracy_score REAL NOT NULL,
                    completeness_score REAL NOT NULL,
                    key_terms_score REAL NOT NULL,
                    feedback TEXT DEFAULT '',
                    spoken_feedback TEXT DEFAULT '',
                    missed_points TEXT DEFAULT '[]',
                    correct_points TEXT DEFAULT '[]',
                    citations TEXT DEFAULT '[]'
                )
                """
            )

            # 3. Flashcards & Spaced Repetition Table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS flashcards (
                    id TEXT PRIMARY KEY,
                    subject TEXT NOT NULL,
                    chapter TEXT DEFAULT '',
                    category TEXT DEFAULT 'definition',
                    front TEXT NOT NULL,
                    back TEXT NOT NULL,
                    citations TEXT DEFAULT '',
                    mastery_level TEXT DEFAULT 'unseen',
                    interval_days INTEGER DEFAULT 1,
                    ease_factor REAL DEFAULT 2.5,
                    repetition_number INTEGER DEFAULT 0,
                    last_reviewed TEXT DEFAULT NULL,
                    next_review_due TEXT DEFAULT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )

            # 4. Study Sessions Table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS study_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_type TEXT NOT NULL,
                    subject TEXT NOT NULL,
                    start_time TEXT NOT NULL,
                    end_time TEXT DEFAULT NULL,
                    duration_seconds INTEGER DEFAULT 0,
                    metadata TEXT DEFAULT '{}'
                )
                """
            )

            # Performance Indexes
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_quiz_subject ON quiz_attempts(subject, timestamp);"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_viva_subject ON viva_evaluations(subject, timestamp);"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_flashcards_subject ON flashcards(subject, mastery_level);"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_flashcards_due ON flashcards(next_review_due);"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_sessions_type ON study_sessions(session_type, start_time);"
            )

            conn.commit()

    # -------------------------------------------------------------------------
    # Quiz Attempts Persistence
    # -------------------------------------------------------------------------
    def log_quiz_attempt(
        self,
        subject: str,
        score: float,
        total_questions: int,
        correct_answers: int,
        chapter: str = "",
        time_taken_seconds: int = 0,
        difficulty: str = "medium",
        questions_data: Optional[List[Dict[str, Any]]] = None,
        weak_topics: Optional[List[str]] = None,
        timestamp: Optional[str] = None,
    ) -> int:
        """Log a completed MCQ mock test attempt. Returns inserted attempt ID."""
        ts = timestamp or datetime.now().isoformat()
        q_json = json.dumps(questions_data or [])
        w_json = json.dumps(weak_topics or [])

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO quiz_attempts (
                    timestamp, subject, chapter, score, total_questions,
                    correct_answers, time_taken_seconds, difficulty,
                    questions_data, weak_topics
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ts,
                    subject,
                    chapter,
                    float(score),
                    int(total_questions),
                    int(correct_answers),
                    int(time_taken_seconds),
                    difficulty,
                    q_json,
                    w_json,
                ),
            )
            conn.commit()
            return cursor.lastrowid

    def get_quiz_history(
        self, limit: int = 50, subject: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve recent quiz attempts, optionally filtered by subject."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if subject and subject.lower() != "all":
                cursor.execute(
                    """
                    SELECT * FROM quiz_attempts
                    WHERE subject = ?
                    ORDER BY id DESC LIMIT ?
                    """,
                    (subject, limit),
                )
            else:
                cursor.execute(
                    """
                    SELECT * FROM quiz_attempts
                    ORDER BY id DESC LIMIT ?
                    """,
                    (limit,),
                )
            rows = cursor.fetchall()
            results = []
            for r in rows:
                item = dict(r)
                item["questions_data"] = json.loads(item.get("questions_data") or "[]")
                item["weak_topics"] = json.loads(item.get("weak_topics") or "[]")
                results.append(item)
            return results

    # -------------------------------------------------------------------------
    # Oral Viva Evaluations Persistence
    # -------------------------------------------------------------------------
    def log_viva_evaluation(
        self,
        subject: str,
        question: str,
        student_response: str,
        score: float,
        accuracy_score: float,
        completeness_score: float,
        key_terms_score: float,
        topic: str = "",
        feedback: str = "",
        spoken_feedback: str = "",
        missed_points: Optional[List[str]] = None,
        correct_points: Optional[List[str]] = None,
        citations: Optional[List[str]] = None,
        timestamp: Optional[str] = None,
    ) -> int:
        """Log an oral viva session evaluation. Returns inserted viva ID."""
        ts = timestamp or datetime.now().isoformat()
        missed_json = json.dumps(missed_points or [])
        correct_json = json.dumps(correct_points or [])
        citations_json = json.dumps(citations or [])

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO viva_evaluations (
                    timestamp, subject, topic, question, student_response,
                    score, accuracy_score, completeness_score, key_terms_score,
                    feedback, spoken_feedback, missed_points, correct_points, citations
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ts,
                    subject,
                    topic,
                    question,
                    student_response,
                    float(score),
                    float(accuracy_score),
                    float(completeness_score),
                    float(key_terms_score),
                    feedback,
                    spoken_feedback,
                    missed_json,
                    correct_json,
                    citations_json,
                ),
            )
            conn.commit()
            return cursor.lastrowid

    def get_viva_history(
        self, limit: int = 50, subject: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve recent viva evaluations, optionally filtered by subject."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if subject and subject.lower() != "all":
                cursor.execute(
                    """
                    SELECT * FROM viva_evaluations
                    WHERE subject = ?
                    ORDER BY id DESC LIMIT ?
                    """,
                    (subject, limit),
                )
            else:
                cursor.execute(
                    """
                    SELECT * FROM viva_evaluations
                    ORDER BY id DESC LIMIT ?
                    """,
                    (limit,),
                )
            rows = cursor.fetchall()
            results = []
            for r in rows:
                item = dict(r)
                item["missed_points"] = json.loads(item.get("missed_points") or "[]")
                item["correct_points"] = json.loads(item.get("correct_points") or "[]")
                item["citations"] = json.loads(item.get("citations") or "[]")
                results.append(item)
            return results

    # -------------------------------------------------------------------------
    # Flashcards & Spaced Repetition (SuperMemo SM-2)
    # -------------------------------------------------------------------------
    def upsert_flashcard(
        self,
        front: str,
        back: str,
        subject: str,
        chapter: str = "",
        category: str = "definition",
        citations: str = "",
        card_id: Optional[str] = None,
        mastery_level: str = "unseen",
    ) -> str:
        """Insert or update a flashcard. Returns card ID."""
        cid = card_id or str(uuid.uuid4())[:8]
        now = datetime.now().isoformat()

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO flashcards (
                    id, subject, chapter, category, front, back, citations,
                    mastery_level, interval_days, ease_factor, repetition_number,
                    last_reviewed, next_review_due, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, 2.5, 0, NULL, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    front = excluded.front,
                    back = excluded.back,
                    category = excluded.category,
                    citations = excluded.citations,
                    chapter = excluded.chapter
                """,
                (cid, subject, chapter, category, front, back, citations, mastery_level, now, now),
            )
            conn.commit()
            return cid

    def save_flashcards_batch(self, cards: List[Dict[str, Any]]) -> int:
        """Batch save flashcards, ignoring duplicates with matching front/subject."""
        count = 0
        with self._get_connection() as conn:
            cursor = conn.cursor()
            now = datetime.now().isoformat()
            for c in cards:
                cid = c.get("id") or str(uuid.uuid4())[:8]
                subject = c.get("subject", "General")
                chapter = c.get("chapter", "")
                category = c.get("category", "definition")
                front = c.get("front", "").strip()
                back = c.get("back", "").strip()
                citations = c.get("citations", "")
                if not front or not back:
                    continue

                # Check if identical front exists for this subject
                cursor.execute(
                    "SELECT id FROM flashcards WHERE subject = ? AND front = ?",
                    (subject, front),
                )
                existing = cursor.fetchone()
                if existing:
                    continue

                cursor.execute(
                    """
                    INSERT INTO flashcards (
                        id, subject, chapter, category, front, back, citations,
                        mastery_level, interval_days, ease_factor, repetition_number,
                        last_reviewed, next_review_due, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, 'unseen', 1, 2.5, 0, NULL, ?, ?)
                    """,
                    (cid, subject, chapter, category, front, back, citations, now, now),
                )
                count += 1
            conn.commit()
        return count

    def update_flashcard_rating(self, card_id: str, rating: str) -> Dict[str, Any]:
        """Update flashcard mastery and schedule next review using SuperMemo SM-2.

        Ratings:
        - 'easy': User knows it well (quality=5). Extends interval significantly.
        - 'review': User knew partially / needs reinforcement (quality=3). Modest interval.
        - 'hard': User forgot / struggled (quality=1). Resets repetition to 0.
        """
        rating_clean = rating.lower().strip()
        quality_map = {"easy": 5, "review": 3, "hard": 1}
        q = quality_map.get(rating_clean, 3)

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM flashcards WHERE id = ?", (card_id,))
            row = cursor.fetchone()
            if not row:
                raise ValueError(f"Flashcard with id '{card_id}' not found.")

            card = dict(row)
            rep = card["repetition_number"]
            interval = card["interval_days"]
            ef = card["ease_factor"]

            # SM-2 calculation
            if q >= 3:
                if rep == 0:
                    interval = 1
                elif rep == 1:
                    interval = 6
                else:
                    interval = int(round(interval * ef))
                rep += 1
            else:
                rep = 0
                interval = 1

            # Adjust ease factor: EF' = EF + (0.1 - (5 - q) * (0.08 + (5 - q) * 0.02))
            ef = ef + (0.1 - (5 - q) * (0.08 + (5 - q) * 0.02))
            if ef < 1.3:
                ef = 1.3

            now = datetime.now()
            next_due = (now + timedelta(days=interval)).isoformat()
            last_rev = now.isoformat()

            # Mastery label
            mastery = "easy" if rating_clean == "easy" else ("review" if rating_clean == "review" else "hard")

            cursor.execute(
                """
                UPDATE flashcards SET
                    mastery_level = ?,
                    interval_days = ?,
                    ease_factor = ?,
                    repetition_number = ?,
                    last_reviewed = ?,
                    next_review_due = ?
                WHERE id = ?
                """,
                (mastery, interval, ef, rep, last_rev, next_due, card_id),
            )
            conn.commit()

            card["mastery_level"] = mastery
            card["interval_days"] = interval
            card["ease_factor"] = ef
            card["repetition_number"] = rep
            card["last_reviewed"] = last_rev
            card["next_review_due"] = next_due
            return card

    def get_flashcards(
        self,
        subject: Optional[str] = None,
        chapter: Optional[str] = None,
        category: Optional[str] = None,
        mastery: Optional[str] = None,
        due_only: bool = False,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Fetch flashcards with flexible filtering options."""
        query = "SELECT * FROM flashcards WHERE 1=1"
        params: List[Any] = []

        if subject and subject.lower() != "all":
            query += " AND subject = ?"
            params.append(subject)
        if chapter:
            query += " AND chapter = ?"
            params.append(chapter)
        if category:
            query += " AND category = ?"
            params.append(category)
        if mastery:
            query += " AND mastery_level = ?"
            params.append(mastery)
        if due_only:
            now_iso = datetime.now().isoformat()
            query += " AND (next_review_due IS NULL OR next_review_due <= ?)"
            params.append(now_iso)

        query += " ORDER BY CASE mastery_level WHEN 'hard' THEN 1 WHEN 'review' THEN 2 WHEN 'unseen' THEN 3 ELSE 4 END, id LIMIT ?"
        params.append(limit)

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, tuple(params))
            return [dict(r) for r in cursor.fetchall()]

    def get_flashcard_stats(self, subject: Optional[str] = None) -> Dict[str, int]:
        """Return counts of flashcards by mastery level."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if subject and subject.lower() != "all":
                cursor.execute(
                    """
                    SELECT mastery_level, COUNT(*) as cnt
                    FROM flashcards
                    WHERE subject = ?
                    GROUP BY mastery_level
                    """,
                    (subject,),
                )
            else:
                cursor.execute(
                    """
                    SELECT mastery_level, COUNT(*) as cnt
                    FROM flashcards
                    GROUP BY mastery_level
                    """
                )
            rows = cursor.fetchall()
            stats = {"total": 0, "unseen": 0, "easy": 0, "review": 0, "hard": 0}
            for r in rows:
                lvl = r["mastery_level"]
                cnt = r["cnt"]
                stats[lvl] = cnt
                stats["total"] += cnt
            return stats

    # -------------------------------------------------------------------------
    # Study Sessions Persistence
    # -------------------------------------------------------------------------
    def log_study_session(
        self,
        session_type: str,
        subject: str,
        duration_seconds: int,
        start_time: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Record a completed or in-progress study session."""
        st = start_time or datetime.now().isoformat()
        end_time = datetime.now().isoformat()
        meta_json = json.dumps(metadata or {})

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO study_sessions (
                    session_type, subject, start_time, end_time, duration_seconds, metadata
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (session_type, subject, st, end_time, int(duration_seconds), meta_json),
            )
            conn.commit()
            return cursor.lastrowid

    # -------------------------------------------------------------------------
    # Analytics & Aggregations
    # -------------------------------------------------------------------------
    def get_analytics_summary(self, subject: Optional[str] = None) -> Dict[str, Any]:
        """Aggregate comprehensive metrics across quizzes, viva, flashcards, and sessions."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            filter_clause = " WHERE subject = ?" if (subject and subject.lower() != "all") else ""
            params = (subject,) if (subject and subject.lower() != "all") else ()

            # 1. Quiz Metrics
            cursor.execute(
                f"""
                SELECT
                    COUNT(*) as total_attempts,
                    COALESCE(AVG(score), 0) as avg_score,
                    COALESCE(MAX(score), 0) as max_score,
                    COALESCE(SUM(total_questions), 0) as total_qs,
                    COALESCE(SUM(correct_answers), 0) as total_correct,
                    COALESCE(SUM(time_taken_seconds), 0) as total_quiz_time
                FROM quiz_attempts
                {filter_clause}
                """,
                params,
            )
            quiz_row = dict(cursor.fetchone() or {})

            # Quiz by subject breakdown
            cursor.execute(
                """
                SELECT
                    subject,
                    COUNT(*) as attempts,
                    ROUND(AVG(score), 1) as avg_score,
                    SUM(total_questions) as total_qs,
                    SUM(correct_answers) as total_correct
                FROM quiz_attempts
                GROUP BY subject
                ORDER BY avg_score DESC
                """
            )
            quiz_by_subject = [dict(r) for r in cursor.fetchall()]

            # 2. Oral Viva Metrics
            cursor.execute(
                f"""
                SELECT
                    COUNT(*) as total_vivas,
                    COALESCE(AVG(score), 0) as avg_viva_score,
                    COALESCE(AVG(accuracy_score), 0) as avg_accuracy,
                    COALESCE(AVG(completeness_score), 0) as avg_completeness,
                    COALESCE(AVG(key_terms_score), 0) as avg_key_terms
                FROM viva_evaluations
                {filter_clause}
                """,
                params,
            )
            viva_row = dict(cursor.fetchone() or {})

            # 3. Flashcards Breakdown
            fc_stats = self.get_flashcard_stats(subject=subject)

            # 4. Weak Topics Extraction
            cursor.execute(
                f"""
                SELECT weak_topics FROM quiz_attempts
                {filter_clause}
                ORDER BY id DESC LIMIT 20
                """,
                params,
            )
            weak_counter: Dict[str, int] = {}
            for r in cursor.fetchall():
                try:
                    topics = json.loads(r["weak_topics"] or "[]")
                    for t in topics:
                        t_clean = t.strip()
                        if t_clean:
                            weak_counter[t_clean] = weak_counter.get(t_clean, 0) + 1
                except Exception:
                    pass

            # Sort weak topics descending by frequency
            sorted_weak = sorted(weak_counter.items(), key=lambda x: x[1], reverse=True)

            # 5. Study Time
            cursor.execute(
                f"""
                SELECT COALESCE(SUM(duration_seconds), 0) as total_study_time
                FROM study_sessions
                {filter_clause}
                """,
                params,
            )
            sess_row = cursor.fetchone()
            total_time = (sess_row["total_study_time"] if sess_row else 0) + quiz_row.get("total_quiz_time", 0)

            # 6. Overall Performance Grade
            avg_score = quiz_row.get("avg_score", 0)
            if avg_score >= 80:
                overall_status = "Exam Ready (Distinction)"
            elif avg_score >= 60:
                overall_status = "Proficient (Passing Level)"
            elif avg_score > 0:
                overall_status = "Needs Focused Revision"
            else:
                overall_status = "Getting Started"

            return {
                "overall_status": overall_status,
                "total_study_seconds": total_time,
                "quiz_stats": {
                    "total_attempts": quiz_row.get("total_attempts", 0),
                    "avg_score": round(quiz_row.get("avg_score", 0), 1),
                    "max_score": round(quiz_row.get("max_score", 0), 1),
                    "total_questions_attempted": quiz_row.get("total_qs", 0),
                    "total_correct": quiz_row.get("total_correct", 0),
                    "accuracy_pct": (
                        round(100.0 * quiz_row.get("total_correct", 0) / max(quiz_row.get("total_qs", 1), 1), 1)
                        if quiz_row.get("total_qs", 0) > 0
                        else 0.0
                    ),
                    "by_subject": quiz_by_subject,
                },
                "viva_stats": {
                    "total_evaluations": viva_row.get("total_vivas", 0),
                    "avg_score": round(viva_row.get("avg_viva_score", 0), 1),
                    "avg_accuracy": round(viva_row.get("avg_accuracy", 0), 1),
                    "avg_completeness": round(viva_row.get("avg_completeness", 0), 1),
                    "avg_key_terms": round(viva_row.get("avg_key_terms", 0), 1),
                },
                "flashcard_stats": fc_stats,
                "weak_topics": [{"topic": k, "frequency": v} for k, v in sorted_weak[:10]],
            }

    # -------------------------------------------------------------------------
    # JSON Export & Import Engine
    # -------------------------------------------------------------------------
    def export_to_json(self, filepath: Optional[str] = None) -> str:
        """Export all database tables into a clean JSON structure."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("SELECT * FROM quiz_attempts ORDER BY id ASC")
            quiz_attempts = [dict(r) for r in cursor.fetchall()]
            for item in quiz_attempts:
                item["questions_data"] = json.loads(item.get("questions_data") or "[]")
                item["weak_topics"] = json.loads(item.get("weak_topics") or "[]")

            cursor.execute("SELECT * FROM viva_evaluations ORDER BY id ASC")
            viva_evaluations = [dict(r) for r in cursor.fetchall()]
            for item in viva_evaluations:
                item["missed_points"] = json.loads(item.get("missed_points") or "[]")
                item["correct_points"] = json.loads(item.get("correct_points") or "[]")
                item["citations"] = json.loads(item.get("citations") or "[]")

            cursor.execute("SELECT * FROM flashcards ORDER BY id ASC")
            flashcards = [dict(r) for r in cursor.fetchall()]

            cursor.execute("SELECT * FROM study_sessions ORDER BY id ASC")
            study_sessions = [dict(r) for r in cursor.fetchall()]
            for item in study_sessions:
                item["metadata"] = json.loads(item.get("metadata") or "{}")

        dump = {
            "version": "1.0",
            "exported_at": datetime.now().isoformat(),
            "quiz_attempts": quiz_attempts,
            "viva_evaluations": viva_evaluations,
            "flashcards": flashcards,
            "study_sessions": study_sessions,
        }

        json_str = json.dumps(dump, indent=2, ensure_ascii=False)
        if filepath:
            os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(json_str)
        return json_str

    def import_from_json(
        self, data_or_path: Union[str, Dict[str, Any]], merge: bool = True
    ) -> Dict[str, int]:
        """Import database records from a JSON string, filepath, or dictionary.

        If merge=False, clears existing tables before importing.
        """
        if isinstance(data_or_path, str):
            if os.path.exists(data_or_path):
                with open(data_or_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            else:
                data = json.loads(data_or_path)
        else:
            data = data_or_path

        counts = {"quizzes": 0, "vivas": 0, "flashcards": 0, "sessions": 0}

        with self._get_connection() as conn:
            cursor = conn.cursor()
            if not merge:
                cursor.execute("DELETE FROM quiz_attempts")
                cursor.execute("DELETE FROM viva_evaluations")
                cursor.execute("DELETE FROM flashcards")
                cursor.execute("DELETE FROM study_sessions")

            # 1. Quizzes
            for q in data.get("quiz_attempts", []):
                q_json = json.dumps(q.get("questions_data", []))
                w_json = json.dumps(q.get("weak_topics", []))
                cursor.execute(
                    """
                    INSERT INTO quiz_attempts (
                        timestamp, subject, chapter, score, total_questions,
                        correct_answers, time_taken_seconds, difficulty,
                        questions_data, weak_topics
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        q.get("timestamp", datetime.now().isoformat()),
                        q.get("subject", "General"),
                        q.get("chapter", ""),
                        float(q.get("score", 0.0)),
                        int(q.get("total_questions", 0)),
                        int(q.get("correct_answers", 0)),
                        int(q.get("time_taken_seconds", 0)),
                        q.get("difficulty", "medium"),
                        q_json,
                        w_json,
                    ),
                )
                counts["quizzes"] += 1

            # 2. Vivas
            for v in data.get("viva_evaluations", []):
                missed_json = json.dumps(v.get("missed_points", []))
                correct_json = json.dumps(v.get("correct_points", []))
                citations_json = json.dumps(v.get("citations", []))
                cursor.execute(
                    """
                    INSERT INTO viva_evaluations (
                        timestamp, subject, topic, question, student_response,
                        score, accuracy_score, completeness_score, key_terms_score,
                        feedback, spoken_feedback, missed_points, correct_points, citations
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        v.get("timestamp", datetime.now().isoformat()),
                        v.get("subject", "General"),
                        v.get("topic", ""),
                        v.get("question", ""),
                        v.get("student_response", ""),
                        float(v.get("score", 0.0)),
                        float(v.get("accuracy_score", 0.0)),
                        float(v.get("completeness_score", 0.0)),
                        float(v.get("key_terms_score", 0.0)),
                        v.get("feedback", ""),
                        v.get("spoken_feedback", ""),
                        missed_json,
                        correct_json,
                        citations_json,
                    ),
                )
                counts["vivas"] += 1

            # 3. Flashcards
            for fc in data.get("flashcards", []):
                cid = fc.get("id") or str(uuid.uuid4())[:8]
                cursor.execute(
                    """
                    INSERT INTO flashcards (
                        id, subject, chapter, category, front, back, citations,
                        mastery_level, interval_days, ease_factor, repetition_number,
                        last_reviewed, next_review_due, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        front = excluded.front,
                        back = excluded.back,
                        citations = excluded.citations,
                        mastery_level = excluded.mastery_level,
                        interval_days = excluded.interval_days,
                        ease_factor = excluded.ease_factor,
                        repetition_number = excluded.repetition_number,
                        next_review_due = excluded.next_review_due
                    """,
                    (
                        cid,
                        fc.get("subject", "General"),
                        fc.get("chapter", ""),
                        fc.get("category", "definition"),
                        fc.get("front", ""),
                        fc.get("back", ""),
                        fc.get("citations", ""),
                        fc.get("mastery_level", "unseen"),
                        int(fc.get("interval_days", 1)),
                        float(fc.get("ease_factor", 2.5)),
                        int(fc.get("repetition_number", 0)),
                        fc.get("last_reviewed"),
                        fc.get("next_review_due"),
                        fc.get("created_at", datetime.now().isoformat()),
                    ),
                )
                counts["flashcards"] += 1

            # 4. Sessions
            for s in data.get("study_sessions", []):
                meta_json = json.dumps(s.get("metadata", {}))
                cursor.execute(
                    """
                    INSERT INTO study_sessions (
                        session_type, subject, start_time, end_time, duration_seconds, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        s.get("session_type", "study"),
                        s.get("subject", "General"),
                        s.get("start_time", datetime.now().isoformat()),
                        s.get("end_time"),
                        int(s.get("duration_seconds", 0)),
                        meta_json,
                    ),
                )
                counts["sessions"] += 1

            conn.commit()
        return counts

    # -------------------------------------------------------------------------
    # Comprehensive Summary Report Generation
    # -------------------------------------------------------------------------
    def generate_summary_report(self, subject: Optional[str] = None) -> str:
        """Generate a formatted Markdown revision report suitable for download."""
        summary = self.get_analytics_summary(subject=subject)
        q = summary["quiz_stats"]
        v = summary["viva_stats"]
        fc = summary["flashcard_stats"]
        weaks = summary["weak_topics"]

        mins = summary["total_study_seconds"] // 60
        secs = summary["total_study_seconds"] % 60
        study_time_str = f"{mins} min {secs} sec" if mins > 0 else f"{secs} sec"

        subject_hdr = subject if (subject and subject.lower() != "all") else "Comprehensive (All Subjects)"

        md = []
        md.append(f"# 📊 StudyAssistant Performance & Revision Report")
        md.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ")
        md.append(f"**Scope:** {subject_hdr}  ")
        md.append(f"**Assessment Status:** `{summary['overall_status']}`  ")
        md.append(f"**Total Tracked Study Time:** `{study_time_str}`  \n")
        md.append("---")

        # 1. Objective Mock Tests Section
        md.append("## 📝 Objective Mock Tests (MCQ)")
        md.append(f"- **Total Tests Taken:** {q['total_attempts']}")
        md.append(f"- **Overall Average Score:** {q['avg_score']}%")
        md.append(f"- **Best Score:** {q['max_score']}%")
        md.append(f"- **Questions Solved:** {q['total_questions_attempted']} (Correct: {q['total_correct']} | Accuracy: {q['accuracy_pct']}%)")

        if q["by_subject"]:
            md.append("\n### Subject-Wise Mock Test Breakdown")
            md.append("| Subject | Tests Taken | Avg Score | Solved | Correct |")
            md.append("| :--- | :---: | :---: | :---: | :---: |")
            for sub in q["by_subject"]:
                md.append(f"| {sub['subject']} | {sub['attempts']} | {sub['avg_score']}% | {sub['total_qs']} | {sub['total_correct']} |")

        # 2. Oral Viva Section
        md.append("\n---\n## 🎙️ Oral Viva Voce (Examiner Mode)")
        md.append(f"- **Total Viva Examinations:** {v['total_evaluations']}")
        md.append(f"- **Average Overall Viva Score:** {v['avg_score']}/100")
        md.append(f"- **Accuracy Rating:** {v['avg_accuracy']}%")
        md.append(f"- **Conceptual Completeness:** {v['avg_completeness']}%")
        md.append(f"- **Key Terminology Mastery:** {v['avg_key_terms']}%")

        # 3. Flashcards Section
        md.append("\n---\n## 🗂️ Flashcard Mastery & Spaced Repetition")
        md.append(f"- **Total Flashcards in Deck:** {fc['total']}")
        md.append(f"- 🟢 **Mastered (Easy):** {fc['easy']}")
        md.append(f"- 🟡 **Needs Review:** {fc['review']}")
        md.append(f"- 🔴 **Challenging (Hard):** {fc['hard']}")
        md.append(f"- ⚪ **Unseen:** {fc['unseen']}")

        # 4. Weak Topics & Action Items
        md.append("\n---\n## ⚠️ High-Priority Weak Topics for Revision")
        if weaks:
            md.append("The following concepts were repeatedly missed during mock tests and oral examinations:")
            for item in weaks:
                md.append(f"- **{item['topic']}** (Flagged {item['frequency']} times)")
        else:
            md.append("🎉 No persistent weak topics flagged yet! Keep practicing to maintain readiness.")

        # 5. Recommended Next Steps
        md.append("\n---\n## 🎯 Recommended Next Revision Actions")
        if fc["hard"] > 0 or fc["review"] > 0:
            md.append(f"1. Drill the **{fc['hard'] + fc['review']} flashcards** in 'Hard' and 'Needs Review' status.")
        if weaks:
            top_weak = weaks[0]["topic"]
            md.append(f"2. Launch a targeted 5-question mock test specifically on **{top_weak}**.")
        md.append("3. Conduct an Oral Viva session on recent tariff updates to test spoken articulation under time pressure.")

        return "\n".join(md)


# Singleton Database Instance
_default_db_instance: Optional[Database] = None


def get_db(db_path: Optional[str] = None) -> Database:
    """Get or create singleton Database instance."""
    global _default_db_instance
    if db_path or _default_db_instance is None:
        inst = Database(db_path=db_path)
        if db_path is None:
            _default_db_instance = inst
        return inst
    return _default_db_instance
