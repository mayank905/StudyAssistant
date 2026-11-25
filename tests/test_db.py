"""Unit and Integration Tests for core.db (SQLite Persistence & Analytics Engine)
"""

import gc
import json
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, ".")

from core.db import Database


class TestDatabase(unittest.TestCase):
    def setUp(self):
        # Create a temporary database file for test isolation
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_study.db")
        self.db = Database(db_path=self.db_path)

    def tearDown(self):
        del self.db
        gc.collect()
        try:
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        except Exception:
            pass

    def test_database_initialization(self):
        """Verify that tables and indexes are created properly upon initialization."""
        self.assertTrue(os.path.exists(self.db_path))
        with self.db._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;"
            )
            tables = [row["name"] for row in cursor.fetchall()]
            self.assertIn("quiz_attempts", tables)
            self.assertIn("viva_evaluations", tables)
            self.assertIn("flashcards", tables)
            self.assertIn("study_sessions", tables)

    def test_log_and_get_quiz_attempt(self):
        """Verify logging quiz attempts and retrieving quiz history with filters."""
        attempt_id = self.db.log_quiz_attempt(
            subject="Fire Insurance",
            score=85.0,
            total_questions=10,
            correct_answers=8,
            chapter="Section 4 - Industrial Tariffs",
            time_taken_seconds=180,
            difficulty="medium",
            questions_data=[
                {"q": "What is the deductible?", "user_choice": "A", "correct": "A"}
            ],
            weak_topics=["Storage Risk Tariff", "Deductibles"],
        )
        self.assertIsInstance(attempt_id, int)
        self.assertGreater(attempt_id, 0)

        # Retrieve history (All)
        history = self.db.get_quiz_history(limit=10)
        self.assertEqual(len(history), 1)
        item = history[0]
        self.assertEqual(item["subject"], "Fire Insurance")
        self.assertEqual(item["score"], 85.0)
        self.assertEqual(item["correct_answers"], 8)
        self.assertEqual(item["weak_topics"], ["Storage Risk Tariff", "Deductibles"])
        self.assertEqual(len(item["questions_data"]), 1)

        # Subject-specific filter
        fire_history = self.db.get_quiz_history(subject="Fire Insurance")
        self.assertEqual(len(fire_history), 1)
        motor_history = self.db.get_quiz_history(subject="Motor Insurance")
        self.assertEqual(len(motor_history), 0)

    def test_log_and_get_viva_evaluation(self):
        """Verify logging viva evaluations and retrieving history."""
        viva_id = self.db.log_viva_evaluation(
            subject="Motor Insurance",
            topic="Third Party Liability",
            question="Explain compulsory third party liability insurance under Motor Vehicles Act.",
            student_response="It covers death, bodily injury and property damage to third parties.",
            score=78.5,
            accuracy_score=80.0,
            completeness_score=75.0,
            key_terms_score=80.0,
            feedback="Good summary, missed specifying Section 146 requirement.",
            spoken_feedback="You covered the basics well, but remember Section 146.",
            missed_points=["Section 146 mandate", "Property damage limit of Rs 7.5 lakhs"],
            correct_points=["Third party death", "Bodily injury coverage"],
            citations=["Motor Write-Up PE 2026, Page 4"],
        )
        self.assertIsInstance(viva_id, int)
        self.assertGreater(viva_id, 0)

        viva_history = self.db.get_viva_history(limit=10)
        self.assertEqual(len(viva_history), 1)
        v = viva_history[0]
        self.assertEqual(v["subject"], "Motor Insurance")
        self.assertEqual(v["score"], 78.5)
        self.assertIn("Section 146 mandate", v["missed_points"])
        self.assertIn("Motor Write-Up PE 2026, Page 4", v["citations"])

    def test_flashcard_upsert_and_sm2_spaced_repetition(self):
        """Verify flashcard upsert, SM-2 updates, and mastery counts."""
        card_id = self.db.upsert_flashcard(
            front="What is Average Clause in Fire Insurance?",
            back="Penalizes under-insurance when Sum Insured is less than actual value.",
            subject="Fire Insurance",
            chapter="General Rules",
            category="rule",
            citations="Fire Tariff Page 8",
        )
        self.assertIsNotNone(card_id)

        # Check initial stats
        stats = self.db.get_flashcard_stats(subject="Fire Insurance")
        self.assertEqual(stats["total"], 1)
        self.assertEqual(stats["unseen"], 1)

        # Update rating to 'easy' (quality=5)
        updated = self.db.update_flashcard_rating(card_id, "easy")
        self.assertEqual(updated["mastery_level"], "easy")
        self.assertEqual(updated["repetition_number"], 1)
        self.assertEqual(updated["interval_days"], 1)

        # Update rating again to 'easy'
        updated2 = self.db.update_flashcard_rating(card_id, "easy")
        self.assertEqual(updated2["repetition_number"], 2)
        self.assertEqual(updated2["interval_days"], 6)

        # Update rating to 'hard' (resets repetition)
        updated3 = self.db.update_flashcard_rating(card_id, "hard")
        self.assertEqual(updated3["mastery_level"], "hard")
        self.assertEqual(updated3["repetition_number"], 0)
        self.assertEqual(updated3["interval_days"], 1)

        # Check updated stats
        stats2 = self.db.get_flashcard_stats(subject="Fire Insurance")
        self.assertEqual(stats2["hard"], 1)

    def test_save_flashcards_batch(self):
        """Verify batch insertion of flashcards and deduplication."""
        cards = [
            {
                "front": "Term 1",
                "back": "Def 1",
                "subject": "Engineering",
                "category": "definition",
            },
            {
                "front": "Term 2",
                "back": "Def 2",
                "subject": "Engineering",
                "category": "formula",
            },
            {
                "front": "Term 1",  # duplicate front for Engineering
                "back": "Def 1 duplicate",
                "subject": "Engineering",
            },
        ]
        inserted = self.db.save_flashcards_batch(cards)
        self.assertEqual(inserted, 2)

        fc_list = self.db.get_flashcards(subject="Engineering")
        self.assertEqual(len(fc_list), 2)

    def test_log_study_session_and_analytics_summary(self):
        """Verify study session logging and aggregated analytics summary."""
        self.db.log_study_session(
            session_type="quiz",
            subject="Fire Insurance",
            duration_seconds=300,
        )
        self.db.log_quiz_attempt(
            subject="Fire Insurance",
            score=90.0,
            total_questions=10,
            correct_answers=9,
            weak_topics=["Cold Storage Clause"],
        )

        summary = self.db.get_analytics_summary()
        self.assertEqual(summary["quiz_stats"]["total_attempts"], 1)
        self.assertEqual(summary["quiz_stats"]["avg_score"], 90.0)
        self.assertEqual(summary["quiz_stats"]["accuracy_pct"], 90.0)
        self.assertGreaterEqual(summary["total_study_seconds"], 300)
        self.assertEqual(summary["overall_status"], "Exam Ready (Distinction)")
        self.assertEqual(len(summary["weak_topics"]), 1)
        self.assertEqual(summary["weak_topics"][0]["topic"], "Cold Storage Clause")

    def test_json_export_and_import(self):
        """Verify full database JSON export, reset, and JSON import."""
        self.db.log_quiz_attempt(
            subject="Engineering",
            score=70.0,
            total_questions=5,
            correct_answers=3,
        )
        self.db.upsert_flashcard(
            front="Test Front",
            back="Test Back",
            subject="Engineering",
        )

        export_json = self.db.export_to_json()
        data = json.loads(export_json)
        self.assertIn("quiz_attempts", data)
        self.assertEqual(len(data["quiz_attempts"]), 1)
        self.assertEqual(len(data["flashcards"]), 1)

        # Clear and restore via import_from_json
        import_stats = self.db.import_from_json(data, merge=False)
        self.assertEqual(import_stats["quizzes"], 1)
        self.assertEqual(import_stats["flashcards"], 1)

        restored_history = self.db.get_quiz_history()
        self.assertEqual(len(restored_history), 1)

    def test_generate_summary_report(self):
        """Verify summary report generation produces formatted text with sections."""
        self.db.log_quiz_attempt(
            subject="Fire Insurance",
            score=80.0,
            total_questions=10,
            correct_answers=8,
            weak_topics=["Tariff Schedule B"],
        )
        report = self.db.generate_summary_report()
        self.assertIn("# 📊 StudyAssistant Performance & Revision Report", report)
        self.assertIn("Fire Insurance", report)
        self.assertIn("Tariff Schedule B", report)


if __name__ == "__main__":
    unittest.main()
