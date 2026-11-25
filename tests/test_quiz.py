"""Unit tests for core.quiz_engine MCQ generation, oral viva evaluation rubric, and flashcard spaced repetition.
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, ".")
from core.db import Database
from core.quiz_engine import (
    QuizEngine,
    MCQQuestion,
    VivaQuestion,
    FlashcardItem,
    get_quiz_engine,
)


class TestQuizEngine(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_quiz_db.db")
        self.db = Database(db_path=self.db_path)
        self.engine = QuizEngine(db=self.db)

    def tearDown(self):
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except Exception:
                pass

    # -------------------------------------------------------------------------
    # MCQ Tests
    # -------------------------------------------------------------------------
    def test_mcq_generation_structure(self):
        questions = self.engine.generate_mcq_quiz(
            subject="Fire Insurance",
            num_questions=5,
            difficulty="medium",
        )
        self.assertEqual(len(questions), 5)
        for q in questions:
            self.assertIsInstance(q, MCQQuestion)
            # Enforce exactly 4 options
            self.assertEqual(len(q.options), 4, f"Question {q.id} must have exactly 4 options")
            # Enforce single correct answer in range 0..3
            self.assertIn(q.correct_index, [0, 1, 2, 3])
            self.assertIn(q.correct_label, ["A", "B", "C", "D"])
            # Explanation and citation must not be empty
            self.assertTrue(len(q.explanation) > 10)
            self.assertTrue(len(q.citation) > 3)

    def test_mcq_subject_filtering(self):
        motor_qs = self.engine.generate_mcq_quiz(subject="Motor Insurance", num_questions=4)
        for q in motor_qs:
            self.assertEqual(q.subject, "Motor Insurance")

        engg_qs = self.engine.generate_mcq_quiz(subject="Engineering Insurance", num_questions=4)
        for q in engg_qs:
            self.assertEqual(q.subject, "Engineering Insurance")

    def test_mcq_evaluation_perfect_score(self):
        questions = self.engine.generate_mcq_quiz(subject="Motor Insurance", num_questions=3)
        # Answer correctly
        user_answers = {i: q.correct_index for i, q in enumerate(questions)}
        result = self.engine.evaluate_quiz_submission(
            questions=questions,
            user_answers=user_answers,
            time_taken_seconds=45,
            persist=True,
        )
        self.assertEqual(result.correct_count, 3)
        self.assertEqual(result.score_percentage, 100.0)
        self.assertIn("Distinction", result.grade)
        self.assertEqual(len(result.weak_topics), 0)

        # Verify DB logged
        history = self.db.get_quiz_history(limit=5, subject="Motor Insurance")
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["score"], 100.0)

    def test_mcq_evaluation_with_mistakes_and_weak_topics(self):
        questions = self.engine.generate_mcq_quiz(subject="Fire Insurance", num_questions=3)
        # Select wrong option for question 0, right for question 1 & 2
        user_answers = {
            0: (questions[0].correct_index + 1) % 4,
            1: questions[1].correct_index,
            2: questions[2].correct_index,
        }
        result = self.engine.evaluate_quiz_submission(
            questions=questions,
            user_answers=user_answers,
            time_taken_seconds=60,
            persist=True,
        )
        self.assertEqual(result.correct_count, 2)
        self.assertEqual(result.score_percentage, 66.7)
        self.assertGreater(len(result.weak_topics), 0)

    # -------------------------------------------------------------------------
    # Oral Viva Voce Tests
    # -------------------------------------------------------------------------
    def test_viva_question_generation(self):
        viva_q = self.engine.generate_viva_question(subject="Motor Insurance")
        self.assertIsInstance(viva_q, VivaQuestion)
        self.assertTrue(len(viva_q.question) > 10)
        self.assertGreaterEqual(len(viva_q.expected_points), 3)
        self.assertGreaterEqual(len(viva_q.key_terms), 2)
        self.assertTrue(len(viva_q.citation) > 0)

    def test_viva_evaluation_rubric_good_response(self):
        viva_q = self.engine.generate_viva_question(subject="Fire Insurance", topic="Condition of Average")
        # Simulate knowledgeable student spoken response
        spoken_response = (
            "Under the Condition of Average, when the sum insured is less than the actual market value of property "
            "at the time of loss, there is under-insurance. The insured is considered their own insurer for the difference, "
            "and the claim is settled on a pro-rata basis using the formula loss times sum insured divided by value."
        )
        evaluation = self.engine.evaluate_viva_response(
            question=viva_q,
            student_response=spoken_response,
            persist=True,
        )
        self.assertGreater(evaluation.score, 70.0)
        self.assertGreater(evaluation.accuracy_score, 60.0)
        self.assertGreater(evaluation.completeness_score, 50.0)
        self.assertGreater(len(evaluation.correct_points), 0)
        self.assertTrue(len(evaluation.spoken_feedback) > 0)
        self.assertNotIn("*", evaluation.spoken_feedback)  # Clean for TTS

        # Verify DB logged
        viva_hist = self.db.get_viva_history(limit=5, subject=viva_q.subject)
        self.assertEqual(len(viva_hist), 1)

    def test_viva_evaluation_empty_response(self):
        viva_q = self.engine.generate_viva_question(subject="Motor Insurance")
        evaluation = self.engine.evaluate_viva_response(
            question=viva_q,
            student_response="",
            persist=False,
        )
        self.assertEqual(evaluation.score, 0.0)
        self.assertEqual(len(evaluation.correct_points), 0)
        self.assertEqual(len(evaluation.missed_points), len(viva_q.expected_points))

    # -------------------------------------------------------------------------
    # Flashcard & Spaced Repetition Tests
    # -------------------------------------------------------------------------
    def test_flashcard_generation(self):
        cards = self.engine.generate_flashcards(subject="Fire Insurance", num_cards=5)
        self.assertGreaterEqual(len(cards), 3)
        for c in cards:
            self.assertIsInstance(c, FlashcardItem)
            self.assertTrue(len(c.front) > 0)
            self.assertTrue(len(c.back) > 0)
            self.assertTrue(len(c.citation) > 0)

        # Check DB auto-populated
        db_cards = self.db.get_flashcards(subject="Fire Insurance")
        self.assertGreaterEqual(len(db_cards), 3)

    def test_flashcard_rating_spaced_repetition(self):
        cards = self.engine.generate_flashcards(subject="Motor Insurance", num_cards=1)
        card_id = cards[0].id

        # Rate easy
        res = self.engine.rate_flashcard(card_id, "easy")
        self.assertEqual(res["mastery_level"], "easy")
        self.assertGreaterEqual(res["interval_days"], 1)

        # Rate hard -> resets
        res2 = self.engine.rate_flashcard(card_id, "hard")
        self.assertEqual(res2["mastery_level"], "hard")
        self.assertEqual(res2["interval_days"], 1)


if __name__ == "__main__":
    unittest.main()
