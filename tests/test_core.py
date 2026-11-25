"""Unit and Integration Tests for core.rag_engine and core.providers
"""

import os
import sys
import unittest

sys.path.insert(0, ".")

from core.providers import (
    ProviderManager,
    LocalHashEmbeddingProvider,
    OfflineExtractiveLLMProvider,
)
from core.rag_engine import (
    Citation,
    DocumentChunk,
    GroundingGuard,
    PDFIngestor,
    RAGEngine,
    RAGQueryResult,
    VectorStore,
)


class TestProviders(unittest.TestCase):
    def setUp(self):
        self.pm = ProviderManager(engine_type="offline")

    def test_provider_manager_status(self):
        status = self.pm.get_status()
        self.assertIn("mode", status)
        self.assertIn("active_llm_provider", status)
        self.assertIn("active_embedding_dimension", status)
        self.assertGreater(status["active_embedding_dimension"], 0)

    def test_offline_embeddings(self):
        embedder = LocalHashEmbeddingProvider(dimension=128)
        vecs = embedder.embed_documents(["Fire insurance policy", "Motor liability coverage"])
        self.assertEqual(len(vecs), 2)
        self.assertEqual(len(vecs[0]), 128)
        self.assertEqual(len(vecs[1]), 128)

        # Single query embed
        q_vec = embedder.embed_text("Fire insurance")
        self.assertEqual(len(q_vec), 128)

    def test_offline_llm_generation(self):
        llm = OfflineExtractiveLLMProvider()
        self.assertTrue(llm.is_available())
        prompt = (
            "Context: Under the All India Fire Tariff, the standard deductible is 5%.\n"
            "Question: What is the standard deductible?"
        )
        response = llm.generate(prompt=prompt)
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)


class TestPDFIngestor(unittest.TestCase):
    def test_detect_subject(self):
        self.assertEqual(
            PDFIngestor.detect_subject("ENGG_Study_Material_Fast_Revision.pdf"),
            "Engineering",
        )
        self.assertEqual(
            PDFIngestor.detect_subject("Fire tariff.pdf"),
            "Fire",
        )
        self.assertEqual(
            PDFIngestor.detect_subject("UPDATED_MOTOR_WRITE_UP_FOR_PE.pdf"),
            "Motor",
        )
        self.assertEqual(
            PDFIngestor.detect_subject("custom_document.pdf", sample_text="Discussion on general topics"),
            "General",
        )

    def test_detect_outline_and_chunk(self):
        pages = [
            (
                1,
                """SECTION 1: GENERAL FIRE RULES
The All India Fire Tariff applies to all industrial and storage risks located within India.
The sum insured must represent the actual market value or reinstatement value.
SECTION 2: DEDUCTIBLE FRANCHISE
A standard excess of 5% of the claim amount applies to all storage risks.""",
            )
        ]
        chunks = PDFIngestor.detect_outline_and_chunk(
            pages=pages,
            file_name="fire_test.pdf",
            subject="Fire",
            chunk_size=150,
            chunk_overlap=30,
        )
        self.assertGreaterEqual(len(chunks), 1)
        for c in chunks:
            self.assertIsInstance(c, DocumentChunk)
            self.assertEqual(c.subject, "Fire")
            self.assertEqual(c.file_name, "fire_test.pdf")
            self.assertEqual(c.page_number, 1)
            self.assertTrue(len(c.content) > 0)


class TestGroundingGuard(unittest.TestCase):
    def test_construct_prompt_and_guard(self):
        chunk = DocumentChunk(
            chunk_id="chk1",
            content="Standard excess for Fire tariff is 5% for storage risks.",
            file_name="Fire_Tariff.pdf",
            subject="Fire",
            chapter="Section 2",
            page_number=4,
        )
        prompt = GroundingGuard.construct_prompt(
            question="What is the standard excess?",
            retrieved_chunks=[(chunk, 0.85)],
            allow_external_knowledge=False,
        )
        self.assertIn("Standard excess for Fire tariff is 5%", prompt)
        self.assertIn("--- CONTEXT EXCERPTS ---", prompt)
        self.assertIn("ZERO HALLUCINATION POLICY", GroundingGuard.SYSTEM_PROMPT)

    def test_extract_citations(self):
        chunk = DocumentChunk(
            chunk_id="chk1",
            content="Standard excess for Fire tariff is 5% for storage risks.",
            file_name="Fire_Tariff.pdf",
            subject="Fire",
            chapter="Section 2",
            page_number=4,
        )
        answer = "Under the rules, the standard excess is 5% [Source 1: Fire_Tariff.pdf | Page 4]."
        citations, score = GroundingGuard.extract_citations(
            answer_text=answer,
            retrieved_chunks=[(chunk, 0.9)],
        )
        self.assertGreaterEqual(len(citations), 1)
        self.assertEqual(citations[0].file_name, "Fire_Tariff.pdf")
        self.assertEqual(citations[0].page_number, 4)
        self.assertGreater(score, 0.0)


class TestVectorStoreAndRAGEngine(unittest.TestCase):
    def setUp(self):
        pm = ProviderManager(engine_type="offline")
        self.rag = RAGEngine(provider_manager=pm)

    def test_dynamic_document_ingestion_and_query(self):
        doc_text = """
INDUSTRIAL ALL RISKS (IAR) POLICY
CHAPTER 1: SCOPE OF COVER
Industrial All Risks (IAR) policy covers all industrial risks with overall sum insured of Rs. 100 Crores and above.
It combines Fire and Special Perils, Machinery Breakdown, and Business Interruption covers.
Deductible for Material Damage is 5% of the claim amount subject to a minimum of Rs. 5 Lakhs.
"""
        chunks_added = self.rag.ingest_document(
            file_source=doc_text,
            file_name="IAR_Guidelines_2026.txt",
            subject="Engineering",
            chapter="Chapter 1: Scope of Cover",
        )
        self.assertGreaterEqual(chunks_added, 1)

        # Check subjects
        subjects = self.rag.get_available_subjects()
        self.assertIn("Engineering", subjects)

        # Query the ingested text
        res = self.rag.query(
            question="What is the minimum sum insured for Industrial All Risks (IAR)?",
            subject="Engineering",
            top_k=2,
        )
        self.assertIsInstance(res, dict)
        self.assertTrue(res["grounded"])
        self.assertIn("Industrial All Risks", res["answer"])
        self.assertGreaterEqual(len(res["citations"]), 1)
        self.assertEqual(res["citations"][0]["file_name"], "IAR_Guidelines_2026.txt")

    def test_rag_stats(self):
        stats = self.rag.get_stats()
        self.assertIn("total_documents", stats)
        self.assertIn("total_chunks", stats)
        self.assertIn("subject_breakdown", stats)


if __name__ == "__main__":
    unittest.main()
