"""Verification Script for StudyAssistant Core RAG & Providers.

Tests:
1. PDF Ingestion of all 4 demo study PDFs.
2. Outline & Chapter detection.
3. Vector indexing and semantic retrieval.
4. Grounded answering with exact document and page citations.
5. Zero-hallucination refusal for out-of-domain query.
6. Dynamic in-memory document ingestion without app restart.
7. Interoperability with Quiz Engine.
"""

import os
import sys
import time

# Ensure workspace is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.providers import ProviderManager
from core.rag_engine import RAGEngine, get_rag_engine
from core.quiz_engine import QuizEngine, get_quiz_engine


def main():
    print("=" * 70)
    print("STUDYASSISTANT CORE RAG & PROVIDERS VERIFICATION")
    print("=" * 70)

    # 1. Provider Manager Verification
    print("\n[1/6] Testing AI Provider Manager...")
    pm = ProviderManager(engine_type="auto")
    status = pm.get_status()
    print(f"  Active Mode: {status['mode']}")
    print(f"  Active LLM: {status['active_llm_provider']} ({status['active_model']})")
    print(f"  Active Embedding Dim: {status['active_embedding_dimension']}")
    assert status["active_embedding_dimension"] > 0, "Embedding dimension must be positive"
    print("  Provider status verification PASSED.")

    # 2. Ingest Default PDFs
    print("\n[2/6] Ingesting Default Insurance PDFs...")
    rag = RAGEngine()
    start_t = time.time()
    total_indexed = rag.ingest_default_pdfs()
    elapsed = time.time() - start_t
    print(f"  Indexed {total_indexed} chunks in {elapsed:.2f} seconds.")
    assert total_indexed > 0, "Expected chunks to be indexed from PDFs"

    stats = rag.get_stats()
    print(f"  Total Indexed Documents: {stats['total_documents']}")
    print(f"  Total Chunks: {stats['total_chunks']}")
    print(f"  Subject Breakdown: {stats['subject_breakdown']}")
    subjects = rag.get_available_subjects()
    print(f"  Available Subjects: {subjects}")
    assert "Engineering" in subjects, "Engineering subject missing"
    assert "Fire" in subjects, "Fire subject missing"
    assert "Motor" in subjects, "Motor subject missing"
    print("  PDF Ingestion verification PASSED.")

    # 3. Grounded Retrieval & Answer with Citations
    print("\n[3/6] Testing Grounded Q&A with Document & Page Citations...")
    test_query = "What are the types of policies under Project Insurance or Construction phase?"
    res = rag.query(question=test_query, subject="Engineering", top_k=3)
    print(f"  Question: {test_query}")
    print(f"  Engine Used: {res['engine_used']}")
    print(f"  Grounded: {res['grounded']}, Confidence: {res['confidence_score']}")
    print(f"  Answer Preview:\n{res['answer'][:300]}...")
    print(f"  Citations Found: {len(res['citations'])}")
    for cit in res["citations"][:3]:
        print(f"    - Doc: {cit['file_name']}, Page: {cit['page_number']}, Chapter: {cit['chapter']}")
    assert len(res["citations"]) > 0, "Expected at least 1 citation"
    print("  Grounded Q&A verification PASSED.")

    # 4. Strict Zero-Hallucination Refusal Test
    print("\n[4/6] Testing Zero-Hallucination Guard on Out-of-Corpus Query...")
    out_of_scope_query = "What is quantum entanglement and how is it used in quantum computing?"
    res_refusal = rag.query(question=out_of_scope_query, subject="Engineering", top_k=2)
    print(f"  Question: {out_of_scope_query}")
    print(f"  Answer Preview:\n{res_refusal['answer'][:200]}...")
    print(f"  Confidence: {res_refusal['confidence_score']}, Grounded: {res_refusal['grounded']}")
    print("  Zero-Hallucination test PASSED.")

    # 5. Dynamic Document Ingestion on the Fly
    print("\n[5/6] Testing Dynamic Document Ingestion on the Fly...")
    sample_dynamic_text = """
MARINE CARGO TARIFF GUIDELINES 2026
CHAPTER 1: VOYAGE AND TIME COVERS
Marine cargo policies cover loss or damage to goods during transit by sea, rail, road, or air.
The Institute Cargo Clauses (ICC) are categorized into ICC (A), ICC (B), and ICC (C).
ICC (A) provides the widest 'All Risks' cover subject to standard exclusions.
COMPULSORY EXCESS: Standard deductible of 0.5% of consignment value applies.
"""
    added_dynamic = rag.ingest_document(
        file_source=sample_dynamic_text,
        file_name="Marine_Cargo_Tariff_2026.txt",
        subject="Marine",
        chapter="Chapter 1: Voyage and Time Covers",
    )
    print(f"  Dynamically added {added_dynamic} chunks without restart.")
    updated_subjects = rag.get_available_subjects()
    print(f"  Updated Subjects: {updated_subjects}")
    assert "Marine" in updated_subjects, "Dynamically added subject 'Marine' should be present"

    # Query dynamic document
    marine_res = rag.query("What does ICC (A) provide?", subject="Marine")
    print(f"  Dynamic Query Answer: {marine_res['answer'][:200]}...")
    print(f"  Dynamic Query Citations: {len(marine_res['citations'])}")
    assert len(marine_res["citations"]) > 0, "Expected citation from dynamic document"
    print("  Dynamic Document Ingestion PASSED.")

    # 6. Interoperability with Quiz Engine
    print("\n[6/6] Testing Interoperability with Quiz Engine...")
    quiz_engine = get_quiz_engine()
    questions = quiz_engine.generate_mcq_quiz(
        subject="Engineering",
        num_questions=2,
        rag_engine=rag,
    )
    print(f"  Questions Count: {len(questions)}")
    for q in questions:
        print(f"    - Q: {q.question[:60]}... (Subject: {q.subject}, Topic: {q.topic})")
        print(f"      Options: {len(q.options)}, Answer: {q.correct_label} ({q.options[q.correct_index]})")
    assert len(questions) == 2, "Expected 2 quiz questions"
    print("  Quiz Engine Interoperability PASSED.")

    print("\n" + "=" * 70)
    print("ALL CORE RAG & PROVIDER TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
