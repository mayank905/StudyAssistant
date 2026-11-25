"""StudyAssistant - AI-Powered Voice-Enabled Insurance Exam Study Application.

Full-stack Streamlit app integrating:
- Dynamic RAG with ChromaDB & zero-hallucination grounding across Insurance PDFs
- Two-way voice communication (Speech-to-Text input & Text-to-Speech output)
- Objective Mock Tests with timed MCQs, instant grading, and citations
- Spoken Oral Viva examination mode with rubric scoring
- Interactive Flashcards with SuperMemo SM-2 spaced repetition
- Persistent SQLite Analytics dashboard & JSON/Markdown report export
- Mobile QR Code & HTTPS tunneling launcher
"""

import os
import sys
import time
import json
from datetime import datetime
import streamlit as st

# Add workspace directory to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from core.providers import ProviderManager
from core.rag_engine import RAGEngine, get_rag_engine
from core.db import Database, get_db
from core.quiz_engine import QuizEngine, get_quiz_engine, MCQQuestion, QuestionEvaluation
from core.voice import (
    render_speech_input_component,
    render_speech_synthesis,
    render_voice_controls,
    render_audio_recorder,
    clean_text_for_speech,
)
from core.tunnel import (
    get_local_ip,
    get_local_url,
    get_all_local_ips,
    render_mobile_access_card,
    generate_qr_code_data_uri,
)

# -----------------------------------------------------------------------------
# Streamlit Page Configuration & Custom CSS
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Insurance StudyAssistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
/* Modern styling for mobile and desktop */
.main-title {
    font-size: 2.1rem;
    font-weight: 700;
    color: #1E3A8A;
    margin-bottom: 0.2rem;
}
.sub-title {
    font-size: 1rem;
    color: #4B5563;
    margin-bottom: 1.2rem;
}
.stat-card {
    background-color: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 14px 18px;
    margin-bottom: 12px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
.stat-val {
    font-size: 1.8rem;
    font-weight: 700;
    color: #1E40AF;
}
.stat-label {
    font-size: 0.85rem;
    color: #64748B;
    text-transform: uppercase;
    font-weight: 600;
}
.citation-box {
    background-color: #EFF6FF;
    border-left: 4px solid #3B82F6;
    padding: 10px 14px;
    border-radius: 4px;
    font-size: 0.88rem;
    margin-top: 8px;
    color: #1E293B;
}
.quiz-card {
    background-color: #FFFFFF;
    border: 1px solid #CBD5E1;
    border-radius: 10px;
    padding: 16px;
    margin-bottom: 14px;
}
.badge-pass {
    background-color: #DCFCE7;
    color: #166534;
    padding: 4px 10px;
    border-radius: 20px;
    font-size: 0.82rem;
    font-weight: 600;
}
.badge-review {
    background-color: #FEF3C7;
    color: #92400E;
    padding: 4px 10px;
    border-radius: 20px;
    font-size: 0.82rem;
    font-weight: 600;
}
/* Touch-friendly buttons on mobile */
button {
    min-height: 40px !important;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Session State Initialization
# -----------------------------------------------------------------------------
@st.cache_resource
def init_services():
    """Initialize core singletons."""
    rag = get_rag_engine()
    db = get_db()
    quiz = get_quiz_engine(db=db)
    # Ensure default PDFs are ingested
    stats = rag.get_stats()
    if stats.get("total_chunks", 0) == 0:
        rag.ingest_default_pdfs()
    return rag, db, quiz


rag_engine, db_client, quiz_engine = init_services()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! I am your Insurance Exam Study Assistant. Ask me anything about Engineering Insurance, Fire Tariff & Policies, or Motor Insurance PE 2026-27.",
            "citations": [],
        }
    ]

if "voice_enabled" not in st.session_state:
    st.session_state.voice_enabled = True

if "speech_rate" not in st.session_state:
    st.session_state.speech_rate = 1.0

if "voice_gender" not in st.session_state:
    st.session_state.voice_gender = "female"

if "active_quiz" not in st.session_state:
    st.session_state.active_quiz = None

if "quiz_answers" not in st.session_state:
    st.session_state.quiz_answers = {}

if "quiz_result" not in st.session_state:
    st.session_state.quiz_result = None

if "viva_question" not in st.session_state:
    st.session_state.viva_question = None

if "viva_evaluation" not in st.session_state:
    st.session_state.viva_evaluation = None

if "flashcards" not in st.session_state:
    st.session_state.flashcards = []

if "flashcard_index" not in st.session_state:
    st.session_state.flashcard_index = 0

if "flashcard_flipped" not in st.session_state:
    st.session_state.flashcard_flipped = False


# -----------------------------------------------------------------------------
# Sidebar: Settings, Scope, Ingestion, & Mobile Access
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ StudyAssistant")

    # 1. AI Engine & Credentials Configuration (Cloud Run / Local)
    st.subheader("🤖 AI Engine & Credentials")

    if "gemini_api_key" not in st.session_state:
        st.session_state.gemini_api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""

    user_api_key = st.text_input(
        "🔑 Gemini API Key",
        value=st.session_state.gemini_api_key,
        type="password",
        placeholder="Paste your Gemini API key here...",
        help="Required when running on Cloud Run or when local Ollama is not available. Get one free at aistudio.google.com",
    )
    if user_api_key != st.session_state.gemini_api_key:
        st.session_state.gemini_api_key = user_api_key
        rag_engine.set_api_key(user_api_key)

    pm_status = rag_engine.provider_mgr.get_status()
    engine_options = ["Auto (Cloud/Offline Fallback)", "Google Gemini (Cloud)", "Ollama / Local (Offline)"]
    selected_engine = st.selectbox(
        "Model Provider",
        engine_options,
        index=0,
        help="Switch between cloud Gemini API and local Ollama / offline models.",
    )
    if selected_engine == "Google Gemini (Cloud)":
        rag_engine.provider_mgr.set_engine("gemini")
    elif selected_engine == "Ollama / Local (Offline)":
        rag_engine.provider_mgr.set_engine("ollama")
    else:
        rag_engine.provider_mgr.set_engine("auto")

    # Refresh status after possible mode change
    pm_status = rag_engine.provider_mgr.get_status()
    if pm_status["gemini_available"]:
        st.success(f"🟢 **Cloud LLM Connected** ({pm_status['active_model']})")
    elif pm_status["ollama_available"]:
        st.info(f"💻 **Local Ollama Connected** ({pm_status['active_model']})")
    else:
        st.warning(
            "🟡 **Offline Extractive Mode**\n\nEnter a Gemini API key above to unlock full generative AI and deep oral viva grading."
        )

    # 2. Strict Grounding Toggle
    strict_grounding = st.toggle("Strict Zero-Hallucination", value=True, help="Refuse questions not answered in the study documents.")
    allow_external = st.toggle("Allow General Knowledge", value=not strict_grounding, help="Permit broader insurance industry context.")

    # 3. Granular Study Scope Selector
    st.markdown("---")
    st.subheader("🎯 Scope Selector")
    available_subjects = ["All Subjects"] + sorted(rag_engine.get_available_subjects())
    selected_subject = st.selectbox("Subject Focus", available_subjects, index=0)

    # Granular chapter selection if a specific subject is selected
    target_subject_filter = None if selected_subject == "All Subjects" else selected_subject
    available_chapters = ["All Chapters"]
    if target_subject_filter:
        chapters = rag_engine.get_available_chapters(target_subject_filter)
        if chapters:
            available_chapters += sorted(chapters)

    selected_chapter = st.selectbox("Chapter / Topic", available_chapters, index=0)
    target_chapter_filter = None if selected_chapter == "All Chapters" else selected_chapter

    # 4. Voice Controls
    st.markdown("---")
    st.subheader("🎙️ Voice Settings")
    st.session_state.voice_enabled = st.toggle("Two-Way Voice", value=st.session_state.voice_enabled)
    if st.session_state.voice_enabled:
        st.session_state.speech_rate = st.slider("Speech Speed", 0.75, 1.5, st.session_state.speech_rate, 0.05)

    # 5. Dynamic Document Uploader
    st.markdown("---")
    with st.expander("📄 Ingest New Study Material"):
        uploaded_file = st.file_uploader("Upload PDF or TXT", type=["pdf", "txt"], help="Upload additional exam guides or tariff circulars.")
        up_subject = st.text_input("Subject Tag", value="General Insurance")
        up_chapter = st.text_input("Chapter/Topic Name (optional)", value="")
        if uploaded_file and st.button("Ingest & Index", type="primary"):
            with st.spinner("Parsing and indexing into ChromaDB..."):
                file_bytes = uploaded_file.read()
                added = rag_engine.ingest_document(
                    file_source=file_bytes,
                    file_name=uploaded_file.name,
                    subject=up_subject.strip() or "General Insurance",
                    chapter=up_chapter.strip() or None,
                )
                st.success(f"Indexed {added} chunks from '{uploaded_file.name}'!")
                st.rerun()

    # 6. Mobile Access & QR Code Launcher
    st.markdown("---")
    with st.expander("📱 Mobile Access & QR Code"):
        local_ip = get_local_ip()
        local_url = get_local_url(port=8501)
        st.write(f"**Local Wi-Fi URL:** `{local_url}`")
        qr_uri = generate_qr_code_data_uri(local_url)
        if qr_uri:
            st.image(qr_uri, caption="Scan to open on smartphone (Same Wi-Fi)", width=180)
        st.caption("Tip: For microphone permissions outside local Wi-Fi, run `cloudflared tunnel` or `npx localtunnel --port 8501` for HTTPS.")


# -----------------------------------------------------------------------------
# Main Application Tabs
# -----------------------------------------------------------------------------
st.markdown('<div class="main-title">🎓 Insurance Exam StudyAssistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Interactive AI Tutor, Timed Mock Exams, Voice Viva, and Spaced-Repetition Flashcards grounded in Fire, Motor, & Engineering Syllabi.</div>',
    unsafe_allow_html=True,
)

tab_chat, tab_quiz, tab_viva, tab_flashcards, tab_analytics = st.tabs(
    ["💬 Grounded Q&A Chat", "📝 Mock Tests (MCQs)", "🎙️ Oral Viva (Voice Exam)", "🗂️ Flashcards & Drill", "📊 Performance Analytics"]
)


# =============================================================================
# TAB 1: Grounded Q&A Chat
# =============================================================================
with tab_chat:
    st.markdown("##### Ask any question grounded strictly in your study materials")

    # Display chat history
    for idx, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("citations"):
                st.markdown(
                    "".join(
                        [
                            f'<div class="citation-box">📌 <b>Source:</b> {c.get("file_name", "Doc")} | <b>Page:</b> {c.get("page_number", "N/A")} | <b>Topic:</b> {c.get("chapter", "General")}</div>'
                            for c in msg["citations"]
                        ]
                    ),
                    unsafe_allow_html=True,
                )

    # Input row: Text and optional Web Speech mic
    user_prompt = st.chat_input("Ask a question or tariff rule (e.g. 'What is the compulsory deductible under CPM policy?')...")

    # Voice Input widget if voice enabled
    spoken_text = None
    if st.session_state.voice_enabled:
        with st.expander("🎙️ Speak your question via Microphone"):
            spoken_text = render_speech_input_component(
                key="chat_mic",
                prompt="Click 'Start Speaking' and state your insurance query...",
            )

    active_query = user_prompt or spoken_text

    if active_query:
        # Add user query to chat
        st.session_state.messages.append({"role": "user", "content": active_query})
        with st.chat_message("user"):
            st.markdown(active_query)

        # Generate grounded response
        with st.chat_message("assistant"):
            with st.spinner("Searching syllabus and formulating grounded response..."):
                response = rag_engine.query(
                    question=active_query,
                    subject=target_subject_filter,
                    chapter=target_chapter_filter,
                    top_k=4,
                    allow_external_knowledge=allow_external,
                )
                answer_text = response.get("answer", "I could not find relevant information in the documents.")
                citations = response.get("citations", [])

                st.markdown(answer_text)

                if citations:
                    st.markdown(
                        "".join(
                            [
                                f'<div class="citation-box">📌 <b>Source:</b> {c.get("file_name", "Doc")} | <b>Page:</b> {c.get("page_number", "N/A")} | <b>Topic:</b> {c.get("chapter", "General")}</div>'
                                for c in citations
                            ]
                        ),
                        unsafe_allow_html=True,
                    )

                # Voice output if enabled
                if st.session_state.voice_enabled:
                    speech_text = clean_text_for_speech(answer_text)
                    render_speech_synthesis(
                        text=speech_text,
                        autoplay=True,
                        rate=st.session_state.speech_rate,
                    )

        # Append assistant message
        st.session_state.messages.append(
            {"role": "assistant", "content": answer_text, "citations": citations}
        )
        # Log study session activity
        db_client.log_study_session(
            session_type="chat_qa",
            subject=target_subject_filter or "General",
            notes=f"Query: {active_query[:50]}...",
        )


# =============================================================================
# TAB 2: Objective Mock Tests (MCQs)
# =============================================================================
with tab_quiz:
    st.markdown("##### 📝 Timed Objective Mock Test (PE Examination Pattern)")

    col_q1, col_q2, col_q3 = st.columns([2, 1, 1])
    with col_q1:
        quiz_subject = st.selectbox(
            "Select Exam Subject",
            ["Engineering", "Fire", "Motor"],
            key="quiz_subject_select",
        )
    with col_q2:
        num_q = st.selectbox("Number of Questions", [3, 5, 10, 15], index=1)
    with col_q3:
        timer_min = st.selectbox("Timer (Minutes)", [3, 5, 10, 15], index=1)

    if st.button("Generate New Mock Test", type="primary"):
        with st.spinner("Generating syllabus-grounded exam questions..."):
            questions = quiz_engine.generate_mcq_quiz(
                subject=quiz_subject,
                num_questions=num_q,
                rag_engine=rag_engine,
            )
            st.session_state.active_quiz = questions
            st.session_state.quiz_answers = {}
            st.session_state.quiz_result = None
            st.rerun()

    # If quiz is active, render questions
    if st.session_state.active_quiz:
        st.info(f"⏱️ **Exam in Progress**: {len(st.session_state.active_quiz)} Questions | Subject: **{quiz_subject}**")

        with st.form("mock_test_form"):
            for i, q in enumerate(st.session_state.active_quiz):
                st.markdown(f"**Q{i+1}: {q.question}**")
                options_display = [f"{chr(65+idx)}. {opt}" for idx, opt in enumerate(q.options)]
                choice = st.radio(
                    f"Select answer for Q{i+1}:",
                    options_display,
                    key=f"q_choice_{i}",
                    index=None,
                    label_visibility="collapsed",
                )
                if choice:
                    selected_idx = ord(choice[0]) - 65
                    st.session_state.quiz_answers[i] = selected_idx
                st.markdown("---")

            submit_test = st.form_submit_button("Submit Exam & View Detailed Results", type="primary")

        if submit_test:
            # Score test
            results = quiz_engine.evaluate_quiz_submission(
                questions=st.session_state.active_quiz,
                user_answers=st.session_state.quiz_answers,
            )
            st.session_state.quiz_result = results

            # Log to SQLite
            db_client.log_quiz_attempt(
                subject=quiz_subject,
                total_questions=results.total_questions,
                correct_answers=results.correct_answers,
                score_percentage=results.score_percentage,
                breakdown=results.details,
            )
            st.rerun()

    # Display Quiz Results if submitted
    if st.session_state.quiz_result:
        res = st.session_state.quiz_result
        st.markdown("### 📋 Exam Scorecard & Explanations")
        badge = '<span class="badge-pass">PASS</span>' if res.score_percentage >= 60 else '<span class="badge-review">NEEDS REVISION</span>'
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-val">{res.score_percentage:.1f}% {badge}</div>
                <div class="stat-label">Score: {res.correct_answers} / {res.total_questions} Correct</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        for item in res.details:
            is_correct = item.get("is_correct", False)
            icon = "✅" if is_correct else "❌"
            st.markdown(f"**{icon} {item.get('question')}**")
            st.markdown(f"- **Your Answer:** {item.get('user_answer', 'Not answered')}")
            st.markdown(f"- **Correct Answer:** **{item.get('correct_answer')}**")
            st.markdown(f"- 💡 **Explanation:** {item.get('explanation')}")
            if item.get("citation"):
                st.markdown(
                    f'<div class="citation-box">📌 <b>Reference:</b> {item.get("citation")}</div>',
                    unsafe_allow_html=True,
                )
            st.markdown("---")


# =============================================================================
# TAB 3: Oral Viva / Spoken Exam Mode
# =============================================================================
with tab_viva:
    st.markdown("##### 🎙️ Interactive Oral Viva Examiner")
    st.caption("AI Examiner asks questions aloud. Speak your answer into the microphone. Receive instant rubric grading & missed-point feedback.")

    col_v1, col_v2 = st.columns([3, 1])
    with col_v1:
        viva_subject = st.selectbox("Viva Subject Focus", ["Motor", "Fire", "Engineering"], key="viva_sub")
    with col_v2:
        st.write("")
        st.write("")
        if st.button("Generate Viva Question", type="primary"):
            with st.spinner("AI Examiner formulating viva question..."):
                q_obj = quiz_engine.generate_viva_question(subject=viva_subject, rag_engine=rag_engine)
                st.session_state.viva_question = q_obj
                st.session_state.viva_evaluation = None
                st.rerun()

    if st.session_state.viva_question:
        v_q = st.session_state.viva_question
        st.markdown(f"### ❓ Examiner Question: *{v_q.get('topic', 'General')}*")
        st.info(f"📢 **\"{v_q.get('question')}\"**")

        # Speak question automatically
        if st.session_state.voice_enabled:
            render_speech_synthesis(
                text=v_q.get("question"),
                autoplay=True,
                rate=st.session_state.speech_rate,
            )

        st.markdown("##### 🎙️ Your Spoken Answer:")
        viva_speech = render_speech_input_component(
            key="viva_mic",
            prompt="Click to speak your viva answer aloud...",
        )

        manual_answer = st.text_area("Or type/edit your response:", value=viva_speech or "", height=100)

        if st.button("Submit Answer for Viva Evaluation", type="primary"):
            if not manual_answer.strip():
                st.warning("Please speak or type your answer before submitting.")
            else:
                with st.spinner("AI Examiner evaluating conceptual accuracy and key terms..."):
                    evaluation = quiz_engine.evaluate_viva_response(
                        viva_question=v_q,
                        student_response=manual_answer,
                        rag_engine=rag_engine,
                    )
                    st.session_state.viva_evaluation = evaluation

                    # Log to SQLite
                    db_client.log_viva_evaluation(
                        subject=viva_subject,
                        topic=v_q.get("topic", "General"),
                        question=v_q.get("question"),
                        student_response=manual_answer,
                        accuracy_score=evaluation.get("accuracy_score", 0),
                        feedback=evaluation.get("feedback", ""),
                        missed_points=evaluation.get("missed_points", []),
                    )
                    st.rerun()

    # Display Viva Evaluation
    if st.session_state.viva_evaluation:
        ev = st.session_state.viva_evaluation
        st.markdown("### 🎖️ Examiner Evaluation & Scoring")
        score = ev.get("accuracy_score", 0)
        badge = '<span class="badge-pass">EXCELLENT</span>' if score >= 80 else ('<span class="badge-pass">SATISFACTORY</span>' if score >= 60 else '<span class="badge-review">NEEDS REVISION</span>')
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-val">{score}/100 {badge}</div>
                <div class="stat-label">Conceptual Completeness & Terminology</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(f"**💬 Examiner Feedback:** {ev.get('feedback')}")

        missed = ev.get("missed_points", [])
        if missed:
            st.markdown("**⚠️ Key Omissions / Concepts to Add:**")
            for m in missed:
                st.markdown(f"- {m}")

        if ev.get("citation"):
            st.markdown(
                f'<div class="citation-box">📌 <b>Syllabus Reference:</b> {ev.get("citation")}</div>',
                unsafe_allow_html=True,
            )

        # Spoken Examiner feedback
        if st.session_state.voice_enabled and ev.get("feedback"):
            render_speech_synthesis(
                text=f"Score {score} out of 100. {clean_text_for_speech(ev.get('feedback'))}",
                autoplay=True,
                rate=st.session_state.speech_rate,
            )


# =============================================================================
# TAB 4: Flashcards & Spaced Repetition Drill
# =============================================================================
with tab_flashcards:
    st.markdown("##### 🗂️ Rapid-Fire Revision Flashcards (SuperMemo SM-2 Spaced Repetition)")

    col_f1, col_f2 = st.columns([3, 1])
    with col_f1:
        fc_subject = st.selectbox("Select Subject Deck", ["Fire", "Motor", "Engineering"], key="fc_sub")
    with col_f2:
        st.write("")
        st.write("")
        if st.button("Load / Generate Deck", type="primary"):
            with st.spinner("Extracting definitions and key tariff rules..."):
                deck = quiz_engine.generate_flashcards(subject=fc_subject, count=8, rag_engine=rag_engine)
                st.session_state.flashcards = deck
                st.session_state.flashcard_index = 0
                st.session_state.flashcard_flipped = False
                st.rerun()

    if st.session_state.flashcards:
        cards = st.session_state.flashcards
        idx = st.session_state.flashcard_index
        total = len(cards)
        current_card = cards[idx]

        st.caption(f"Card {idx+1} of {total} | Topic: **{current_card.get('topic', 'General')}**")

        # Flashcard Display Box
        card_content = current_card.get("answer") if st.session_state.flashcard_flipped else current_card.get("question")
        header_text = "💡 REVERSE (Answer / Explanation)" if st.session_state.flashcard_flipped else "❓ FRONT (Term / Definition)"

        st.markdown(
            f"""
            <div style="background-color: #F8FAFC; border: 2px solid #3B82F6; border-radius: 12px; padding: 24px; min-height: 140px; margin-bottom: 15px;">
                <div style="font-size: 0.85rem; font-weight: 700; color: #3B82F6; margin-bottom: 8px;">{header_text}</div>
                <div style="font-size: 1.25rem; font-weight: 600; color: #1E293B;">{card_content}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_b1, col_b2, col_b3, col_b4 = st.columns([2, 1, 1, 1])
        with col_b1:
            if st.button("🔄 Flip Card", key="flip_btn", type="secondary"):
                st.session_state.flashcard_flipped = not st.session_state.flashcard_flipped
                st.rerun()

        # Mastery Rating buttons (SuperMemo SM-2)
        with col_b2:
            if st.button("🟢 Easy", key="rate_easy"):
                quiz_engine.rate_flashcard(current_card.get("id"), rating=5)
                st.session_state.flashcard_index = (idx + 1) % total
                st.session_state.flashcard_flipped = False
                st.rerun()
        with col_b3:
            if st.button("🟡 Review", key="rate_review"):
                quiz_engine.rate_flashcard(current_card.get("id"), rating=3)
                st.session_state.flashcard_index = (idx + 1) % total
                st.session_state.flashcard_flipped = False
                st.rerun()
        with col_b4:
            if st.button("🔴 Hard", key="rate_hard"):
                quiz_engine.rate_flashcard(current_card.get("id"), rating=1)
                st.session_state.flashcard_index = (idx + 1) % total
                st.session_state.flashcard_flipped = False
                st.rerun()


# =============================================================================
# TAB 5: Performance Analytics & Data Export/Import
# =============================================================================
with tab_analytics:
    st.markdown("##### 📊 Study Progress & Performance Analytics")

    summary = db_client.get_analytics_summary()

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.markdown(
            f"""<div class="stat-card"><div class="stat-val">{summary.get('total_quizzes', 0)}</div><div class="stat-label">Mock Tests Taken</div></div>""",
            unsafe_allow_html=True,
        )
    with col_m2:
        avg_score = summary.get("average_quiz_score", 0.0)
        st.markdown(
            f"""<div class="stat-card"><div class="stat-val">{avg_score:.1f}%</div><div class="stat-label">Average Score</div></div>""",
            unsafe_allow_html=True,
        )
    with col_m3:
        st.markdown(
            f"""<div class="stat-card"><div class="stat-val">{summary.get('total_viva_sessions', 0)}</div><div class="stat-label">Oral Viva Drills</div></div>""",
            unsafe_allow_html=True,
        )
    with col_m4:
        st.markdown(
            f"""<div class="stat-card"><div class="stat-val">{summary.get('total_study_sessions', 0)}</div><div class="stat-label">Study Sessions</div></div>""",
            unsafe_allow_html=True,
        )

    # Subject breakdown
    st.markdown("#### 🎯 Performance by Subject")
    subj_perf = summary.get("subject_breakdown", {})
    if subj_perf:
        for subj, perf in subj_perf.items():
            col_s1, col_s2 = st.columns([2, 5])
            with col_s1:
                st.write(f"**{subj}**")
            with col_s2:
                pct = perf.get("average_score", 0)
                st.progress(min(pct / 100.0, 1.0), text=f"{pct:.1f}% ({perf.get('attempts', 0)} attempts)")
    else:
        st.info("Take mock tests or viva drills to see your subject performance breakdown.")

    # Weak areas flagged for revision
    weak_topics = summary.get("weak_topics", [])
    if weak_topics:
        st.markdown("#### ⚠️ Topics Recommended for Revision")
        for wt in weak_topics:
            st.warning(f"📌 **{wt.get('subject')} - {wt.get('topic')}**: Score {wt.get('average_score', 0):.1f}%")

    # Data Export and Backup
    st.markdown("---")
    st.markdown("#### 💾 Data Export & Backup")
    col_e1, col_e2 = st.columns(2)
    with col_e1:
        json_dump = db_client.export_to_json()
        st.download_button(
            label="📥 Export Progress to JSON",
            data=json_dump,
            file_name=f"study_assistant_backup_{datetime.now().strftime('%Y%m%d')}.json",
            mime="application/json",
        )
    with col_e2:
        md_report = db_client.generate_summary_report()
        st.download_button(
            label="📄 Download Markdown Revision Report",
            data=md_report,
            file_name=f"exam_revision_report_{datetime.now().strftime('%Y%m%d')}.md",
            mime="text/markdown",
        )
