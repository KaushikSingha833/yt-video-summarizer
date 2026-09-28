# ===== STREAMLIT USER INTERFACE =====

import os
import sys
import streamlit as st

import spacy
if not spacy.util.is_package('en_core_web_sm'):
    spacy.cli.download('en_core_web_sm')

import nltk
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('punkt_tab')

# Set page config with wide layout and custom title
st.set_page_config(
    page_title="Multilingual Hybrid Summarizer Studio",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Ensure src directory is in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# Import pipelines
from pipeline import run_hybrid_pipeline
from doc_pipeline import run_doc_pipeline

# Custom CSS for modern glassmorphism aesthetic and typography
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Outfit', 'Inter', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 50%, #06b6d4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.4rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .sub-header {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }

    .glass-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1.25rem;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
    }

    .timestamp-badge {
        background-color: #2563eb;
        color: #ffffff;
        padding: 0.25rem 0.6rem;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
        margin-right: 0.5rem;
    }

    .entity-tag {
        display: inline-block;
        background: rgba(59, 130, 246, 0.15);
        color: #60a5fa;
        border: 1px solid rgba(59, 130, 246, 0.3);
        border-radius: 20px;
        padding: 0.2rem 0.7rem;
        font-size: 0.85rem;
        margin: 0.2rem;
    }
    
    .metric-box {
        background: rgba(15, 23, 42, 0.6);
        border-left: 4px solid #3b82f6;
        padding: 1rem;
        border-radius: 4px;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

def render_video_summarizer():
    # Sidebar Controls for Video
    with st.sidebar:
        st.header("⚙️ Video Configuration")

        target_lang = st.selectbox(
            "Output Language",
            options=["English", "Hindi", "French (Bonus)", "German (Bonus)"],
            index=0,
            help="Language in which the final summary will be presented."
        )
        lang_code_map = {
            "English": "en",
            "Hindi": "hi",
            "French (Bonus)": "fr",
            "German (Bonus)": "de"
        }
        chosen_lang_code = lang_code_map[target_lang]

        length_option = st.selectbox(
            "Summary Length",
            options=["Small (~100 words)", "Medium (~250 words)", "Long (~500 words)"],
            index=1
        )
        length_key_map = {
            "Small (~100 words)": "small",
            "Medium (~250 words)": "medium",
            "Long (~500 words)": "long"
        }
        chosen_length_key = length_key_map[length_option]

        mode = st.radio(
            "Summarization Engine Mode",
            options=["Fast Mode (Single Model)", "Consensus Mode (Multi-Model Fusion)"],
            index=0,
            help="Fast mode uses DistilBART; Consensus mode runs multi-model agreement voting."
        )
        is_fast = (mode == "Fast Mode (Single Model)")
        st.markdown("---")

    # Main Input Card
    with st.container():
        st.subheader("1. Enter Video Source")
        col_input, col_demos = st.columns([3, 2])

        if "video_input_val" not in st.session_state:
            st.session_state["video_input_val"] = "en_short_01"

        with col_input:
            video_input = st.text_input(
                "YouTube Video URL or Demo ID:",
                value=st.session_state["video_input_val"],
                placeholder="https://www.youtube.com/watch?v=... or en_short_01",
                help="Paste any YouTube video link or use one of the offline demo samples below."
            )

        with col_demos:
            st.write("Or pick a pre-cached offline sample:")
            demo_cols = st.columns(3)
            if demo_cols[0].button("EN Short (3m)"):
                st.session_state["video_input_val"] = "en_short_01"
                st.rerun()
            if demo_cols[1].button("EN Lecture"):
                st.session_state["video_input_val"] = "en_medium_01"
                st.rerun()
            if demo_cols[2].button("HI Talk (5m)"):
                st.session_state["video_input_val"] = "hi_clean_01"
                st.rerun()

        summarize_clicked = st.button("🚀 Summarize Video", type="primary", use_container_width=True)

    # Process and Output
    if summarize_clicked and video_input:
        progress_bar = st.progress(0, text="Initializing Pipeline...")
        def update_progress(step_num, total_steps, message):
            pct = int((step_num / total_steps) * 100)
            remaining = 100 - pct
            progress_bar.progress(pct, text=f"Stage {step_num}/{total_steps} ({remaining}% remaining): {message}")

        with st.spinner("Executing classical extraction & neural summarization pipeline..."):
            try:
                results = run_hybrid_pipeline(
                    video_input=video_input.strip(),
                    target_language=chosen_lang_code,
                    summary_length=chosen_length_key,
                    fast_mode=is_fast,
                    verbose=False,
                    progress_callback=update_progress
                )

                progress_bar.progress(100, text="✅ Pipeline Execution Complete!")
                st.success("✅ Summary Generated Successfully!")

                video_id = results.get("video_id", "")
                base_yt_url = f"https://www.youtube.com/watch?v={video_id}" if video_id and not video_id.startswith("en_") and not video_id.startswith("hi_") else ""

                st.markdown("---")
                st.markdown(f"### 🎬 {results.get('video_title', 'Video Summary')}")
                t_col1, t_col2, t_col3, t_col4 = st.columns(4)
                t_col1.metric("Duration", results.get("duration_formatted", "N/A"))
                t_col2.metric("Detected Language", results.get("detected_language", "en").upper())
                t_col3.metric("Output Language", results.get("target_language", "en").upper())
                t_col4.metric("Word Count", f"{results.get('word_count', 0)} words")

                st.markdown("---")
                st.markdown("### 📌 TL;DR")
                st.info(results.get("tldr", ""))

                if chosen_length_key != "small" and results.get("chapters_summary"):
                    st.markdown("### 📑 Chapters & Discussion")
                    for ch in results.get("chapters_summary", []):
                        timestamp_str = ch["timestamp"]
                        secs = ch.get("seconds", 0)
                        yt_link = f"{base_yt_url}&t={secs}" if base_yt_url else f"#{timestamp_str}"
                        title_str = ch["title"]
                        ch_header = f"<a href='{yt_link}' target='_blank' style='text-decoration:none; color:#3b82f6; font-weight:700;'>[{timestamp_str}]</a> **{title_str}**"
                        st.markdown(ch_header, unsafe_allow_html=True)
                        if chosen_length_key == "long":
                            st.write(ch.get("paragraph", ""))
                        else:
                            for bullet in ch.get("bullets", []):
                                st.markdown(f"- {bullet}")
                        st.write("")

                st.markdown("### 💡 Key Takeaways")
                for tk in results.get("key_takeaways", []):
                    timestamp_str = tk["timestamp"]
                    secs = tk.get("seconds", 0)
                    yt_link = f"{base_yt_url}&t={secs}" if base_yt_url else f"#{timestamp_str}"
                    takeaway_line = f"<a href='{yt_link}' target='_blank' style='text-decoration:none; color:#2563eb; font-weight:600;'>[{timestamp_str}]</a> {tk['text']}"
                    st.markdown(f"- {takeaway_line}", unsafe_allow_html=True)

                st.markdown("---")
                kw_col, ent_col = st.columns(2)
                with kw_col:
                    st.markdown("### 🏷️ Top Keywords")
                    if results.get("top_keywords"):
                        kw_tags = "".join([f"<span class='entity-tag'>#{kw}</span>" for kw, _ in results["top_keywords"][:12]])
                        st.markdown(kw_tags, unsafe_allow_html=True)
                    else:
                        st.write("No dominant keywords extracted.")

                with ent_col:
                    st.markdown("### 👥 Named Entities")
                    raw_entities = results.get("entities", {})
                    grouped = {
                        "People": raw_entities.get("PERSON", []),
                        "Organizations": raw_entities.get("ORG", []) + raw_entities.get("ORGANIZATION", []),
                        "Places": raw_entities.get("GPE", []) + raw_entities.get("LOC", []) + raw_entities.get("LOCATION", [])
                    }
                    has_any_entity = False
                    for group_name, items in grouped.items():
                        unique_items = list(dict.fromkeys(items))
                        if unique_items:
                            has_any_entity = True
                            st.markdown(f"**{group_name}**:")
                            tags_html = "".join([f"<span class='entity-tag'>{e}</span>" for e in unique_items[:8]])
                            st.markdown(tags_html, unsafe_allow_html=True)
                    if not has_any_entity:
                        st.write("No distinct named entities detected.")

                st.markdown("---")
                
                with st.expander("🔍 Compare methods (Hybrid / NLP Pipeline Only / Models Only)", expanded=False):
                    c_tab1, c_tab2, c_tab3 = st.tabs([
                        "✨ Hybrid (Part A + B)",
                        "📑 NLP Pipeline Only (Part A Extractive)",
                        "🤖 Models Only (Direct Neural Baseline)"
                    ])
                    with c_tab1:
                        st.markdown("**Hybrid Result (Primary Output):**")
                        st.info(results.get("hybrid_summary", ""))
                    with c_tab2:
                        st.markdown("**Part A Classical Extractive Baseline:**")
                        st.write(results.get("extractive_summary", ""))
                    with c_tab3:
                        st.markdown("**Part B Direct Neural Baseline:**")
                        st.write(results.get("model_only_summary", ""))

            except Exception as e:
                st.error(f"❌ Error processing video: {str(e)}")

def render_doc_summarizer():
    # Sidebar Controls for Document
    with st.sidebar:
        st.header("⚙️ Document Configuration")
        doc_mode = st.radio(
            "Document Summarizer Mode",
            options=["Extractive Mode", "Abstractive Mode"],
            index=1,
            help="Extractive pulls exact sentences. Abstractive uses AI to rewrite them."
        )
        is_doc_abstractive = (doc_mode == "Abstractive Mode")
        
        doc_length_option = st.selectbox(
            "Document Summary Length",
            options=["Small (~100 words)", "Medium (~250 words)", "Long (~500 words)"],
            index=1,
            key="doc_length_select"
        )
        doc_length_key = {
            "Small (~100 words)": "small",
            "Medium (~250 words)": "medium",
            "Long (~500 words)": "long"
        }[doc_length_option]
        st.markdown("---")
        st.caption("Capstone Project: Group 6")

    st.subheader("📑 Intelligent Multi-Document Summarization System")
    st.write("Upload one or multiple text documents. The system will merge them, remove redundant information, and extract the core facts.")
    
    uploaded_files = st.file_uploader("Upload Documents (.txt, .pdf, .docx)", type=["txt", "pdf", "docx"], accept_multiple_files=True)
    
    def extract_text(file) -> str:
        name = file.name.lower()
        if name.endswith(".pdf"):
            import pypdf
            import re
            reader = pypdf.PdfReader(file)
            text = "\n".join([page.extract_text() or "" for page in reader.pages])
            
            # 1. Fix line-wrap hyphenation (e.g., "summa-\nrization" -> "summarization")
            text = re.sub(r'-\n\s*', '', text)
            
            # 2. Remove stray symbols often from PDF bullets/artifacts
            text = re.sub(r'[□•▪►❖]+', '', text)
            
            # 3. Clean common PDF extraction spacing artifacts
            text = re.sub(r'\b(\w)\s+(\w{2,})\b', r'\1\2', text)
            
            # 4. Preserve true paragraphs but collapse line wraps
            text = re.sub(r'[ \t]+', ' ', text)
            text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
            text = re.sub(r'\n{3,}', '\n\n', text)
            
            return text.strip()
        elif name.endswith(".docx"):
            import docx
            doc = docx.Document(file)
            return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
        else:
            return file.read().decode("utf-8", errors="ignore")
    
    if st.button("🚀 Summarize Documents", type="primary", use_container_width=True):
        if not uploaded_files:
            st.warning("Please upload at least one document.")
            return
            
        doc_contents = []
        doc_names = []
        for file in uploaded_files:
            doc_contents.append(extract_text(file))
            doc_names.append(file.name)
            
        progress_bar = st.progress(0, text="Initializing Document Pipeline...")
        def update_progress(step_num, total_steps, message):
            pct = int((step_num / total_steps) * 100)
            progress_bar.progress(pct, text=f"Stage {step_num}/{total_steps}: {message}")
            
        with st.spinner("Running Multi-Document Pipeline..."):
            try:
                results = run_doc_pipeline(
                    doc_contents=doc_contents,
                    doc_names=doc_names,
                    mode="abstractive" if is_doc_abstractive else "extractive",
                    summary_length=doc_length_key,
                    progress_callback=update_progress
                )
                
                progress_bar.progress(100, text="✅ Document Processing Complete!")
                
                st.markdown("---")
                
                # 1. Multi-document summarization
                st.markdown("### 1️⃣ Multi-Document Processing")
                st.info(f"Successfully parsed **{results['num_docs']} documents** containing **{results['total_sentences']} total sentences**.")
                
                # 2. Keyword extraction
                st.markdown("### 2️⃣ Keyword Extraction (TF-IDF)")
                if results.get("top_keywords"):
                    kw_tags = "".join([f"<span class='entity-tag'>#{kw}</span>" for kw, _ in results["top_keywords"][:15]])
                    st.markdown(kw_tags, unsafe_allow_html=True)
                else:
                    st.write("No dominant keywords extracted.")
                    
                # 3. Sentence ranking
                st.markdown("### 3️⃣ Sentence Ranking (Embeddings)")
                with st.expander("View Top 10 Ranked Sentences Across Corpus", expanded=False):
                    for i, s in enumerate(results["ranked_sentences"]):
                        doc_label = s.get('doc_id', 'Document')
                        para = s.get('para_num', '?')
                        line = s.get('line_num', '?')
                        score = s.get('composite_score', 0)
                        st.markdown(f"**#{i+1} [Score: {score:.2f}]** ({doc_label} | Para {para}, Line {line}): {s['text']}")
                
                # 4. Redundancy removal
                st.markdown("### 4️⃣ Redundancy Removal")
                r_stats = results["redundancy_stats"]
                st.warning(f"**Semantic Duplicate Filtering:** Removed **{r_stats['removed']} redundant sentences** across the documents. (Reduced from {r_stats['initial']} to {r_stats['final']} unique core sentences).")
                if r_stats.get("dropped_details"):
                    with st.expander("View Dropped Redundancy Logs"):
                        for drop in r_stats["dropped_details"]:
                            cand, chosen, sim = drop
                            st.write(f"**Dropped:** {cand['text']}")
                            st.caption(f"Reason: Matched \"{chosen['text'][:80]}...\" with {sim:.2f} similarity.")
                
                # 5. Extractive summarization
                st.markdown("### 5️⃣ Extractive Summarization")
                st.write(results["extractive_summary"])
                
                # 6. Summary compression (Abstractive only)
                if is_doc_abstractive:
                    st.markdown("### 6️⃣ Summary Compression (Abstractive Rewrite)")
                    if results["mode"] == "extractive" and not results["abstractive_summary"]:
                        st.warning("⚠️ **QA Gate Triggered:** The AI attempted an Abstractive rewrite, but the generated text failed the Coherence/Consistency thresholds. The system automatically discarded it and fell back to the safe Extractive output above to prevent hallucinations.")
                    else:
                        st.success(results["abstractive_summary"])
                
                # 7. Coherence evaluation
                st.markdown("### 7️⃣ Coherence Evaluation")
                c_score = results["coherence_score"]
                st.metric("Sentence-to-Sentence Semantic Coherence Score", f"{c_score}%")
                if results.get("coherence_details"):
                    with st.expander("View Sentence Flow Analysis"):
                        for pair in results["coherence_details"]:
                            st.write(f"**Flow Transition (Similarity: {pair['sim']:.2f}):**")
                            st.caption(f"1. {pair['sent1']}")
                            st.caption(f"2. {pair['sent2']}")
                            st.divider()
                
                # 8. Factual consistency
                st.markdown("### 8️⃣ Factual Consistency Analysis")
                fc = results["fact_check"]
                st.metric("Factual Alignment w/ Source", f"{fc['overall_consistency_pct']}%")
                with st.expander("View Detailed Fact Check Report", expanded=False):
                    for item in fc.get("details", []):
                        st.write(f"- **Sentence:** {item['sentence']}")
                        st.write(f"  - **Status:** {item['status']}")
                        st.write(f"  - **Supporting Source:** {item['best_source_match']}")
                        st.write(f"  - **Similarity:** {item['support_score']:.2f}")
                        
            except Exception as e:
                st.error(f"❌ Error processing documents: {str(e)}")

def main():
    # Header Banner
    st.markdown('<div class="main-header">🧠 NLP Capstone Summarizer Studio</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Multi-Document Analysis & Hybrid YouTube Summarization (Group 6)</div>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["📺 YouTube Video Summarizer", "📑 Multi-Document Summarizer"])
    
    with tab1:
        render_video_summarizer()
        
    with tab2:
        render_doc_summarizer()

if __name__ == "__main__":
    main()
