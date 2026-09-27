import os
import sys
from typing import List, Dict, Any
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Import Step Modules
from steps.step03_clean_text import clean_transcript
from steps.step05_tokenize import tokenize_transcript
from steps.step06_stop_words import remove_stop_words
from steps.step07_stem_lemmatize import stem_and_lemmatize
from steps.step08_pos_tagging import tag_pos
from steps.step09_ner import extract_entities, group_entities_by_type
from steps.step10_tfidf_keyphrases import extract_tfidf_features
from steps.step11_embeddings_similarity import compute_sentence_embeddings
from steps.step12_score_sentences import score_sentences
from steps.step14_remove_redundancy import remove_redundancy_and_order

from models.abstractive_summarizer import run_abstractive_inference
from models.fact_checker import fact_check_summary, get_nlp, _sentence_embedder

def evaluate_coherence(sentences: List[str]) -> float:
    """
    Evaluates coherence by computing the average cosine similarity between consecutive sentences.
    A completely disjointed text will have score ~0-0.2. A cohesive text ~0.4+.
    Returns a score 0-100.
    """
    if len(sentences) < 2:
        return 100.0
    
    from sentence_transformers import SentenceTransformer
    global _sentence_embedder
    if _sentence_embedder is None:
        _sentence_embedder = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        
    embeddings = _sentence_embedder.encode(sentences)
    similarities = []
    details = []
    for i in range(len(embeddings) - 1):
        sim = float(cosine_similarity([embeddings[i]], [embeddings[i+1]])[0][0])
        similarities.append(sim)
        details.append({
            "sent1": sentences[i],
            "sent2": sentences[i+1],
            "sim": sim
        })
        
    avg_sim = float(np.mean(similarities))
    # Map typical cosine similarity ranges (0.1 to 0.7) to a 0-100 scale for UI aesthetics
    coherence_score = max(0, min(100, int((avg_sim * 100) * 1.5)))
    return coherence_score, details

def run_doc_pipeline(
    doc_contents: List[str],
    doc_names: List[str],
    mode: str = "extractive",
    summary_length: str = "medium",
    progress_callback = None
) -> Dict[str, Any]:
    """
    Academic Multi-Document Summarization Pipeline.
    """
    if progress_callback: progress_callback(1, 10, "Parsing Documents")
    
    # 1. Multi-Document parsing & merging
    raw_sentences = []
    total_docs = len(doc_contents)
    
    import nltk
    for i, content in enumerate(doc_contents):
        paragraphs = [p for p in content.split('\n') if p.strip()]
        for p_idx, para in enumerate(paragraphs, 1):
            try:
                sents = nltk.sent_tokenize(para)
            except:
                sents = [s.strip() for s in para.split('.') if len(s.strip()) > 5]
                
            for s in sents:
                if len(s.split()) > 3:
                    raw_sentences.append({
                        "text": s, 
                        "doc_id": doc_names[i], 
                        "start": float(len(raw_sentences)),  # Enables chronological sorting
                        "duration": 0.0,
                        "para_num": p_idx,
                        "line_num": len(raw_sentences) + 1
                    })
                
    if not raw_sentences:
        raise ValueError("No valid text found in uploaded documents.")

    # Run core NLP pipeline
    if progress_callback: progress_callback(2, 10, "Cleaning & Tokenizing")
    cleaned = clean_transcript(raw_sentences, verbose=False)
    tokenized = tokenize_transcript(cleaned, verbose=False)
    
    if progress_callback: progress_callback(3, 10, "NLP Tagging & Entities")
    no_stop = remove_stop_words(tokenized, verbose=False)
    lemmatized = stem_and_lemmatize(no_stop, verbose=False)
    pos_tagged = tag_pos(lemmatized, verbose=False)
    entities_sents = extract_entities(pos_tagged, verbose=False)
    
    if progress_callback: progress_callback(4, 10, "Keyword Extraction (TF-IDF)")
    tfidf_sents, top_keywords = extract_tfidf_features(entities_sents, verbose=False)
    
    if progress_callback: progress_callback(5, 10, "Computing Semantic Embeddings")
    embedded_sents, sim_matrix = compute_sentence_embeddings(tfidf_sents, verbose=False)
    
    if progress_callback: progress_callback(6, 10, "Sentence Ranking")
    scored = score_sentences(embedded_sents, sim_matrix, verbose=False)
    
    if progress_callback: progress_callback(7, 10, "Redundancy Removal")
    initial_count = len(scored)
    non_redundant, dropped_details = remove_redundancy_and_order(scored, verbose=False, return_dropped=True)
    final_count = len(non_redundant)
    redundancy_stats = {
        "initial": initial_count,
        "final": final_count,
        "removed": initial_count - final_count,
        "dropped_details": dropped_details
    }
    
    # 8. Extractive Summarization
    if progress_callback: progress_callback(8, 10, "Generating Extractive Summary")
    
    # Determine base sentence count based on requested length
    if mode == "extractive":
        num_sentences = len(non_redundant) # No limit for extractive mode
    else:
        num_sentences = {"small": 8, "medium": 15, "long": 30}.get(summary_length, 15)
    
    # Sort by score first
    sorted_by_score = sorted(non_redundant, key=lambda x: x.get("composite_score", 0), reverse=True)
    
    # To prevent starvation of smaller documents when mixing large ones,
    # let's try to guarantee at least 3 sentences per document if available
    top_n = []
    per_doc_counts = {doc: 0 for doc in doc_names}
    
    # Pass 1: Guarantee minimum representation
    for s in sorted_by_score:
        doc = s.get("doc_id")
        if per_doc_counts[doc] < 3 and len(top_n) < num_sentences:
            top_n.append(s)
            per_doc_counts[doc] += 1
            
    # Pass 2: Fill the rest purely by highest score
    for s in sorted_by_score:
        if len(top_n) >= num_sentences:
            break
        if s not in top_n:
            top_n.append(s)
            
    # Sort the final selected sentences by score for the UI ranking view
    top_n_for_ranking = sorted(top_n, key=lambda x: x.get("composite_score", 0), reverse=True)
    
    # Restore chronological order for the Extractive Summary text
    from steps.step14_remove_redundancy import restore_chronological_order
    top_n_ordered = restore_chronological_order(top_n)
    
    # Group extractive summary by document for readability
    doc_groups = {}
    for s in top_n_ordered:
        d_id = s.get("doc_id", "Document")
        if d_id not in doc_groups:
            doc_groups[d_id] = []
        doc_groups[d_id].append(s["text"])
        
    extractive_summary_text = ""
    for d_id, sents in doc_groups.items():
        extractive_summary_text += f"**From {d_id}:**\n" + " ".join(sents) + "\n\n"
    extractive_summary_text = extractive_summary_text.strip()
    
    # 9. Abstractive Compression
    abstractive_summary_text = ""
    if mode == "abstractive":
        if progress_callback: progress_callback(9, 10, "Abstractive Compression (DistilBART)")
        abstractive_summary_text = run_abstractive_inference(
            extractive_summary_text, 
            model_name="sshleifer/distilbart-cnn-12-6",
            max_len=250, 
            min_len=80, 
            fast_mode=True
        )
    
    if progress_callback: progress_callback(10, 10, "Evaluating Coherence & Consistency")
    
    # Coherence Evaluation
    summary_to_eval = abstractive_summary_text if mode == "abstractive" else extractive_summary_text
    import re
    eval_sents = [s.strip() for s in re.split(r'(?<=[.!?]) +', summary_to_eval) if s.strip()]
    coherence_score, coherence_details = evaluate_coherence(eval_sents)
    
    # Factual Consistency
    fact_check = fact_check_summary(summary_to_eval, tokenized, verbose=False)
    
    results = {
        "num_docs": total_docs,
        "total_sentences": initial_count,
        "top_keywords": top_keywords,
        "ranked_sentences": top_n_for_ranking,
        "redundancy_stats": redundancy_stats,
        "extractive_summary": extractive_summary_text,
        "abstractive_summary": abstractive_summary_text,
        "coherence_score": coherence_score,
        "coherence_details": coherence_details,
        "fact_check": fact_check,
        "mode": mode
    }
    return results
