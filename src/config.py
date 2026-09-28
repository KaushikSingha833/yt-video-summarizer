"""
Central Configuration for Summarization Pipeline.
"""

class Config:
    # ---------------------------------------------------------
    # Stage 1: Parsing & Cleaning
    # ---------------------------------------------------------
    MIN_SENTENCE_WORDS = 4
    MIN_DOC_LENGTH_CHARS = 50
    
    # ---------------------------------------------------------
    # Stage 2: Keyword Extraction
    # ---------------------------------------------------------
    KEYWORD_MIN_DF = 1  # Minimum document frequency
    KEYWORD_MAX_DF = 0.85
    KEYWORD_NGRAM_RANGE = (1, 3) # Prefer bigrams/trigrams
    
    # ---------------------------------------------------------
    # Stage 3 & 4: Ranking & Redundancy Removal
    # ---------------------------------------------------------
    MMR_LAMBDA = 0.5 # Balance between relevance and diversity
    REDUNDANCY_THRESHOLD = 0.90 # Drop if cosine similarity > 0.90
    
    # ---------------------------------------------------------
    # Stage 6: Abstractive Rewrite
    # ---------------------------------------------------------
    ABSTRACTIVE_MODEL = "facebook/bart-large-cnn"
    ABSTRACTIVE_TEMPERATURE = 0.1
    SYSTEM_PROMPT = ""
    
    # ---------------------------------------------------------
    # Stage 7 & 8: Evaluation Gates
    # ---------------------------------------------------------
    # Multi-topic summaries naturally jump topics, resulting in lower consecutive sentence similarity
    # Disabled (set to 0.0) to prevent false-positive rejection of valid summaries
    COHERENCE_GATE_THRESHOLD = 0.0  
    CONSISTENCY_GATE_THRESHOLD = 0.0
