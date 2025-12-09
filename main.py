import os
import pandas as pd
import re
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer, util
import torch

# --- Paths, Configuration & Constants ---

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESUME_FOLDER = os.path.join(BASE_DIR, "resumes")
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_CSV = os.path.join(BASE_DIR, "hybrid_ranking_final.csv")

JOB_DESCRIPTION = """
We are looking for a Data Scientist with strong Python skills.
Experience with Machine Learning libraries like Scikit-Learn and Pandas is a must.
Knowledge of NLP and SQL is a plus.
"""

REQUIRED_SKILLS = {
    "Python": ["python", "py"],
    "Machine Learning": ["machine learning", "ml", "deep learning", "neural networks", "artificial intelligence", "ai"],
    "Scikit-Learn": ["scikit-learn", "sklearn", "sci-kit learn"],
    "Pandas": ["pandas", "pd"],
    "NLP": ["nlp", "natural language processing", "text analytics", "text mining"],
    "SQL": ["sql", "mysql", "postgresql", "mssql", "t-sql", "database queries"]
}

SBERT_MODEL_NAME = 'all-MiniLM-L6-v2'
SEMANTIC_WEIGHT = 0.20
KEYWORD_WEIGHT = 0.30
EXPERIENCE_WEIGHT = 0.50

# --- Helper Functions ---

def get_keyword_score(resume_text, required_skills_dict):
    """
    Calculates percentage of required skills (including synonyms) present in the resume.
    """
    resume_lower = resume_text.lower()
    matched_primary_skills_count = 0
    missing_primary_skills = []

    for primary_skill, synonyms in required_skills_dict.items():
        found_skill = False
        for synonym in synonyms:
            if re.search(r'\b' + re.escape(synonym.lower()) + r'\b', resume_lower):
                found_skill = True
                break

        if found_skill:
            matched_primary_skills_count += 1
        else:
            missing_primary_skills.append(primary_skill)

    total_primary_skills = len(required_skills_dict)
    if total_primary_skills == 0:
        return 0.0, []

    return (matched_primary_skills_count / total_primary_skills), missing_primary_skills

def extract_text_from_pdf(pdf_path):
    """
    Reads text from a PDF file.
    """
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text.replace('\n', ' ').strip()
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        return ""

def get_semantic_score(resume_text, jd_embedding, model):
    """
    Uses the "Sliding Window" chunking method to compare resume against JD.
    """
    window_size = 200
    overlap = 50
    words = resume_text.split()

    if len(words) <= window_size:
        resume_embedding = model.encode(resume_text, convert_to_tensor=True)
        return util.pytorch_cos_sim(jd_embedding, resume_embedding).item()

    chunks = []
    for i in range(0, len(words), window_size - overlap):
        chunk_words = words[i: i + window_size]
        chunks.append(" ".join(chunk_words))

    if not chunks:
        return 0.0

    chunk_embeddings = model.encode(chunks, convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(jd_embedding, chunk_embeddings)
    return torch.max(cosine_scores).item()

def get_experience_score(resume_text):
    """
    Extracts years of relevant experience from resume text and returns a score (0.0 to 1.0).
    """
    experience_years = 0

    patterns = [
        r'(\d+)\s*\+?\s*(?:years|yrs)\s*(?:of\s*experience|experience)',
        r'(?:over|more\s*than)\s*(\d+)\s*(?:years|yrs)\s*(?:of\s*experience|experience)',
        r'(\d+)\s*-\s*\d+\s*(?:years|yrs)\s*(?:of\s*experience|experience)',
        r'(?:total\s*of|a\s*total\s*of)\s*(\d+)\s*(?:years|yrs)',
        r'(\d+)(?:\s*\+\s*year(?:s)?|\s*\+)(?:\s*of\s*experience|\s*experience)?'
    ]

    for pattern in patterns:
        match = re.search(pattern, resume_text, re.IGNORECASE)
        if match:
            try:
                exp = int(match.group(1))
                experience_years = max(experience_years, exp)
            except ValueError:
                pass

    if experience_years >= 10:
        score = 1.0
    elif experience_years >= 6:
        score = 0.8
    elif experience_years >= 3:
        score = 0.5
    elif experience_years >= 1:
        score = 0.2
    else:
        score = 0.0

    return score

# --- Main Execution ---

def run_screener():
    # Ensure required directories exist (creates models/ under project)
    os.makedirs(RESUME_FOLDER, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("Loading AI Model...")
    model = SentenceTransformer(SBERT_MODEL_NAME, cache_folder=MODEL_DIR)
    jd_embedding = model.encode(JOB_DESCRIPTION, convert_to_tensor=True)

    results = []
    print(f"Scanning resumes in '{RESUME_FOLDER}'...")

    if not os.path.exists(RESUME_FOLDER):
        print(f"Error: Directory '{RESUME_FOLDER}' does not exist.")
        return

    files = [f for f in os.listdir(RESUME_FOLDER) if f.endswith('.pdf')]

    for filename in files:
        filepath = os.path.join(RESUME_FOLDER, filename)
        resume_text = extract_text_from_pdf(filepath)

        if resume_text:
            semantic = get_semantic_score(resume_text, jd_embedding, model)
            keyword_score, missing_skills = get_keyword_score(resume_text, REQUIRED_SKILLS)
            experience_score = get_experience_score(resume_text)

            final_score = (
                semantic * SEMANTIC_WEIGHT
                + keyword_score * KEYWORD_WEIGHT
                + experience_score * EXPERIENCE_WEIGHT
            )

            results.append({
                "Candidate": filename,
                "Final Score": round(final_score * 100, 2),
                "AI Score": round(semantic * 100, 1),
                "Keyword Match": round(keyword_score * 100, 1),
                "Experience Score": round(experience_score * 100, 1),
                "Missing Skills": ", ".join(missing_skills) if missing_skills else "None"
            })

    if results:
        df = pd.DataFrame(results)
        df = df.sort_values(by="Final Score", ascending=False)
        print("\n--- ⭐ HYBRID RANKING RESULTS (WITH CONFIGURABLE WEIGHTS) ⭐ ---")
        print(df.to_string(index=False))
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"\nSaved results to: {OUTPUT_CSV}")
    else:
        print("No resumes found or processed.")

if __name__ == "__main__":
    run_screener()
