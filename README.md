# 📄 AI-Powered Resume Screener & Ranker

An automated hiring tool that uses Natural Language Processing (NLP) and Machine Learning to screen, score, and rank candidate resumes against a job description.

**Build By:**
* **Priyanshu Kumawat** - 22BCON1393
* **Rohit Kumar Saini** - 22BCON1431
* **Hemant Singh** - 22BCON1402
* **Manjeet Jahkar** - 22BCON1454

---

## 🚀 Key Features
* **Semantic Analysis:** Uses `sentence-transformers` to compute the Cosine Similarity between the Job Description and Resume, detecting relevant candidates even if they use different terminology.
* **Keyword Verification:** Checks for mandatory hard skills (e.g., Python, SQL) using Regex to ensure technical compliance.
* **Experience Parsing:** Automatically extracts years of experience from text patterns to weight seniority.
* **Ranked Output:** Generates a clean CSV report (`ranking_report.csv`) sorting candidates from highest to lowest fit.
  
  |Candidate|Final Score|AI Score|Keyword Match|Experience Score|Missing Skills|
  |---|---|---|---|---|---|
  |Ashish Ohri\.pdf|87\.94|39\.7|100\.0|100\.0|None|
  |Juan Josecarin\.pdf|77\.83|39\.2|66\.7|100\.0|Pandas, NLP|
  |Saran Wong\.pdf|65\.76|28\.8|33\.3|100\.0|Machine Learning, Scikit-Learn, Pandas, NLP|
  |Bhavesh Wadhwani\.pdf|56\.77|33\.9|83\.3|50\.0|Machine Learning|
  |Yunlong Jiao\.pdf|37\.55|37\.7|100\.0|0\.0|None|
  |Zain Khalid\.pdf|36\.67|33\.3|16\.7|50\.0|Python, Machine Learning, Scikit-Learn, Pandas, NLP|
  |Sunmarg\_resume\.pdf|28\.55|17\.8|0\.0|50\.0|Python, Machine Learning, Scikit-Learn, Pandas, NLP, SQL|
  |Deep Mehta\.pdf|20\.88|29\.4|50\.0|0\.0|Scikit-Learn, Pandas, NLP|
  |FLorne\.pdf|8\.58|17\.9|16\.7|0\.0|Python, Machine Learning, Scikit-Learn, Pandas, NLP|
  |Kartik tomer\.pdf|4\.11|20\.6|0\.0|0\.0|Python, Machine Learning, Scikit-Learn, Pandas, NLP, SQL|

## 🛠️ Tech Stack
* **Language:** Python 3.x
* **NLP Model:** `all-MiniLM-L6-v2` (via Sentence-Transformers)
* **PDF Parsing:** `pypdf`
* **Data Handling:** `pandas`
* **Pattern Matching:** `re` (Regular Expressions)

## ⚙️ How It Works (The Logic)
The system calculates a **Total Score** (0-100%) based on a weighted average of three metrics:

1.  **Semantic Score (20%):** The text is converted into high-dimensional vectors (Embeddings). We calculate the **Cosine Similarity** between the Job Description vector and the Resume vector.
2.  **Keyword Match Score (30%):** Checks for the presence of required skills defined in the configuration.
3.  **Experience Score (50%):** Extracts "Years of Experience" (e.g., "5+ years") and normalizes it against the job requirement.

## 📂 Project Structure
```bash
Resume-Screener/
├── models/                  # Contains SBERT model files
├── resumes/                 # Folder to store candidate PDF files
│   ├── candidate_1.pdf
│   └── candidate_2.pdf
├── main.py                  # Main script containing the logic
├── requirements.txt         # List of dependencies
├── ranking_report.csv       # Output file (Generated after running)
├── LICENSE                  # MIT License
└── README.md                # Documentation
