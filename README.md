# 🏥 Clinical Trial Matcher — AI Powered

> Intelligent patient-to-clinical-trial matching using **BioBERT embeddings + TF-IDF hybrid retrieval** on real ClinicalTrials.gov data — with a live interactive Gradio demo.

<div align="center">

[![Live Demo](https://img.shields.io/badge/🚀%20Live%20Demo-Hugging%20Face%20Spaces-orange?style=for-the-badge)](https://huggingface.co/spaces/yashikaSharma/clinical-trial-matcher)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-black?style=for-the-badge&logo=github)](https://github.com/yashikasharma2004/clinical-trial-matcher)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![BioBERT](https://img.shields.io/badge/BioBERT-PubMed--BERT-green)
![Precision@3](https://img.shields.io/badge/Precision%40K-90%25-brightgreen)
![F1](https://img.shields.io/badge/F1%20Score-0.87-brightgreen)
![Deployed](https://img.shields.io/badge/Deployed-Hugging%20Face-yellow)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

</div>

---

## 🌐 Try It Live

**👉 [https://huggingface.co/spaces/yashikaSharma/clinical-trial-matcher](https://huggingface.co/spaces/yashikaSharma/clinical-trial-matcher)**

No setup needed — just open the link and start matching patients to clinical trials instantly!

---

## 📌 Problem

Matching patients to the right clinical trial is a manual, time-consuming process that requires:
- Reading hundreds of trial eligibility criteria
- Cross-referencing patient demographics and medical history
- Filtering by age, gender, condition, and comorbidities

This project **automates that pipeline end-to-end using NLP** — achieving 90% Precision@3 on real ClinicalTrials.gov data.

---

## 🧠 Architecture

```
Patient Clinical Note
        │
        ▼
 Condition Keyword Anchoring
 (CDC-grounded medical terms)
        │
        ▼
 BioBERT Embeddings          +       TF-IDF Keyword Overlap
 (Semantic similarity)               (Medical term matching)
        │                                      │
        └──────────── Hybrid Score ────────────┘
                      α·semantic + β·keyword
                      (0.7 × BioBERT + 0.3 × TF-IDF)
                             │
                             ▼
               Hard Eligibility Filters
               (Age range, Gender match)
                             │
                             ▼
                   Top-K Ranked Trials
```

**Why BioBERT over general-purpose models?**

BioBERT is pre-trained on PubMed abstracts and clinical text, making it significantly better at capturing medical semantics. MiniLM-L6 (general purpose) scored ~33% Precision@3 on this task, while **BioBERT hybrid reached 90% Precision@3** — a 57 point improvement.

---

## 📊 Results

| Metric | MiniLM (Baseline) | BioBERT Hybrid (Ours) |
|---|---|---|
| Precision@1 | 40% | **96.7%** ✅ |
| Precision@3 | 33% | **90.0%** ✅ |
| Precision@5 | 28% | **90.7%** ✅ |
| F1 Score | — | **0.87** ✅ |
| Age/Gender Filters | ❌ | ✅ |
| Keyword Overlap | ❌ | ✅ |
| Condition Anchoring | ❌ | ✅ |

> Evaluation on **30 patients × 50 real trials** with explicit positive/negative pair labeling — no leakage from source metadata.

![Results Dashboard](clinical_trial_matching_results.png)

---

## 🖥️ Live Demo — Features

The deployed Gradio app supports:

- 🔍 **Patient input** — age, gender, condition, symptoms
- ⚡ **Real-time matching** — BioBERT + TF-IDF hybrid scoring
- 📋 **Top-K results** — ranked trials with match scores
- 🧪 **Quick fill examples** — Diabetes, Cancer, Hypertension, Alzheimer's
- 🔧 **Adjustable filters** — number of results, trial status

---

## 🗂️ Dataset

**Patients (30 total)**
- Medically grounded demographics based on CDC condition profiles
- 5 conditions: Diabetes, Hypertension, Asthma, Cancer, Chronic Kidney Disease
- Fields: age, gender, condition, comorbidities, clinical note

**Clinical Trials (50 total)**
- Fetched live from [ClinicalTrials.gov API v2](https://clinicaltrials.gov/api/v2/studies)
- Fields: NCT ID, title, condition, eligibility criteria, age range, gender

---

## 🔧 Tech Stack

| Component | Technology |
|---|---|
| Domain Embeddings | `pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb` |
| Keyword Retrieval | `scikit-learn` TF-IDF |
| Trial Data | ClinicalTrials.gov REST API v2 |
| Orchestration | LangChain |
| Frontend / Demo | Gradio |
| Deployment | Hugging Face Spaces |
| Data Processing | Pandas, NumPy |
| Evaluation | Custom Precision@K, F1 with threshold sweep |

---

## 🔍 Sample Predictions

```
Patient P0001 | Condition: diabetes | Age: 52 | Female
  ✅ [0.683] Insulin Infusion and Infectious Diabetic Foot Ulcers (IIIFU)
  ✅ [0.680] Closing the Loop in Adults With Type 1 Diabetes Under Free Living Conditions
  ✅ [0.680] Safety Evaluation of Adverse Reactions in Diabetes

Patient P0002 | Condition: cancer | Age: 61 | Male
  ✅ [0.710] Non-Small Cell Lung Carcinoma Stage III Immunotherapy Trial
  ✅ [0.705] PD-L1 Positive Checkpoint Inhibitor Study
  ✅ [0.681] Platinum-Based Chemotherapy Alternatives Trial
```

---

## 📈 Key Engineering Decisions

**1. Domain-specific embeddings**
General-purpose models treat "HbA1c" and "glucose level" as unrelated. BioBERT, trained on PubMed, understands these are semantically close in a diabetes context.

**2. Condition keyword anchoring**
Patient clinical notes describe symptoms; trial titles describe interventions. Adding condition-specific medical keyword anchors bridges this vocabulary gap.

**3. Hybrid scoring**
`score = 0.7 × BioBERT_similarity + 0.3 × TF-IDF_overlap`
Semantic meaning dominates, but keyword overlap catches cases where specific medical terms would otherwise be missed.

**4. Hard eligibility filters**
Age range and gender are non-negotiable clinical constraints — applied before scoring to prevent false positives.

**5. Honest evaluation**
Precision@K is computed by ranking all 50 trials using only model scores — no condition metadata leakage.

---

## 🚀 How to Run Locally

### 1. Clone the repo
```bash
git clone https://github.com/yashikasharma2004/clinical-trial-matcher.git
cd clinical-trial-matcher
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Gradio app
```bash
python app.py
```

### 4. Or run the notebook
Open `clinical_trial_matcher.ipynb` in Kaggle or Jupyter and run all cells.

> ⚠️ Internet access required for ClinicalTrials.gov API calls.

---

## 📁 Project Structure

```
clinical-trial-matcher/
│
├── app.py                              # Gradio web app (deployed on HF Spaces)
├── clinical_trial_matcher.ipynb        # Main notebook — all cells A through M
├── clinical_trial_matching_results.png # Results dashboard (4 charts)
├── evaluation_results.csv              # Patient-trial pairs with labels + scores
├── patients_dataset.csv                # 30 generated patients
├── trials_dataset.csv                  # 50 real trials from ClinicalTrials.gov
├── model_summary.json                  # Final metrics summary
├── requirements.txt                    # Python dependencies
└── README.md
```

---

## 🧪 Evaluation Methodology

- **No metadata leakage** — Trial condition field never used during scoring
- **Explicit labeling** — Each patient-trial pair manually labeled relevant/non-relevant
- **Threshold sweep** — F1 computed across 0.1–0.9 similarity thresholds; optimal at 0.5
- **Hard filters first** — Age + gender constraints applied before semantic scoring

---

## 🔮 Future Work

- [ ] Expand to 500+ trials across 20+ conditions
- [ ] Parse full eligibility criteria text (inclusion/exclusion)
- [ ] Fine-tune BioBERT on clinical trial matching pairs
- [ ] Add FAISS index for scalable retrieval (10,000+ trials)
- [ ] Multi-condition patient support (comorbidities matching)
- [ ] Deploy as FastAPI microservice

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

## 🙋 Author

**Yashika Sharma**  
Built as a portfolio project demonstrating applied NLP in healthcare.  
Feel free to ⭐ star the repo or open issues and PRs!

[![Live Demo](https://img.shields.io/badge/🚀%20Try%20Live%20Demo-Click%20Here-orange?style=for-the-badge)](https://huggingface.co/spaces/yashikaSharma/clinical-trial-matcher)
