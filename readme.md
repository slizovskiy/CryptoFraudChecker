# Crypto-Check: LLM-Based Fraud Detection for Whitepapers

**Live Application:** [crypto-check.streamlit.app](https://crypto-check.streamlit.app/)  
**Research and Development:** Sergey Slizovskiy

## 🚀 Overview
Crypto-Check is an automated fraud detection pipeline designed to bridge the gap between pre-launch investment and regulatory oversight. While traditional crypto-forensics focus on on-chain anomalies (post-hoc), this tool performs **Narrative Evidence Analysis** to detect fraudulent intent and regulatory violations within project whitepapers *before* capital is deployed.

The system acts as a "Policy-Constrained First Reader," mapping unstructured text to formal legal criteria from the **SEC (USA)**, **FCA (UK)**, and **HKSFC (Hong Kong)**.

---

## 🛠 Methodology: Criteria Selection & Model Training

The application uses a hybrid approach combining LLM-based extraction with statistical machine learning.

### 1. Criteria Generation (Model Context Protocol)
To ensure legal accuracy and eliminate hallucination:
* **Protocol:** Regulatory guidelines were accessed via a dedicated **Model Context Protocol (MCP)** server.
* **Consistency:** Each regulation set was generated **5 times** independently. A secondary summarization prompt consolidated these into a final set, ensuring all legal references were verbatim and verified.
* **Scope:** * **FCA:** 13 criteria (Focus on AML and promotions).
    * **HKSFC:** 17 criteria (Focus on virtual asset service provision).
    * **SEC:** 46 criteria (Broadest scope, including Investment Company Act logic).

### 2. Feature Selection Pipeline
From an initial pool of over 70 criteria, we developed an "Optimal Set" through:
* **Correlation-Based Removal:** Mitigating multicollinearity by removing features with a correlation threshold > 0.8.
* **Recursive Feature Elimination (RFE):** Using Logistic Regression to select the top 6 predictive features.
* **Negative Coefficient Filtering:** Ensuring every selected criterion is a positive indicator of fraud to maintain model transparency and logic.

---

## ⚖️ Scoring Logic and Weights

The app calculates a fraud probability based on the following feature encoding:

For any given criterion $j$, the effective feature value $x_j$ is:

$$x_j = W \cdot R + (1 - W) \cdot 0.5$$

Where:
* **$R$ (Response):** 1 if the LLM detects a violation ("Yes"), 0 if not ("No").
* **$W$ (Weight):** Confidence weight based on evidence.
    * **1.0:** Abundant evidence (Highest confidence).
    * **0.5:** Some evidence.
    * **0.0:** Insufficient evidence (Neutral score of 0.5).

### Score Mapping Table
| Evidence Strength | LLM Result | Numerical Score ($x_j$) |
| :--- | :--- | :--- |
| **Abundant Evidence** | Yes | **1.00** |
| **Some Evidence** | Yes | **0.75** |
| **Insufficient Evidence** | Undecided | **0.50** |
| **Some Evidence** | No | **0.25** |
| **No Evidence** | No | **0.00** |

The final "Fraud" vs "Non-Fraud" decision is made via a Logistic Regression model trained on a curated dataset of 48 whitepapers (23 known frauds, 25 legitimate), achieving a **Cross-Validation F1-score of 0.93**.

---

## 🔍 Verifiability & Interpretability

Axiomatic AI emphasizes "Verifiable AI." This app implements this through three core features:

1.  **Evidence Provenance:** For every criterion marked "Yes," the app provides a **verbatim quote** from the whitepaper as proof. The LLM is instructed to locate and retain sentences verbatim to prevent paraphrasing hallucinations.
2.  **Relevance Scores (SHAP):** The app displays how much each criterion contributed to the final fraud score using SHAP (SHapley Additive exPlanations) values.
3.  **Human-in-the-loop (HITL):** Designed to assist human investigators (like *CoffeeZilla*, used as a benchmark in the paper), the app identifies narrative-driven "red flags" that humans often miss, such as undisclosed centralization contradictions.



---

## 💻 Technical Stack
* **LLM Engine:** Google Gemini 2.0 / 2.5 (Utilizing 1M+ token context window for large regulatory corpora).
* **Backend:** Python, Scikit-Learn (Logistic Regression, RFE).
* **Frontend:** Streamlit.
* **Deployment:** Containerized via **Docker** to ensure reproducible environments across research and production.

---

## 📖 Research Context
This tool is the technical implementation of the paper: *"Role of LLM in detecting investment scams,"* which evaluates the alignment between LLM automated reasoning and human expert investigative transcripts.