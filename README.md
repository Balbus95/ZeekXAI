# ZeekXAI - Analysis and Classification of Attacks in Critical Infrastructures

ZeekXAI is a Master's thesis project focused on detecting and classifying cyberattack tactics—mapped to the MITRE ATT&CK framework—within critical infrastructures using network traffic data (Zeek logs).

This system integrates Machine Learning algorithms (Random Forest, Decision Tree, Logistic Regression, Naive Bayes) with **Explainable AI (XAI)** techniques, such as SHAP (Shapley Additive Explanations) and PDP (Partial Dependence Plots). The goal is to ensure that the models' decision-making processes are transparent, interpretable, and understandable for cybersecurity analysts.

## 🚀 Key Features

*   **Zeek Data Analysis**: Efficient loading and parsing of network logs in Parquet format.
*   **Feature Engineering**: Automatic calculation of derived metrics (e.g., `pktAtsec`, `BitRate`, `interTime`) to enrich the dataset.
*   **Machine Learning**: Training and evaluation of models for:
    *   **Binary Classification**: Distinguishing between benign and malicious traffic.
    *   **Multiclass Classification**: Identifying specific MITRE tactics (e.g., *Reconnaissance*, *Discovery*, *Resource Development*).
*   **Explainable AI (XAI)**:
    *   **SHAP (Global & Local)**: Feature importance analysis and its impact on predictions (Beeswarm plots, Bar charts, Waterfall plots).
    *   **Partial Dependence Plots (PDP)**: Visualizing the marginal relationship between features and the model's output.
*   **Automated Reporting**: Generation of performance metrics (CSV/JSON) and visualizations (PDF).

## 🛠️ Installation

Make sure you have Python 3.8+ installed.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Balbus95/ZeekXAI-private.git
    cd ZeekXAI-private
    ```

2.  **Create a virtual environment (optional but highly recommended):**
    ```bash
    python -m venv .venv
    
    # On Windows:
    .venv\Scripts\activate
    
    # On Linux/macOS:
    source .venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## 📦 Project Structure

*   `zeekXAI.py`: The main script to execute the end-to-end pipeline (training, evaluation, and XAI generation).
*   `analyze_shap.py` / `analyze_shap_waterfall.py` / `analyze_pdp.py`: Dedicated auxiliary scripts for running and refining specific SHAP and PDP visualizations.
*   `list_features_values.py`: Utility script for exploring feature values.
*   `UWF-ZeekDataFall22/`: Directory containing the dataset in Parquet format.
*   `cache/`: Directory for caching trained models and processed datasets to speed up subsequent runs.
*   `output/`: Directory where all generated results (graphs, reports, execution logs) are saved.
*   `latex/` & `presentazione/`: LaTeX source files, thesis documentation, and presentation materials.
*   `* .md, .txt, .pdf`: Various notes, guidelines, and reference papers (e.g., `paper_dataset_ZeekDataFall22.pdf`) supporting the thesis research.

## 💻 Usage

### End-to-End Execution (Training + Validation + XAI)
To start the complete analysis pipeline, run:

```bash
python zeekXAI.py
```

This script will automatically perform the following steps:
1.  Load, clean, and preprocess the dataset.
2.  Train the ML models (or load them from the `cache/` if they were already trained).
3.  Evaluate the models' performance and compute metrics.
4.  Generate SHAP and PDP plots, saving them in a dedicated timestamped folder inside `output/` (e.g., `output/test_<timestamp>`).

## 📊 Generated Outputs

All analysis results and reports are saved within the corresponding `output/test_<timestamp>/` directory:
*   **Metrics**: `B_01_MetricsTable_Comparative_AllMulticlass.csv`
*   **Visualizations**:
    *   `A_01_ClassDistribution...pdf`: Target class distribution representation.
    *   `B_02_ConfusionMatrix...pdf`: Confusion matrices outlining classification accuracy.
    *   `C_01_SHAP_Beeswarm...pdf`: Beeswarm plots explaining model interpretability.
    *   `C_03_PDP...pdf`: Partial dependence plots mapping feature marginal effects.
