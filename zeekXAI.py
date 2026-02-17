from __future__ import annotations

import json
import pickle
import textwrap
import shutil
import warnings
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, cast

import joblib
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import lime
import lime.lime_tabular
from sklearn.tree import plot_tree
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from pyspark.sql import SparkSession

try:
    from pyspark.errors.exceptions.base import AnalysisException
except ImportError:
    from pyspark.sql.utils import AnalysisException
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import PartialDependenceDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

try:
    from shap import Explanation as ShapExplanation
except (ImportError, AttributeError):
    ShapExplanation = tuple()

# --- SETUP LOGGING & WARNINGS ---
warnings.filterwarnings("ignore")

run_timestamp = datetime.now().strftime("%d-%m-%Y_%H.%M")
report_buffer = []

def log(message: str = ""):
    """Prints to console and appends to report buffer."""
    print(message)
    report_buffer.append(message)

plt.style.use("seaborn-v0_8")
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

FEATURE_DESCRIPTIONS = {
    # Feature Native
    "orig_pkts": "Origin Packets (Count)",
    "resp_pkts": "Response Packets (Count)",
    "orig_bytes": "Origin Bytes (Volume)",
    "resp_bytes": "Response Bytes (Volume)",
    "duration": "Duration (Seconds)",
    
    # Feature Derivate
    "pktAtsec": "Packet Density (pkts/sec)",
    "BitRate": "BitRate (bits/sec)",
    "interTime": "Inter-Arrival Time (sec)",
    "avgLenPkt": "Avg Packet Size (bytes)",
    "numPKT": "Total Packets (Orig+Resp)",
    "numBytes": "Total Bytes (Orig+Resp)",
    
    # Prefissi Categorici (One-Hot)
    "proto_": "Protocol: ",
    "service_": "Service: ",
    "conn_state_": "State: ",
    "history_": "History: "
}

def _prettify_feature_name(feature_name: str) -> str:
    if feature_name in FEATURE_DESCRIPTIONS:
        return FEATURE_DESCRIPTIONS[feature_name]
    
    for prefix, pretty_prefix in FEATURE_DESCRIPTIONS.items():
        if prefix.endswith("_") and feature_name.startswith(prefix):
            raw_value = feature_name[len(prefix):]
            
            if prefix in ["proto_", "service_"]:
                raw_value = raw_value.upper()
                if raw_value == "NONE":
                    raw_value = "No Service"
                
            return f"{pretty_prefix}{raw_value}"

    if len(feature_name) > 32:
        return f"{feature_name[:29]}..."
    
    return feature_name

def log_feature_legend():
    log("\n" + "="*40)
    log(" LEGENDA FEATURE (Technical -> Readable)")
    log("="*40)
    log(f"{'NOME TECNICO':<20} | {'DESCRIZIONE ESTESA'}")
    log("-" * 60)
    for tech, desc in FEATURE_DESCRIPTIONS.items():
        if not tech.endswith("_"):
            log(f"{tech:<20} | {desc}")
    log("-" * 60)

# Configurazione Path: Dataset, Cache e Output
BASE_PATH = Path.cwd()
DATA_DIR = BASE_PATH / "UWF-ZeekDataFall22" / "parquet"

RESULTS_DIR = BASE_PATH / "cache"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TEST_DIR = BASE_PATH / "output" / f"test_{run_timestamp}"
TEST_DIR.mkdir(parents=True, exist_ok=True)

PLOTS_DIR = TEST_DIR 

OVERWRITE_ARTIFACTS = False

log("="*60)
log(f"   ZEEK XAI ANALYSIS REPORT - {run_timestamp}")
log("="*60)
log(f"Working directory:         {BASE_PATH}")
log(f"Dataset directory:         {DATA_DIR.relative_to(BASE_PATH)}")
log(f"Cache directory (Models):  {RESULTS_DIR.relative_to(BASE_PATH)}")
log(f"Output directory:          {TEST_DIR.relative_to(BASE_PATH)}")
log(f"Overwrite artifacts:       {OVERWRITE_ARTIFACTS}")
log("-" * 60)

try:
    spark = SparkSession.builder \
        .appName("UWFZeekXAI") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .config("spark.sql.shuffle.partitions", "200") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    log(f"Spark session avviata: {spark.version}")
except Exception as exc:
    raise RuntimeError("Impossibile inizializzare Spark. Verifica l'installazione di pyspark.") from exc

parquet_files = []
for subdir in sorted(DATA_DIR.glob("202*")):
    parquet_files.extend(sorted(subdir.glob("*.snappy.parquet")))

parquet_files = [str(p) for p in parquet_files if "_parquet_" not in p.name]

if not parquet_files:
    raise FileNotFoundError("Nessun file Parquet trovato. Verifica la cartella del dataset.")

log(f"\n[DATASET INFO]")
log(f"File Parquet individuati: {len(parquet_files)}")
log("Esempio primi 3 file:")
for sample_path in parquet_files[:3]:
    log(f"  - {Path(sample_path).name}")

try:
    df_spark = spark.read.parquet(*parquet_files)
except AnalysisException as exc:
    raise RuntimeError("Errore nella lettura dei Parquet. Controllare le colonne disponibili.") from exc

selected_columns = [
    "orig_pkts", "resp_pkts", "orig_bytes", "resp_bytes", "duration",
    "proto", "service", "conn_state", "history",
    "label_binary", "label_tactic", "label_technique"
]
existing_columns = [col for col in selected_columns if col in df_spark.columns]
missing_columns = sorted(set(selected_columns) - set(existing_columns))

if missing_columns:
    raise ValueError(f"Colonne mancanti nel dataset Spark: {missing_columns}")

filtered_spark = df_spark.select(*existing_columns)
filtered_spark.cache()

label_balance = (filtered_spark
                 .groupBy("label_binary")
                 .count()
                 .toPandas()
                 .sort_values("label_binary"))
log(str(label_balance))

df_pandas = filtered_spark.toPandas()
log("\n[DATA QUALITY CHECK]")
log(f"Shape dataset Pandas: {df_pandas.shape}")
log("\nValori nulli per colonna:")
log(str(df_pandas.isna().sum()))

log("\nDistribuzione Tattiche (prima del cleaning):")
log(str(df_pandas["label_tactic"].value_counts(dropna=False)))

# Utility per produrre nomi file sicuri e salvare ogni figura come PDF
def _slugify_filename(name: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name)
    cleaned = cleaned.strip("_")
    return cleaned or "plot"

def _save_figure(fig, filename):
    """Salva una figura in PDF e PNG."""
    try:
        pdf_path = TEST_DIR / f"{filename}.pdf"
        fig.savefig(pdf_path, bbox_inches="tight", dpi=300)
        
        # png_path = TEST_DIR / f"{filename}.png"
        # fig.savefig(png_path, bbox_inches="tight", dpi=150)
        
        try:
            rel_path = pdf_path.relative_to(BASE_PATH)
        except ValueError:
            rel_path = pdf_path
            
        log(f"Grafico salvato: {rel_path}")
        plt.close(fig)
    except Exception as e:
        log(f"Errore salvataggio grafico {filename}: {e}")

def _add_zeek_legend(fig, fontsize=10, x=0.87, y=0.09, ha='right', va='bottom'):
    """Aggiunge una legenda esplicativa per i codici Zeek (State/History)."""
    legend_text = (
        "== Legenda Codici Zeek ==\n"
        "\nStato Connessione (State):\n"
        "  S0: Tentativo visto, nessuna risposta\n"
        "  S1: Stabilita, non terminata\n"
        "  SF: Stabilita e terminata normalmente\n"
        "  REJ: Tentativo rifiutato (RST)\n"
        "  S2/S3: Stabilita, tentativo chiusura visto\n"
        "  RSTO/RSTR: Reset da origine/risponditore\n"
        "  RSTRH: Reset durante handshake\n"
        "  SH/SHR: SYN senza ACK (orig/risp)\n"
        "  OTH: Nessun SYN visto (traffico parziale)\n"
        "\nStoria Pacchetti (History):\n"
        "  S/s: SYN (Maiusc=Origine, Min=Risp.)\n"
        "  H/h: SYN+ACK\n"
        "  A/a: ACK (Conferma)\n"
        "  D/d: Dati (Payload)\n"
        "  F/f: FIN (Terminazione)\n"
        "  R/r: RST (Reset)\n"
        "  ^: Inversione direzione"
    )
    fig.text(x, y, legend_text, fontsize=fontsize, ha=ha, va=va, 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))

# Divisione robusta che evita NaN/infiniti dovuti a denominatori nulli
def safe_div(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator_clean = denominator.replace({0: np.nan}).fillna(1.0).astype(float)
    result = numerator.astype(float) / denominator_clean
    return result.replace([np.inf, -np.inf], 0.0).fillna(0.0)

numerical_cols = ["orig_pkts", "resp_pkts", "orig_bytes", "resp_bytes", "duration"]
for col in numerical_cols:
    if col not in df_pandas.columns:
        raise KeyError(f"Colonna numerica {col} assente nel dataset.")

# Creazione Feature Derivate (Feature Engineering)
df_pandas["pktAtsec"] = safe_div(df_pandas["orig_pkts"], df_pandas["duration"])
df_pandas["BitRate"] = safe_div((df_pandas["orig_bytes"] + df_pandas["resp_bytes"]) * 8, df_pandas["duration"])
df_pandas["interTime"] = safe_div(df_pandas["duration"], df_pandas["orig_pkts"].replace({0: 1}))
df_pandas["avgLenPkt"] = safe_div(df_pandas["orig_bytes"], df_pandas["orig_pkts"].replace({0: 1}))
df_pandas["numPKT"] = df_pandas["orig_pkts"] + df_pandas["resp_pkts"]
df_pandas["numBytes"] = df_pandas["orig_bytes"] + df_pandas["resp_bytes"]

engineered_columns = ["pktAtsec", "BitRate", "interTime", "avgLenPkt", "numPKT", "numBytes"]
log(f"Feature derivate create: {engineered_columns}")

log_feature_legend()

mandatory_labels = ["label_binary", "label_tactic", "label_technique"]
missing_label_rows = df_pandas[mandatory_labels].isna().any(axis=1)
if missing_label_rows.any():
    log(f"Righe con etichette mancanti: {missing_label_rows.sum()}")
    df_pandas.loc[missing_label_rows, ["label_tactic", "label_technique"]] = df_pandas.loc[missing_label_rows, ["label_tactic", "label_technique"]].fillna("Unknown")

# Normalizzazione Etichette
df_pandas["label_tactic"] = df_pandas["label_tactic"].replace({"none": "Benign"})

categorical_map = {"True": 1, "False": 0, "Duplicate": 1}
df_pandas["label_binary"] = df_pandas["label_binary"].replace(categorical_map)

log("\n" + "="*40)
log(" A. EXPLORATORY DATA ANALYSIS (EDA)")
log("="*40)
log("[EDA] Generazione grafico distribuzione classi...")
plt.figure(figsize=(12, 6))
order = df_pandas["label_tactic"].value_counts().index
ax = sns.countplot(y="label_tactic", data=df_pandas, order=order, palette="viridis")
ax.set_xscale("log")
ax.bar_label(ax.containers[0], padding=3)
plt.title("Distribuzione delle Tattiche", fontsize=15)
plt.xlabel("Conteggio (Scala Logaritmica)", fontsize=12)
plt.ylabel("Tattica MITRE ATT&CK", fontsize=12)
plt.tight_layout()
_save_figure(plt.gcf(), "A_01_ClassDistribution_Single_Dataset")
log(f"[EDA] Grafico salvato: A_01_ClassDistribution_Single_Dataset")

# DATA EXPORT A.01
extended_analysis_data = {} # Initialize here if not already
extended_analysis_data["A_01_ClassDistribution"] = df_pandas["label_tactic"].value_counts().to_dict()

initial_rows = len(df_pandas)
df_pandas = df_pandas.dropna(subset=["label_binary"])
log(f"Righe eliminate per label binaria mancante: {initial_rows - len(df_pandas)}")

continuous_features = [
    "orig_pkts", "resp_pkts", "orig_bytes", "resp_bytes", "duration",
    "pktAtsec", "BitRate", "interTime", "avgLenPkt", "numPKT", "numBytes"
]
categorical_features = ["proto", "service", "conn_state", "history"]
feature_columns = continuous_features + categorical_features

missing_features = sorted(set(feature_columns) - set(df_pandas.columns))
if missing_features:
    raise ValueError(f"Feature mancanti nel dataset: {missing_features}")

X = df_pandas[feature_columns].copy()
y_binary = df_pandas["label_binary"].astype(int)

tactic_encoder = LabelEncoder()
y_tactic = tactic_encoder.fit_transform(df_pandas["label_tactic"].astype(str))

log(f"Numero classi tattiche: {len(tactic_encoder.classes_)}")

X_train, X_test, y_binary_train, y_binary_test, y_tactic_train, y_tactic_test = train_test_split(
    X,
    y_binary,
    y_tactic,
    test_size=0.2,
    stratify=y_binary,
    random_state=RANDOM_SEED,
)

log(f"Distribuzione label binaria (train): {Counter(y_binary_train)}")
log(f"Distribuzione label binaria (test): {Counter(y_binary_test)}")

# Pipeline di preprocessing condivisa fra tutti i modelli
preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]),
            continuous_features
        ),
        (
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            ]),
            categorical_features
        ),
    ],
    remainder="drop",
    verbose_feature_names_out=False
)

# --- CACHING DATASET PROCESSATO ---
processed_data_path = RESULTS_DIR / "dataset_processed.npz"
feature_names_path = RESULTS_DIR / "dataset_feature_names.pkl"

if not OVERWRITE_ARTIFACTS and processed_data_path.exists() and feature_names_path.exists():
    log("Caricamento dati processati e nomi feature dalla cache...")
    data = np.load(processed_data_path)
    X_train_processed = data["X_train"]
    X_test_processed = data["X_test"]
    y_binary_train = data["y_binary_train"]
    y_binary_test = data["y_binary_test"]
    y_tactic_train = data["y_tactic_train"]
    y_tactic_test = data["y_tactic_test"]
    
    with open(feature_names_path, "rb") as f:
        all_feature_names = pickle.load(f)

    log("Refitting preprocessor per inverse_transform...")
    preprocessor.fit(X_train)

else:
    log("Preprocessing e salvataggio cache...")
    X_train_processed = np.asarray(preprocessor.fit_transform(X_train))
    X_test_processed = np.asarray(preprocessor.transform(X_test))
    
    np.savez_compressed(
        processed_data_path,
        X_train=X_train_processed,
        X_test=X_test_processed,
        y_binary_train=y_binary_train,
        y_binary_test=y_binary_test,
        y_tactic_train=y_tactic_train,
        y_tactic_test=y_tactic_test
    )
    
    processed_cat_features = list(preprocessor.named_transformers_["cat"].named_steps["ohe"].get_feature_names_out(categorical_features))
    all_feature_names = continuous_features + processed_cat_features
    
    with open(feature_names_path, "wb") as feat_file:
        pickle.dump(all_feature_names, feat_file)

log(f"Shape train processed: {X_train_processed.shape}")
log(f"Shape test processed: {X_test_processed.shape}")

plot_feature_names = [_prettify_feature_name(name) for name in all_feature_names]

# --- Modelli Binari ---
binary_models = {
    "Random Forest": RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=RANDOM_SEED, n_jobs=-1),
    "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=RANDOM_SEED),
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear", random_state=RANDOM_SEED),
    "Naive Bayes": GaussianNB()
}

log("\n" + "="*40)
log(" B. MODEL EVALUATION")
log("="*40)

log("\n--- 1. Valutazione Modelli Binari ---")
binary_reports = {}
for name, model in binary_models.items():
    model_path = RESULTS_DIR / f"model_binary_{name.lower().replace(' ', '_')}.joblib"
    
    if not OVERWRITE_ARTIFACTS and model_path.exists():
        log(f"{name} (Binary): caricamento dalla cache.")
        model = joblib.load(model_path)
    else:
        log(f"{name} (Binary): addestramento...")
        model.fit(X_train_processed, y_binary_train)
        joblib.dump(model, model_path)

    y_pred = model.predict(X_test_processed)
    log(f"\n{name} (Binary):")
    log(classification_report(y_binary_test, y_pred, zero_division=0))
    binary_reports[name] = classification_report(y_binary_test, y_pred, output_dict=True, zero_division=0)
    
    binary_models[name] = model

# --- Modelli Multiclasse ---
multi_models = {
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=35, class_weight="balanced", random_state=RANDOM_SEED, n_jobs=-1),
    "Decision Tree": DecisionTreeClassifier(max_depth=15, class_weight="balanced", random_state=RANDOM_SEED),
    "Naive Bayes": GaussianNB(),
    "Logistic Regression": LogisticRegression(C=10.0, max_iter=3000, solver="saga", class_weight="balanced", n_jobs=-1, random_state=RANDOM_SEED)
}

log("\n--- 2. Valutazione Modelli Multiclasse ---")
metrics_data = []
tactic_predictions = {}
trained_multi_models = {}

fig_cm, axes_cm = plt.subplots(2, 2, figsize=(16, 12))
axes_cm = axes_cm.flatten()

for idx, (name, model) in enumerate(multi_models.items()):
    model_path = RESULTS_DIR / f"model_multiclass_{name.lower().replace(' ', '_')}.joblib"
    preds_path = RESULTS_DIR / f"predictions_multiclass_{name.lower().replace(' ', '_')}.npy"

    model_exists = model_path.exists()
    preds_exist = preds_path.exists()
    skip_training = (not OVERWRITE_ARTIFACTS and model_exists and preds_exist)

    if skip_training:
        log(f"{name}: artefatti esistenti, caricamento dal disco.")
        model = joblib.load(model_path)
        y_pred = np.load(preds_path)
    else:
        log(f"{name}: addestramento in corso...")
        model.fit(X_train_processed, y_tactic_train)
        y_pred = model.predict(X_test_processed)
        
        if OVERWRITE_ARTIFACTS or not model_exists:
            joblib.dump(model, model_path)
        
        if OVERWRITE_ARTIFACTS or not preds_exist:
            np.save(preds_path, y_pred)

    trained_multi_models[name] = model
    tactic_predictions[name] = y_pred
    
    acc = accuracy_score(y_tactic_test, y_pred)
    prec = precision_score(y_tactic_test, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_tactic_test, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_tactic_test, y_pred, average="macro", zero_division=0)
    
    metrics_data.append({
        "Model": name,
        "Accuracy": acc,
        "Precision (Macro)": prec,
        "Recall (Macro)": rec,
        "F1-Score (Macro)": f1
    })
    
    # Fix: Explicitly pass all labels to ensure matrix size is consistent (N_classes x N_classes)
    # even if some classes are missing in the test set predictions.
    all_labels_indices = range(len(tactic_encoder.classes_))
    cm = confusion_matrix(y_tactic_test, y_pred, labels=all_labels_indices, normalize='true')
    
    # Plot con annotazioni (valori numerici) e rotazione label
    sns.heatmap(cm, annot=False, fmt=".2f", cmap="Blues", ax=axes_cm[idx], xticklabels=tactic_encoder.classes_, yticklabels=tactic_encoder.classes_)
    
    axes_cm[idx].set_title(f"Matrice di Confusione (Recall) - {name}", fontsize=11)
    axes_cm[idx].set_xlabel("Classe Predetta", fontsize=10)
    axes_cm[idx].set_ylabel("Classe Reale", fontsize=10)
    axes_cm[idx].tick_params(axis='x', rotation=90, labelsize=9)
    axes_cm[idx].tick_params(axis='y', rotation=0, labelsize=9)

plt.tight_layout()
_save_figure(fig_cm, "B_02_ConfusionMatrix_Individual_AllMulticlass")

# DATA EXPORT B.02
extended_analysis_data.setdefault("B_02_ConfusionMatrices", {})
for name, model in trained_multi_models.items():
    if name in tactic_predictions:
        y_p = tactic_predictions[name]
        # Calculate CM again to be sure or use stored if available. Re-calcluating for clarity.
        all_labels_indices = range(len(tactic_encoder.classes_))
        cm_norm = confusion_matrix(y_tactic_test, y_p, labels=all_labels_indices, normalize='true')
        
        cm_dict = {}
        for i, true_class in enumerate(tactic_encoder.classes_):
            row_dict = {}
            for j, pred_class in enumerate(tactic_encoder.classes_):
                row_dict[pred_class] = float(cm_norm[i, j])
            cm_dict[true_class] = row_dict
            
        extended_analysis_data["B_02_ConfusionMatrices"][name] = cm_dict

# Tabella Riassuntiva
metrics_df = pd.DataFrame(metrics_data)
log("\n=== Tabella Riassuntiva Performance (Multiclasse) ===")
log("La seguente tabella mostra le metriche principali per tutti i modelli multiclasse:")
log(str(metrics_df))
metrics_df.to_csv(TEST_DIR / "B_01_MetricsTable_Comparative_AllMulticlass.csv", index=False)

# =============================================================================
# 5. GLOBAL EXPLAINABILITY (SHAP & PDP)
# =============================================================================

log("\n" + "="*40)
log(" C. GLOBAL EXPLAINABILITY (SHAP & PDP)")
log("="*40)
shap_sample_size = min(100, X_test_processed.shape[0]) 
rng = np.random.default_rng(RANDOM_SEED)
shap_indices = rng.choice(X_test_processed.shape[0], shap_sample_size, replace=False)
X_test_sample = X_test_processed[shap_indices]
X_train_summary = shap.kmeans(X_train_processed, 10) 

shap_values_dict = {}

for name, model in trained_multi_models.items():
    log(f"Calcolo SHAP per {name}...")
    try:
        if name in ["Random Forest", "Decision Tree"]:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test_sample)
        else:
            explainer = shap.KernelExplainer(model.predict_proba, X_train_summary)
            shap_values = explainer.shap_values(X_test_sample)

        if isinstance(shap_values, list):
            shap_values_combined = np.sum([np.abs(sv) for sv in shap_values], axis=0)
        else:
            shap_values_combined = np.abs(shap_values)
            if len(shap_values_combined.shape) == 3: # (samples, features, classes)
                 shap_values_combined = np.sum(shap_values_combined, axis=2)

        shap_values_dict[name] = np.mean(shap_values_combined, axis=0)

        INTERESTING_CLASSES = ["Benign", "Resource Development", "Reconnaissance", "Discovery"]
        
        # A. Beeswarm Plots per Classi Specifiche
        
        for cls_name in INTERESTING_CLASSES:
            if cls_name in tactic_encoder.classes_:
                cls_idx = list(tactic_encoder.classes_).index(cls_name)
                sv_class = None
                
                if isinstance(shap_values, list):
                    if cls_idx < len(shap_values):
                        sv_class = shap_values[cls_idx]
                elif hasattr(shap_values, "shape") and len(shap_values.shape) == 3:
                    if cls_idx < shap_values.shape[2]:
                        sv_class = shap_values[:, :, cls_idx]
                
            if sv_class is not None:
                    plt.figure()
                    # Impostiamo max_display=10 per mostrare le 10 feature più importanti (somma valori SHAP assoluti)
                    shap.summary_plot(sv_class, X_test_sample, feature_names=plot_feature_names, max_display=6, show=False)
                    fig = plt.gcf()
                    fig.set_size_inches(10, 5)
                    plt.title(f"Impatto SHAP (Beeswarm) - {name}: {cls_name}", fontsize=18)
                    plt.xlabel("Valore SHAP (impatto sull'output del modello)", fontsize=14)
                    plt.yticks(fontsize=12)
                    plt.xticks(fontsize=12)
                    # _add_zeek_legend(fig,fontsize=8)
                    plt.tight_layout()
                    _save_figure(fig, f"C_01_SHAP_Beeswarm_{_slugify_filename(name)}_{_slugify_filename(cls_name)}")
        
        # B. Global Bar Plot (Feature Importance Overall)
        plt.figure()
        shap.summary_plot(shap_values, X_test_sample, feature_names=plot_feature_names, class_names=tactic_encoder.classes_, plot_type="bar", max_display=10, show=False)
        fig = plt.gcf()
        fig.set_size_inches(10, 8)
        plt.xlabel("Impatto Medio sull'Output del Modello (Media |Valore SHAP|)", fontsize=14) # Custom label
        plt.title(f"Importanza Globale delle Feature - {name}", fontsize=18)
        plt.yticks(fontsize=12)
        plt.xticks(fontsize=12)
        plt.legend(fontsize=12, title_fontsize=12)
        # _add_zeek_legend(fig)
        plt.tight_layout()
        _save_figure(fig, f"C_01_SHAP_GlobalBar_{_slugify_filename(name)}")

        model_shap_data = {
            "Model": name,
            "Global_Feature_Importance": {
                feat: float(val) for feat, val in zip(plot_feature_names, shap_values_dict[name])
            },
            "Class_Specific_Analysis": {}
        }

        for cls_name in INTERESTING_CLASSES:
            if cls_name in tactic_encoder.classes_:
                cls_idx = list(tactic_encoder.classes_).index(cls_name)
                sv_class = None
                
                if isinstance(shap_values, list):
                     if cls_idx < len(shap_values):
                         sv_class = shap_values[cls_idx]
                elif hasattr(shap_values, "shape") and len(shap_values.shape) == 3:
                     if cls_idx < shap_values.shape[2]:
                         sv_class = shap_values[:, :, cls_idx]
                
                if sv_class is not None:
                     class_analysis = {}
                     mean_abs_class = np.mean(np.abs(sv_class), axis=0)

                     for i, feat_name in enumerate(plot_feature_names):
                         feature_values = X_test_sample[:, i]
                         shap_vals_feature = sv_class[:, i]
                         
                         if np.std(feature_values) < 1e-9 or np.std(shap_vals_feature) < 1e-9:
                             corr = 0.0
                         else:
                             corr = np.corrcoef(feature_values.astype(float), shap_vals_feature.astype(float))[0, 1]
                         
                         direction = "Mixed/Unclear"
                         if corr > 0.1:
                             direction = "High_Value_Increases_Prob"
                         elif corr < -0.1:
                             direction = "High_Value_Decreases_Prob"
                         
                         class_analysis[feat_name] = {
                             "Importance": float(mean_abs_class[i]),
                             "Impact_Correlation": float(corr),
                             "Insight": direction
                         }

                     model_shap_data["Class_Specific_Analysis"][cls_name] = class_analysis

        json_path = TEST_DIR / f"shap_summary_{_slugify_filename(name)}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(model_shap_data, f, indent=4)
        log(f"SHAP JSON salvato: {json_path.name}")

    except Exception as e:
        log(f"Errore SHAP per {name}: {e}")

log("Generazione grafico comparativo Feature Importance...")
fi_data = []
for name, mean_shap in shap_values_dict.items():
    if np.max(mean_shap) > 0:
        norm_shap = mean_shap / np.max(mean_shap)
    else:
        norm_shap = mean_shap
    
    for i, feat in enumerate(plot_feature_names):
        fi_data.append({
            "Model": name,
            "Feature": feat,
            "Importance": norm_shap[i]
        })

fi_df = pd.DataFrame(fi_data)
# Top 10 feature medie
top_features = fi_df.groupby("Feature")["Importance"].mean().sort_values(ascending=False).head(10).index
fi_df_filtered = fi_df[fi_df["Feature"].isin(top_features)]

plt.figure(figsize=(12, 8))
sns.barplot(data=fi_df_filtered, y="Feature", x="Importance", hue="Model", palette="viridis")
plt.title("Importanza Comparativa Feature (SHAP Normalizzato)", fontsize=15)
# _add_zeek_legend(plt.gcf())
plt.tight_layout()
_save_figure(plt.gcf(), "C_02_FeatureImportance_Comparative_AllModels")

extended_analysis_data["C_02_FeatureImportance"] = fi_data

log("Generazione PDP per Random Forest (Binario)...")
pdp_binary_data = []
try:
    rf_binary_model = binary_models["Random Forest"]
    importances_bin = rf_binary_model.feature_importances_
    indices_bin = np.argsort(importances_bin)[::-1]
    top_continuous_bin = [i for i in indices_bin if all_feature_names[i] in continuous_features][:3]
    
    if top_continuous_bin:
        from sklearn.inspection import partial_dependence
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        scaler = preprocessor.named_transformers_["num"].named_steps["scaler"]
        
        for i, feat_idx in enumerate(top_continuous_bin):
            feat_name = all_feature_names[feat_idx]
            readable_name = plot_feature_names[feat_idx]
            
            pdp_results = partial_dependence(
                rf_binary_model, X_test_processed, [feat_idx], kind="average", grid_resolution=50
            )
            
            if "values" in pdp_results:
                grid_values_norm = pdp_results["values"][0]
            elif "grid_values" in pdp_results:
                grid_values_norm = pdp_results["grid_values"][0]
            else:
                raise ValueError(f"Keys not found in PDP results: {pdp_results.keys()}")
            
            try:
                orig_feat_idx = continuous_features.index(feat_name)
                
                dummy_matrix = np.zeros((len(grid_values_norm), len(continuous_features)))
                dummy_matrix[:, orig_feat_idx] = grid_values_norm
                
                raw_values = scaler.inverse_transform(dummy_matrix)[:, orig_feat_idx]
                x_axis_values = raw_values
                x_label_suffix = " (Raw)"
            except ValueError as e:
                log(f"Errore inverse_transform PDP per {feat_name}: {e}")
                x_axis_values = grid_values_norm
                x_label_suffix = " (Norm)"

            y_pdp = pdp_results["average"][0] if len(pdp_results["average"].shape) > 1 else pdp_results["average"]
            
            axes[i].plot(x_axis_values, y_pdp)
            axes[i].set_xlabel(readable_name + x_label_suffix)
            axes[i].set_ylabel("Dipendenza Parziale")
            axes[i].set_title(f"PDP - {readable_name}")
            axes[i].grid(True, alpha=0.3)
            
            pdp_binary_data.append({
                "feature": readable_name,
                "x_values": x_axis_values.tolist() if isinstance(x_axis_values, np.ndarray) else list(x_axis_values),
                "y_values": y_pdp.tolist() if isinstance(y_pdp, np.ndarray) else list(y_pdp)
            })

        plt.suptitle("PDP - Random Forest (Binario - Top 3 Feature)", y=1.05, fontsize=16)
        plt.tight_layout()
        _save_figure(plt.gcf(), "C_03_PDP_Individual_RF_Binary")
        extended_analysis_data["C_03_PDP_Binary"] = pdp_binary_data
except Exception as e:
    log(f"Errore PDP Binario: {e}")
    import traceback
    log(traceback.format_exc())

# PDP per Random Forest (Multiclasse - Top 3 Feature)
log("Generazione PDP per Random Forest...")
pdp_multiclass_data = []
try:
    rf_model = trained_multi_models["Random Forest"]
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    top_continuous = [i for i in indices if all_feature_names[i] in continuous_features][:3]
    
    target_class_name = "Resource Development"
    if target_class_name in tactic_encoder.classes_:
        target_idx = list(tactic_encoder.classes_).index(target_class_name)
    else:
        target_idx = 0
        target_class_name = tactic_encoder.classes_[0]

    if top_continuous:
        from sklearn.inspection import partial_dependence
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        scaler = preprocessor.named_transformers_["num"].named_steps["scaler"]

        for i, feat_idx in enumerate(top_continuous):
            feat_name = all_feature_names[feat_idx]
            readable_name = plot_feature_names[feat_idx]
            
            pdp_results = partial_dependence(
                rf_model, X_test_processed, [feat_idx], kind="average", grid_resolution=50
            )
            
            if "values" in pdp_results:
                grid_values_norm = pdp_results["values"][0]
            elif "grid_values" in pdp_results:
                grid_values_norm = pdp_results["grid_values"][0]
            else:
                raise ValueError(f"Keys not found in PDP results: {pdp_results.keys()}")
            
            try:
                orig_feat_idx = continuous_features.index(feat_name)
                dummy_matrix = np.zeros((len(grid_values_norm), len(continuous_features)))
                dummy_matrix[:, orig_feat_idx] = grid_values_norm
                raw_values = scaler.inverse_transform(dummy_matrix)[:, orig_feat_idx]
                x_axis_values = raw_values
                x_label_suffix = " (Raw)"
            except ValueError as e:
                log(f"Errore inverse_transform PDP per {feat_name}: {e}")
                x_axis_values = grid_values_norm
                x_label_suffix = " (Norm)"

            y_pdp = pdp_results["average"][target_idx]

            axes[i].plot(x_axis_values, y_pdp, color="tab:orange")
            axes[i].set_xlabel(readable_name + x_label_suffix)
            axes[i].set_ylabel(f"Dipendenza Parziale ({target_class_name})")
            axes[i].set_title(f"PDP - {readable_name}")
            axes[i].grid(True, alpha=0.3)
            
            pdp_multiclass_data.append({
                "feature": readable_name,
                "target_class": target_class_name,
                "x_values": x_axis_values.tolist() if isinstance(x_axis_values, np.ndarray) else list(x_axis_values),
                "y_values": y_pdp.tolist() if isinstance(y_pdp, np.ndarray) else list(y_pdp)
            })

        plt.suptitle(f"PDP - Random Forest (Multiclasse - Target: {target_class_name})", y=1.05, fontsize=16)
        plt.tight_layout()
        _save_figure(plt.gcf(), "C_04_PDP_Individual_RF_Multiclass")
        extended_analysis_data["C_04_PDP_Multiclass"] = pdp_multiclass_data
except Exception as e:
    log(f"Errore PDP Multiclasse: {e}")
    import traceback
    log(traceback.format_exc())

# =============================================================================
# 6. INTRINSIC EXPLAINABILITY (Decision Tree & Logistic Regression)
# =============================================================================

log("\n" + "="*40)
log(" D. INTRINSIC EXPLAINABILITY")
log("="*40)

log("Visualizzazione Albero Decisionale (Binario)...")
try:
    dt_model = binary_models["Decision Tree"]
    
    plt.figure(figsize=(37, 25))
    binary_class_names = ["Benign", "Attack"]
    
    plot_tree(
        dt_model, 
        max_depth=2,
        feature_names=plot_feature_names, 
        class_names=binary_class_names, 
        filled=True, 
        rounded=True,       
        impurity=False,     
        proportion=True,    
        precision=2,        
        fontsize=40
    )
    plt.title("Struttura Albero Decisionale Binario", fontsize=48)
    # _add_zeek_legend(plt.gcf(), fontsize=26, x=0.99, y=0.99, ha='right', va='top') 
    plt.tight_layout()
    _save_figure(plt.gcf(), "D_01_TreeStructure_Binary_DT")
    
    from sklearn.tree import export_text
    tree_rules = export_text(dt_model, feature_names=plot_feature_names, max_depth=10)
    
    rules_path = TEST_DIR / "D_01_TreeRules_Binary_DT.txt"
    with open(rules_path, "w", encoding="utf-8") as f:
        f.write("REGOLE ALBERO DECISIONALE BINARIO\n")
        f.write("="*50 + "\n\n")
        f.write(tree_rules)
        
    try:
        rel_rules = rules_path.relative_to(BASE_PATH)
    except ValueError:
        rel_rules = rules_path
    log(f"Regole albero salvate: {rel_rules}")
    
    extended_analysis_data["D_01_TreeRules"] = tree_rules
    
except Exception as e:
    log(f"Errore visualizzazione DT: {e}")

log("Visualizzazione Coefficienti LR...")
try:
    lr_model = trained_multi_models["Logistic Regression"]
    coefs = np.mean(np.abs(lr_model.coef_), axis=0)
    
    coef_df = pd.DataFrame({
        "Feature": plot_feature_names,
        "Coefficient": coefs
    }).sort_values("Coefficient", ascending=False).head(15)
    
    plt.figure(figsize=(10, 8))
    sns.barplot(data=coef_df, x="Coefficient", y="Feature", palette="magma")
    plt.title("Coefficienti Regressione Logistica (Impatto Medio Ass.)", fontsize=15)
    # _add_zeek_legend(plt.gcf(), x=0.95, y=0.08)
    plt.tight_layout()
    _save_figure(plt.gcf(), "D_02_Coefficients_Individual_LR")
    extended_analysis_data["D_02_LR_Coefficients"] = coef_df.to_dict(orient="records")
except Exception as e:
    log(f"Errore visualizzazione LR: {e}")

# =============================================================================
# 7. LOCAL EXPLAINABILITY (LIME & WATERFALL)
# =============================================================================

log("\n" + "="*40)
log(" E. LOCAL EXPLAINABILITY (LIME & WATERFALL)")
log("="*40)

lime_analysis_data = []
waterfall_analysis_data = []

attack_indices = np.where(y_binary_test == 1)[0]
if len(attack_indices) > 0:
    sample_indices = attack_indices[:2]
else:
    sample_indices = [0, 1]

rf_model = trained_multi_models["Random Forest"]
explainer_shap_tree = shap.TreeExplainer(rf_model)

for i, idx in enumerate(sample_indices):
    log(f"Analisi Locale Campione {i+1} (Indice {idx})...")
    instance = X_test_processed[idx]
    
    # LIME
    try:
        explainer_lime = lime.lime_tabular.LimeTabularExplainer(
            X_train_processed,
            feature_names=plot_feature_names,
            class_names=tactic_encoder.classes_,
            discretize_continuous=True
        )
        exp = explainer_lime.explain_instance(instance, rf_model.predict_proba, num_features=10, top_labels=1)
        
        top_label = exp.top_labels[0]
        fig = exp.as_pyplot_figure(label=top_label)
        plt.title(f"LIME - Campione #{idx} (Predetto: {tactic_encoder.classes_[top_label]})", fontsize=14)
        plt.tight_layout()
        _save_figure(fig, f"E_01_LIME_Individual_RF_Sample_{idx}_{tactic_encoder.classes_[top_label]}")
        
        lime_analysis_data.append({
            "sample_index": int(idx),
            "predicted_class": str(tactic_encoder.classes_[top_label]),
            "prediction_probabilities": {
                str(cls): float(prob) for cls, prob in zip(tactic_encoder.classes_, exp.predict_proba)
            },
            "explanation": exp.as_list(label=top_label)
        })
        
    except Exception as e:
        import traceback
        log(f"Errore LIME per campione {idx}: {repr(e)}")
        log(traceback.format_exc())

    # SHAP Waterfall
    try:
        shap_values_single = explainer_shap_tree(instance.reshape(1, -1))
        
        shap_values_single.feature_names = plot_feature_names
        
        pred_class = rf_model.predict(instance.reshape(1, -1))[0]
        
        plt.figure()
        shap.plots.waterfall(shap_values_single[0, :, pred_class], show=False, max_display=8)
        fig = plt.gcf()
        plt.title(f"SHAP Waterfall - Campione #{idx} (Predetto: {tactic_encoder.classes_[pred_class]})")
        # _add_zeek_legend(fig, x=0.95, y=0.05)
        plt.tight_layout()
        
        base_val = float(shap_values_single.base_values[0, pred_class]) if hasattr(shap_values_single, "base_values") else 0.0
        shap_vals = shap_values_single.values[0, :, pred_class].tolist() if hasattr(shap_values_single, "values") else []
        final_val = base_val + sum(shap_vals) if shap_vals else base_val
        
        waterfall_analysis_data.append({
            "sample_index": int(idx),
            "predicted_class": str(tactic_encoder.classes_[pred_class]),
            "base_value": base_val,
            "final_value": final_val,
            "feature_values": shap_values_single.data[0].tolist() if hasattr(shap_values_single, "data") else [],
            "shap_values": shap_vals
        })

        _save_figure(fig, f"E_02_Waterfall_Individual_RF_Sample_{idx}_{tactic_encoder.classes_[pred_class]}")

    except Exception as e:
        log(f"Errore SHAP Waterfall: {e}")

log("\n" + "="*60)
log(" ANALISI COMPLETATA CON SUCCESSO")
log("="*60)

extended_analysis_data["E_01_LIME"] = lime_analysis_data
extended_analysis_data["E_02_Waterfall"] = waterfall_analysis_data

# Save Extended Analysis JSON
extended_json_path = TEST_DIR / "extended_analysis.json"
try:
    with open(extended_json_path, "w", encoding="utf-8") as f:
        def np_encoder(object):
            if isinstance(object, np.generic):
                return object.item()
            raise TypeError
            
        json.dump(extended_analysis_data, f, indent=4, default=np_encoder)
    log(f"Analisi numerica estesa salvata: {extended_json_path.name}")
except Exception as e:
    log(f"Errore salvataggio extended_analysis.json: {e}")

# --- ARCHIVIAZIONE FINALE ---

log_file = TEST_DIR / f"test_output_{run_timestamp}.txt"
with open(log_file, "w", encoding="utf-8") as f:
    f.write("\n".join(report_buffer))
try:
    rel_log = log_file.relative_to(BASE_PATH)
except ValueError:
    rel_log = log_file
print(f"Log salvato in: {rel_log}")

try:
    rel_test_dir = TEST_DIR.relative_to(BASE_PATH)
except ValueError:
    rel_test_dir = TEST_DIR
print(f"Tutti gli output sono in: {rel_test_dir}")
