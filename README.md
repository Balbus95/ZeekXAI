# ZeekXAI - Analisi e Classificazione di Attacchi in Infrastrutture Critiche

ZeekXAI è un progetto di tesi che mira a rilevare e classificare tattiche di attacco informatico (basate sulla matrice MITRE ATT&CK) all'interno di infrastrutture critiche, utilizzando dati di traffico di rete (log Zeek).

Il sistema integra algoritmi di Machine Learning (Random Forest, Decision Tree, Logistic Regression, Naive Bayes) con tecniche di **Explainable AI (XAI)** come SHAP (Shapley Additive Explanations) e PDP (Partial Dependence Plots) per rendere trasparenti e interpretabili le decisioni dei modelli.

## 🚀 Caratteristiche Principali

*   **Analisi Dati Zeek**: Caricamento e parsing efficiente di log di rete in formato Parquet.
*   **Feature Engineering**: Calcolo automatico di metriche derivate (es. `pktAtsec`, `BitRate`, `interTime`) per arricchire il dataset.
*   **Machine Learning**: Addestramento e valutazione di modelli per:
    *   **Classificazione Binaria**: Distinzione tra traffico benigno e maligno.
    *   **Classificazione Multiclasse**: Identificazione della specifica tattica MITRE (es. *Reconnaissance*, *Discovery*, *Resource Development*).
*   **Explainable AI (XAI)**:
    *   **SHAP Global & Local**: Analisi dell'importanza delle feature e del loro impatto sulle predizioni (Beeswarm plots, Bar charts).
    *   **Partial Dependence Plots (PDP)**: Visualizzazione della relazione marginale tra feature e output del modello.
*   **Reportistica**: Generazione automatica di grafici (PDF) e metriche di performance (CSV/JSON).

## 🛠️ Installazione

Assicurati di avere Python 3.8+ installato.

1.  Clona il repository:
    ```bash
    git clone https://github.com/username/ZeekXAI.git
    cd ZeekXAI
    ```

2.  Crea un ambiente virtuale (opzionale ma consigliato):
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # Su Windows: .venv\Scripts\activate
    ```

3.  Installa le dipendenze:
    ```bash
    pip install -r requirements.txt
    ```

## 📦 Struttura del Progetto

*   `zeekXAI.py`: Script principale per l'addestramento dei modelli e la generazione delle spiegazioni XAI.
*   `UWF-ZeekDataFall22/`: Cartella contenente il dataset (file Parquet).
*   `cache/`: Directory per il salvataggio dei modelli addestrati e dataset processati (per velocizzare esecuzioni successive).
*   `output/`: Directory contenente i risultati (grafici, report, log).

## 💻 Utilizzo

### Esecuzione Completa (Training + XAI)
Per avviare l'intera pipeline di analisi, eseguire:

```bash
python zeekXAI.py
```

Lo script eseguirà:
1.  Caricamento e pulizia dei dati.
2.  Addestramento dei modelli (se non presenti in cache).
3.  Valutazione delle performance.
4.  Generazione dei plot SHAP e PDP nella cartella `output/test_<timestamp>`.

## 📊 Output Generati

Tutti i risultati vengono salvati in `output/test_<timestamp>/`:
*   **Metriche**: `B_01_MetricsTable_Comparative_AllMulticlass.csv`
*   **Grafici**:
    *   `A_01_ClassDistribution...pdf`: Distribuzione delle classi.
    *   `B_02_ConfusionMatrix...pdf`: Matrici di confusione.
    *   `C_01_SHAP_Beeswarm...pdf`: Grafici Beeswarm per l'interpretabilità.
    *   `C_03_PDP...pdf`: Grafici di dipendenza parziale.
