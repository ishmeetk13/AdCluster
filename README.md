# AdCluster: Ad Performance Clustering & Persona Intelligence

**AdCluster** is an unsupervised machine learning platform engineered for digital marketing agencies and performance marketing teams. By converting raw campaign telemetry into standardized efficiency metrics, AdCluster automatically audits ad portfolios, isolates wasted budget, and segments ad variants into **5 actionable performance personas** using K-Means clustering (K=5) and Principal Component Analysis (PCA).

🔗 **Live Interactive Application:** [https://adcluster-prkqskkkyc7jd8rwyfpsja.streamlit.app/](https://adcluster-prkqskkkyc7jd8rwyfpsja.streamlit.app/)

---

## Executive Summary & Problem Statement

Modern AI marketing agencies generate hundreds of ad copy, visual, and headline variants at scale. However, managing large volumes of variants creates a major operational bottleneck:
- Manual spreadsheet analysis is slow, reactive, and doesn't scale.
- Low-performing ads burn client budget for days before being noticed.
- Evaluating raw metrics alone (e.g., total impressions) masks true acquisition cost and profitability.

**The Solution:** AdCluster sits downstream of the ad creation pipeline. It ingests campaign performance data, normalizes efficiency ratios, and tags every ad variant with a clear strategic agency directive (*Scale*, *Pause*, *Redesign*, *Cap*).

---

## Dataset & Feature Engineering

### Dataset Overview
- **Source:** *Sales Conversion Optimization* (Kaggle)
- **Scope:** 1,143 Facebook ad records across 3 anonymous macro-campaigns.
- **Data Cleaning:** Filtered out inactive/zero-click ads, resulting in **936 active ad records**.

### Feature Definitions & Formulas

Raw metric counts (like total clicks or spend) are transformed into standardized efficiency ratios to prevent high-budget campaigns from mathematically dominating low-budget variants during clustering:

1. **CTR (Click-Through Rate):** Measures ad creative appeal and audience hook.
   - Formula: `Clicks / Impressions`

2. **CPC (Cost Per Click):** Measures media buying efficiency across target segments.
   - Formula: `Spent / Clicks`

3. **Conversion Rate (Conv. Rate):** Measures audience purchase intent and landing page effectiveness.
   - Formula: `Approved Conversions / Clicks`

4. **Cost Per Purchase (CPP):** Measures the ultimate acquisition cost per paying customer.
   - If Approved Conversions > 0: `Spent / Approved Conversions`
   - If Approved Conversions = 0: `Spent + Max(Spent)` *(Zero-Conversion Penalty Shift)*
   
   *Note: Zero-conversion ads receive a penalty shift (Spent + $639.95) to prevent division-by-zero errors and force K-Means to isolate non-converting ads into their own cluster.*

5. **Spent:** Total budget consumed by the individual ad variant.

---

## Machine Learning Methodology

**Pipeline:**
Raw Telemetry ➔ Feature Engineering ➔ StandardScaler ➔ K-Means (K=5) ➔ PCA (2D) ➔ Streamlit UI

1. **Feature Normalization:** All 5 continuous features are scaled using `StandardScaler` (Mean = 0, Variance = 1) because K-Means computes Euclidean distance, which requires uniform variance across dimensions.
2. **Cluster Selection (K=5):**
   - **Elbow Method (Inertia):** Evaluated K from 2 to 8; confirmed a sharp inflection in WCSS at K=5.
   - **Silhouette Analysis:** Achieved a global maximum silhouette score of **0.3506** at K=5, demonstrating optimal geometric separation between personas.
3. **Dimensionality Reduction (PCA):** Reduced the 5D feature space down to 2 principal components strictly for interactive visualization in Plotly.

---

## The 5 Ad Personas & Strategic Playbook

| Persona Tag | Ad Count | Avg Spend | Avg Purchases | Conv. Rate | Avg CPP | Strategic Agency Directive |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Core Anchors** | 316 | $53.18 | 1.60 | 10.33% | $37.87 | **Maintain / Scale Incrementally** - Steady revenue backbone with stable acquisition costs. |
| **Budget Drainers** | 313 | $35.04 | 0.00 | 0.00% | $674.99* | **Pause Ad Immediately** - Pure budget leaks. Eliminates non-converting spend. |
| **Hidden Gems** | 71 | $1.82 | 1.07 | 87.04% | $1.70 | **Scale Budget Aggressively** - Exceptional conversion efficiency on small test budgets. |
| **Underperformers** | 132 | $12.72 | 0.23 | 4.08% | $506.89 | **Redesign Copy / Refresh Audience** - Weak conversion rates and elevated CAC on test budgets. |
| **Volume Drivers** | 104 | $280.04 | 3.75 | 2.13% | $181.40 | **Cap Spend / Optimize Funnel** - High total revenue drivers experiencing audience saturation. |

*\*Note: Budget Drainers CPP reflects the standardized zero-conversion penalty shift calculation.*

---

## Web Application Features (Streamlit)

1. **Portfolio Audit Dashboard:** High-level executive view showing total analyzed spend, overall purchase volume, persona distribution pie charts, and identified wasted budget.
2. **Persona Explorer:** Benchmark tables and expandable strategic playbooks breakdown per cluster.
3. **Real-Time Ad Segmenter:** Interactive constrained sliders (min/max dataset bounds) allowing media buyers to test live ad metrics and receive instant persona predictions and recommended actions.
4. **2D/3D Cluster Space Visualizer:** Interactive Plotly charts mapping 2D PCA projections and 3D feature coordinates (Spent x Conv. Rate x CPP).

---

## Tech Stack & Dependencies

- **Language:** Python
- **Machine Learning:** `scikit-learn` (`KMeans`, `StandardScaler`, `PCA`)
- **Web Framework:** `streamlit`
- **Interactive Data Visualization:** `plotly` (`express` & `graph_objects`)
- **Data Manipulation:** `pandas`, `numpy`
- **Model Serialization:** `joblib`

---

## Limitations & Real-World Considerations

1. **Observational vs. Causal Data:** K-Means identifies correlation patterns in historical performance, not direct causality. Scaling a "Hidden Gem" assumes audience intent holds as budget increases.
2. **Dataset Scale:** Trained on 936 Facebook ad records. Performance metrics reflect Meta's auction environment dynamics.
3. **Anonymized Interest Codes:** Categorical targeting fields (`interest`) are anonymous numeric IDs in the underlying dataset.
4. **Saturation Curve Thresholds:** Linear budget adjustments do not replace ad platform fatigue algorithms; high spend bounds should be capped to prevent cost spikes.

---

## Project Architecture

- `app.py` — Streamlit application UI & inference logic
- `cluster_model.pkl` — Serialized bundle (Scaler, KMeans, PCA, Persona Map)
- `KAG_conversion_data.csv` — Raw dataset for portfolio auditing & slider bounds
- `requirements.txt` — Explicit dependency list for cloud deployment
- `README.md` — Technical documentation & project guide
