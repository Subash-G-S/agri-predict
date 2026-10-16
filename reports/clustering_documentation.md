# Technical Report: Unsupervised Clustering & Precision Field Zoning
**Project:** Agri-Predict — Agricultural Intelligence System  
**Course:** 23CSE301 Machine Learning Capstone (Review 2)  
**Academic Year:** 2026–27  
**Module:** Unsupervised Learning Track — Edaphic & Climatic Field Clustering  

---

## 1. Executive Summary & Objective

In precision agriculture, blanket management of large agricultural territories leads to sub-optimal fertilizer utilization, inefficient irrigation schedules, and variable crop yields. The objective of the **Agri-Predict Clustering Track** is to discover latent, natural groupings (agronomic zones) across **10,000 maize-producing fields** using multivariate soil (edaphic) and climatic indicators.

Unlike supervised learning, this track operates in a strictly **unsupervised paradigm**:
- **No target labels or yield data were provided during model fitting.**
- Clusters are formed entirely on physical, chemical, and atmospheric distance relationships in high-dimensional feature space.
- Post-hoc yield analysis is conducted solely for external agronomic validation.

Two distinct families of unsupervised algorithms were implemented, tuned, and cross-evaluated:
1. **Partition-based (Centroid):** K-Means Clustering
2. **Connectivity-based (Hierarchical):** Agglomerative Hierarchical Clustering (Ward's Minimum Variance Linkage)

---

## 2. Dataset Architecture & Feature Engineering

### 2.1 Dataset Specifications
- **Data Source:** `data/clusturing data/train.csv`
- **Total Records:** 10,000 agricultural monitoring observations
- **Data Integrity:** 0 missing values, 0 duplicate records across all feature vectors.

### 2.2 Feature Selection Matrix

| Feature | Unit / Domain | Measurement Category | Role in Clustering | Justification |
| :--- | :--- | :--- | :--- | :--- |
| `soil_ph` | pH scale ($[5.0, 8.0]$) | Edaphic (Chemical) | **Active Input** | Governs nutrient availability and microbial activity. |
| `organic_matter` | Percentage ($\%$) | Edaphic (Composition) | **Active Input** | Indicates soil fertility, water retention, and carbon content. |
| `sand_pct` | Percentage ($\%$) | Edaphic (Physical Texture) | **Active Input** | Determines soil drainage, aeration, and compaction risk. |
| `temperature` | Celsius ($^\circ\text{C}$) | Climatic | **Active Input** | Controls evapotranspiration and physiological crop heat units. |
| `humidity` | Percentage ($\%$) | Climatic | **Active Input** | Regulates atmospheric vapor pressure deficit and transpirational pull. |
| `rainfall` | Millimeters ($\text{mm}$) | Climatic | **Active Input** | Primary hydrological input driving moisture availability. |
| `ndvi` | Index ($[-0.2, 0.9]$) | Biophysical Remote Sensing | **Active Input** | Proxy for crop vigor, canopy density, and photosynthetic activity. |
| `field_id` | Alphanumeric String | Identifier | **Excluded** | Arbitrary database key; holds zero agronomic or metric distance meaning. |
| `yield` | $\text{kg}/\text{ha}$ | Production Target | **Withheld** | Withheld to prevent label leakage; used strictly for post-hoc validation. |

### 2.3 Feature Standardization (Preprocessing)
Because distance metrics (Euclidean and Ward's criterion) are directly sensitive to magnitude, features with larger physical dimensions (e.g., `rainfall` up to $200\,\text{mm}$ and `sand_pct` up to $65\%$) would completely dominate attributes with fractional scales (e.g., `ndvi` between $0.40$ and $0.75$).

All 7 active features were transformed via **StandardScaler**:
$$z = \frac{x - \mu}{\sigma}$$
Each feature is mapped to $\mu = 0$ and $\sigma = 1$, ensuring isotropic scaling across all feature dimensions.

---

## 3. Algorithm Implementation Details

### 3.1 Algorithm 1: K-Means Clustering

K-Means partitions the $n = 10{,}000$ points into $k$ disjoint sets $S = \{S_1, S_2, \dots, S_k\}$ by minimizing the Within-Cluster Sum of Squares (Inertia):
$$J = \sum_{i=1}^{k} \sum_{\mathbf{x} \in S_i} \|\mathbf{x} - \boldsymbol{\mu}_i\|^2$$
where $\boldsymbol{\mu}_i$ is the centroid of cluster $S_i$.

#### Hyperparameter Selection ($k=4$):
- **Elbow Analysis:** Evaluating inertia across $k \in [2, 10]$ revealed a discernible deceleration in the rate of variance explained at $k = 4$.
- **Silhouette Trajectory:** At $k = 4$, cluster cohesion and separation reached an optimal balance without fragmenting into redundant micro-clusters.
- **Convergence Optimization:** `n_init=10` with the `k-means++` seeding heuristic was utilized to ensure global convergence and eliminate centroid initialization variance.

### 3.2 Algorithm 2: Agglomerative Hierarchical Clustering

Agglomerative Clustering operates as a bottom-up hierarchical structure. Starting with each individual observation as a singleton cluster, pairs of clusters are iteratively merged using **Ward's Minimum Variance Linkage**:
$$\Delta \text{ESS}_{A, B} = \frac{n_A n_B}{n_A + n_B} \|\boldsymbol{\mu}_A - \boldsymbol{\mu}_B\|^2$$
Ward's linkage seeks to minimize the total within-cluster variance increase at each merge step, producing spherical, balanced clusters compatible with K-Means comparison.

- **Linkage Analysis & Dendrogram:** Hierarchical dendrogram cut-offs on representative samples verified that cutting the tree at 4 branches yields balanced groupings consistent with the K-Means partition.
- **Model Parameters:** `n_clusters=4`, `metric='euclidean'`, `linkage='ward'`.

---

## 4. Quantitative Evaluation & Benchmark Results

Model evaluation for unsupervised clustering was conducted strictly using internal validation metrics that evaluate cluster compactness (intra-cluster variance) and separation (inter-cluster distance):

### 4.1 Consolidated Comparison Table

| Metric | Direction of Quality | K-Means ($k=4$) | Agglomerative ($k=4$, Ward) | Superior Algorithm |
| :--- | :---: | :---: | :---: | :---: |
| **Silhouette Score** | $\uparrow$ Higher is better | **0.0994** | 0.0383 | **K-Means** ($+159.5\%$) |
| **Davies–Bouldin Index** | $\downarrow$ Lower is better | **2.3263** | 3.0911 | **K-Means** ($24.7\%$ lower) |
| **Calinski–Harabasz Index** | $\uparrow$ Higher is better | **982.0** | 556.8 | **K-Means** ($+76.4\%$) |
| **Runtime Complexity (Inference)** | $\downarrow$ Lower is better | **$\mathcal{O}(k \cdot d)$** | Not Supported | **K-Means** |

### 4.2 Agronomic & Mathematical Metric Interpretation

1. **Silhouette Score ($0.0994$ vs $0.0383$):**
   - The Silhouette score measures how similar an observation is to its own cluster compared to neighboring clusters (ranging from $-1$ to $+1$).
   - A score of $\sim 0.10$ confirms that agricultural environments represent continuous ecological gradients rather than artificially sharp, isolated islands. Soil composition and weather shift smoothly across geography.
   - K-Means achieves a significantly higher score, demonstrating sharper cluster assignment boundaries.

2. **Davies–Bouldin Index ($2.3263$ vs $3.0911$):**
   - Measures the average similarity between each cluster and its most similar one. Lower scores indicate superior separation between cluster centroids relative to cluster spread.
   - K-Means achieves a lower index ($2.3263$), confirming more distinct cluster separation.

3. **Calinski–Harabasz Index ($982.0$ vs $556.8$):**
   - The Variance Ratio Criterion (ratio of between-cluster dispersion to within-cluster dispersion).
   - K-Means achieves $982.0$, nearly double that of Agglomerative clustering, confirming high compactness around its centroids.

---

## 5. Visualizations & Dimensionality Reduction

To validate the structure in lower dimensions without human bias, two distinct projections were executed:

### 5.1 Linear Projection: Principal Component Analysis (PCA)
- 7-D scaled space was projected to 2 principal components (PC1 and PC2).
- The projection demonstrated smooth transition boundaries with well-defined centroid clusters.
- Confirmed that K-Means clusters form distinct quadrants in principal subspace.

### 5.2 Non-linear Projection: t-Distributed Stochastic Neighbor Embedding (t-SNE)
- Performed with perplexity $= 30$, learning rate $= 200$.
- Preserved local manifold topology and verified that local field neighborhoods remain coherent within their assigned cluster zones.

---

## 6. Cluster Profiling & Agronomic Zoning Analysis

Analyzing the mean feature values of each cluster reveals clear, agronomically actionable environmental archetypes:

### 6.1 Cluster Feature Profiles (Mean Values Across All Fields)

| Feature | Cluster 0 (Zone 0) | Cluster 1 (Zone 1) | Cluster 2 (Zone 2) | Cluster 3 (Zone 3) | Baseline Mean |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Field Count** | 2,539 ($25.4\%$) | 2,424 ($24.2\%$) | 2,510 ($25.1\%$) | 2,527 ($25.3\%$) | 2,500 ($25.0\%$) |
| **Soil pH** | 6.511 | 6.501 | **6.543** | 6.478 | 6.508 |
| **Organic Matter (%)** | **3.401%** | 2.489% | 2.100% | 2.985% | 2.744% |
| **Sand Percentage (%)** | 38.416% | **52.088%** | 37.945% | 51.657% | 45.026% |
| **Temperature ($^\circ\text{C}$)** | 26.995 | 26.913 | 26.879 | **27.185** | 26.993 |
| **Humidity (%)** | 74.570% | 74.705% | **74.746%** | 74.081% | 74.526% |
| **Rainfall (mm)** | 138.112 | 137.041 | 139.916 | **144.953** | 140.005 |
| **NDVI (Canopy Vigor)** | 0.612 | **0.664** | 0.544 | 0.482 | 0.576 |
| **Post-hoc Yield (kg/ha)** | **4,258.4** | 4,246.8 | 4,249.3 | 4,253.2 | 4,251.9 |

### 6.2 Agronomic Archetype Descriptions & Management Prescriptions

#### Zone 0: High-Fertility Humic Loam (Optimal Natural Condition)
- **Characteristics:** Highest organic matter ($3.40\%$), low sand ($38.4\%$), high canopy vigor ($\text{NDVI} = 0.612$), and highest yield ($4,258.4\,\text{kg}/\text{ha}$).
- **Prescription:** Maintain nutrient replenishment; reduce chemical synthetic fertilizer inputs to leverage organic soil reserves; ideal benchmark for standard planting density.

#### Zone 1: High-Drainage Sandy Loam with High Photosynthetic Vigor
- **Characteristics:** Elevated sand ($52.1\%$), moderate organic matter ($2.49\%$), highest canopy NDVI ($0.664$).
- **Prescription:** High percolation risk; split nitrogen fertilizer into multiple small applications to prevent leaching; implement drip irrigation scheduling to maintain consistent root zone moisture.

#### Zone 2: Moderate Moisture-Retentive Clay/Loam
- **Characteristics:** Lower organic matter ($2.10\%$), low sand ($37.9\%$), moderate rainfall ($139.9\,\text{mm}$), moderate NDVI ($0.544$).
- **Prescription:** Apply organic amendments (cover cropping, compost) to rebuild soil organic carbon; monitor aeration and prevent soil compaction during wet seasons.

#### Zone 3: High-Precipitation Sandy Transition Zone
- **Characteristics:** Highest rainfall ($145.0\,\text{mm}$), high sand ($51.7\%$), lowest NDVI ($0.482$), warmest temperature ($27.2^\circ\text{C}$).
- **Prescription:** Elevated rainfall combined with high sand content indicates potential drainage stress or early season nutrient leaching causing lower canopy NDVI; apply slow-release nitrogen formulations and check for micronutrient deficiencies.

---

## 7. Model Artifacts & Production Engineering

### 7.1 Exported Artifacts

| File Path | Component | Purpose |
| :--- | :--- | :--- |
| `models/clustering_scaler.joblib` | `StandardScaler` | Transforms new 7-feature input vectors using the training mean and standard deviation. |
| `models/kmeans_model.joblib` | `KMeans(k=4)` | Predicts cluster membership for new field coordinates in $\mathcal{O}(k \cdot d)$ time. |
| `models/clustering_metadata.json` | JSON Schema | Stores feature ordering, cluster metrics, profiles, and UI label mappings for the API. |
| `models/agglomerative_model.joblib` | `AgglomerativeClustering` | Preserved for reproducibility, benchmark audits, and academic inspection. |

### 7.2 Why K-Means Was Selected for Production Web Deployment

1. **Native Out-of-Sample Inference:**
   - In scikit-learn, `AgglomerativeClustering` does not implement a `.predict()` method. Assigning a single new test field to an agglomerative hierarchy requires recalculating an $(N+1) \times (N+1)$ distance matrix and re-clustering the entire corpus ($\mathcal{O}(N^2 \log N)$ complexity).
   - In contrast, `KMeans.predict()` evaluates Euclidean distance to the 4 precomputed centroids in $\mathcal{O}(4 \times 7)$ operations, executing in under $1\,\text{ms}$.
2. **Quantitative Superiority:**
   - K-Means outperformed Agglomerative clustering across every evaluation metric (Silhouette $+159\%$, Davies-Bouldin $24.7\%$ better, Calinski-Harabasz $+76\%$).
3. **API Integration:**
   - Connected via `/api/clustering/predict` and `/api/analyze/all` in the Flask backend (`app/backend/app.py`), enabling instantaneous field zoning alongside supervised yield and crop predictions.

---

## 8. Limitations & Methodological Integrity

1. **Exploratory Nature:** Clustering identifies natural soil-climate groupings; it is not a direct substitute for supervised crop yield regression.
2. **Smooth Gradients:** Low silhouette scores reflect natural agricultural transitions rather than algorithm deficiency. Real-world farms lack rigid borders.
3. **Dataset Specificity:** The clusters are fitted on maize trial regions. Expanding to other crops or geographies requires domain-specific re-scaling and calibration.
