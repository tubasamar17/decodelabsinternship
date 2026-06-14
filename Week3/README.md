#  decodelabsinternships
# Customer Segmentation Using K-Means Clustering

##  Project Overview

This project is part of my **Data Science Internship at DecodeLabs (Week 2)**. The goal is to perform **unsupervised learning** on customer data to discover hidden segments and create actionable business personas.

**Dataset:** Mall Customer Segmentation Data (Kaggle) – 200 customers  
**Algorithm:** K-Means Clustering  
**Techniques Used:** PCA, Elbow Method, Silhouette Score

---

##  Business Problem

A retail company wants to understand its customers better. They have data on annual income and spending scores but no predefined customer categories. Using clustering, we can group customers into meaningful segments and tailor marketing strategies accordingly.

---

##  Tasks Completed

### 1. Data Loading & Preprocessing
- Loaded `mall_customers.csv` (200 rows, 5 columns)
- Selected features: `Annual Income (k$)` and `Spending Score (1-100)`
- Scaled data using `StandardScaler` for distance-based algorithm

### 2. Dimensionality Reduction (PCA)
- Applied PCA to reduce 2 features into 2 principal components
- Visualized clusters in PCA-transformed space

### 3. Optimal K Determination
- **Elbow Method:** Plotted inertia vs. number of clusters → elbow at K=5
- **Silhouette Score:** Calculated scores for K=2 to 10 → confirmed K=5 (score: 0.55)

### 4. K-Means Clustering
- Applied K-Means with K=5, random_state=42, n_init=10
- Assigned each customer to a cluster

### 5. Business Personas Creation
Analyzed each cluster's characteristics and created 5 customer personas:

| Cluster | Persona | Characteristics | Business Strategy |
|---------|---------|-----------------|--------------------|
| 0 | Standard | Medium income, medium spending | Maintain with regular offers |
| 1 | High-Value | High income, high spending | Premium products, loyalty rewards |
| 2 | Careful | High income, low spending | Discount campaigns, coupons |
| 3 | Target | Low income, high spending | Installment plans, budget deals |
| 4 | Low Engagement | Low income, low spending | Re-engagement campaigns |

---

##  Visualizations Generated

| Graph | Purpose |
|-------|---------|
| Elbow Method | Find optimal K using inertia |
| Silhouette Score | Validate optimal K mathematically |
| Cluster Scatter Plot | Visualize segments (Income vs Spending) |
| PCA Visualization | View clusters in reduced dimensions |

---

##  File Structure
# project2
 mall_customers.csv # Raw dataset
 task3_segmentation.py # Complete Python code
 README.md # Project documentation

##   How to Run

1. Install required libraries:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
2. Run the script:
   python task3_segmentation.py

   
Name: Tooba Samar
Internship: DecodeLabs Data Science Intern
Week: 2
Project: Customer Segmentation (Unsupervised Learning)
