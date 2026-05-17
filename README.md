# Retail-Data-Set-Analysis

This repository contains Python code that demonstrates end-to-end data analysis and machine learning pipeline built on a 1,000-transaction retail dataset. Covers exploratory data analysis, customer segmentation, product classification, and revenue prediction using three distinct ML approaches. Tech Stack: Python, pandas, NumPy, matplotlib, seaborn, scikit-learn

Models & Results

KMeans Clustering — segmented customers into 4 distinct behavioral groups; identified a high-value middle-aged segment (~$1,345 avg spend) and an underserved youth segment (~$211 avg spend)
KNN Classifier — predicted product category purchase with 38% accuracy (vs 33% random baseline); weak signal confirmed that demographics alone are insufficient category predictors
MLP Neural Network — achieved R²=0.995 and MAE=$23 vs Linear Regression's MAE=$173, demonstrating the value of non-linear modeling for revenue prediction

Key Business Insights

- Nov–Dec sales jump 35% vs yearly average → seasonal inventory and promo planning opportunity
- Saturday revenue is around 1.4x Tuesday → mid-week flash sales could smooth demand
- Top 10% of orders (>$800) drive  around 45% of revenue
- Under-25 customers contribute the fewest dollars → opportunity for entry-level SKUs
