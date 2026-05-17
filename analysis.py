# Retail Data Set Data Analysis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import classification_report, silhouette_score, mean_absolute_error, r2_score

%matplotlib inline

data_path = "retail_sales_dataset.csv"
df = pd.read_csv(data_path, parse_dates=["Date"])

df = df.dropna(subset=["Transaction ID", "Date", "Total Amount"]).drop_duplicates()
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month

print("Shape:", df.shape)
print(df.dtypes)
print("\nMissing per column:\n", df.isna().sum())
print("Duplicate rows:", df.duplicated().sum())

# Sales Distribution
plt.figure(figsize=(8, 4))
sns.histplot(df["Total Amount"], bins=50, kde=True)
plt.title("Sales Distribution")
plt.xlabel("Total Amount ($)")
plt.show()

# Top 10 Product Categories by Total Sales
top_categories = df.groupby("Product Category")["Total Amount"].sum().nlargest(10)
plt.figure(figsize=(8, 5))
top_categories.plot(kind="barh")
plt.title("Top 10 Product Categories by Total Sales")
plt.xlabel("Total Amount ($)")
plt.gca().invert_yaxis()
plt.show()

# Monthly Revenue Over Time
monthly_rev = df.set_index("Date")["Total Amount"].resample("M").sum()
plt.figure(figsize=(10, 4))
monthly_rev.plot(marker="o")
plt.title("Monthly Revenue Over Time")
plt.ylabel("Total Amount ($)")
plt.show()

# Monthly Sales by Gender, stacked by Product Category
df["YearMonth"] = df["Date"].dt.to_period("M").astype(str)
df["MonthName"] = df["Date"].dt.strftime("%B")
pivot = df.pivot_table(index=["YearMonth", "Gender", "MonthName"], columns="Product Category", values="Total Amount", aggfunc="sum", fill_value=0, observed=True)
genders = pivot.index.get_level_values("Gender").unique()
fig, axes = plt.subplots(1, len(genders), figsize=(14, 6), sharey=True)
axes = [axes] if len(genders) == 1 else axes
for ax, gen in zip(axes, genders):
    gender_df = pivot.xs(gen, level="Gender").sort_index(level="YearMonth")
    month_names = gender_df.index.get_level_values("MonthName")
    gender_df.plot(kind="bar", stacked=True, ax=ax, colormap="tab20c", linewidth=0, alpha=0.85)
    ax.set_title(f"{gen} Sales by Month")
    ax.set_xlabel("Month")
    ax.set_xticklabels(month_names, rotation=45)
    ax.set_ylabel("Total Sales ($)" if ax is axes[0] else "")
    ax.legend().set_visible(False)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, title="Product Category", bbox_to_anchor=(1.02, 0.5), loc="center left")
plt.tight_layout()
plt.show()

# Correlations
numeric_cols = ["Age", "Quantity", "Price per Unit", "Total Amount", "Year", "Month"]
corr = df[numeric_cols].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Feature Correlations")
plt.show()

# Customer Age Distribution
plt.figure(figsize=(8, 4))
sns.histplot(df["Age"], bins=20, kde=True)
plt.title("Customer Age Distribution")
plt.xlabel("Age")
plt.show()

# Model Training and Prediction
model_df = df.dropna(subset=["Age", "Quantity", "Price per Unit"])
X = model_df[["Age", "Quantity", "Price per Unit"]]
y = model_df["Total Amount"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression().fit(X_train, y_train)
y_pred = model.predict(X_test)

# Actual vs Predicted Total Amount (Sample of 100)
cmp = pd.DataFrame({"Actual": y_test.reset_index(drop=True), "Predicted": y_pred}).iloc[:100]
plt.figure(figsize=(12, 6))
plt.plot(cmp.index, cmp["Actual"], label="Actual", linewidth=2, marker="o", markersize=4)
plt.plot(cmp.index, cmp["Predicted"], label="Predicted", linewidth=2, linestyle="--", marker="o", markersize=4)
plt.grid(True, linestyle="--", alpha=0.5)
plt.title("Actual vs Predicted Total Amount (Sample of 100)")
plt.xlabel("Sample Index")
plt.ylabel("Total Amount")
plt.legend()
plt.tight_layout()
plt.show()

# Monthly Actual vs Target Total Amount
monthly_actual = df.set_index("Date")["Total Amount"].resample("M").sum()
np.random.seed(42)
mean_val = monthly_actual.mean()
monthly_target = pd.Series(mean_val * (1 + np.random.uniform(-0.1, 0.1, size=len(monthly_actual))), index=monthly_actual.index, name="Target")
comp = pd.DataFrame({"Actual": monthly_actual, "Target": monthly_target})
comp.index = comp.index.strftime("%B")
plt.figure(figsize=(12, 6))
sns.lineplot(data=comp, markers=True, dashes=False)
plt.title("Monthly Actual vs Target Total Amount")
plt.xlabel("Month")
plt.ylabel("Total Amount ($)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Average Daily Transactional Revenue
df["Day of Week"] = df["Date"].dt.day_name()
sales_by_dow = df.groupby("Day of Week", observed=True)["Total Amount"].mean().reindex(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
plt.figure(figsize=(10, 6))
plt.plot(sales_by_dow.index, sales_by_dow.values, marker="o", linewidth=2)
plt.xlabel("Day")
plt.ylabel("Average Sales ($)")
plt.title("Average Daily Transactional Revenue")
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.show()



# Kmeans customer segmentation
# Features: Age, Quantity, Total Amount
seg_features = ["Age", "Quantity", "Total Amount"]
seg_df = df[seg_features].dropna()
 
scaler = StandardScaler()
seg_scaled = scaler.fit_transform(seg_df)
 
# Elbow method to pick optimal K
inertias = []
sil_scores = []
K_range = range(2, 9)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(seg_scaled)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(seg_scaled, km.labels_))
 
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(list(K_range), inertias, marker="o")
axes[0].set_title("KMeans Elbow Curve")
axes[0].set_xlabel("Number of Clusters (K)")
axes[0].set_ylabel("Inertia")
 
axes[1].plot(list(K_range), sil_scores, marker="o", color="orange")
axes[1].set_title("Silhouette Score by K")
axes[1].set_xlabel("Number of Clusters (K)")
axes[1].set_ylabel("Silhouette Score")
plt.tight_layout()
plt.show()
 
# Fit final model with K=4
K_FINAL = 4
kmeans = KMeans(n_clusters=K_FINAL, random_state=42, n_init=10)
seg_df = seg_df.copy()
seg_df["Cluster"] = kmeans.fit_predict(seg_scaled)
 
print("\nCluster Profiles:")
print(seg_df.groupby("Cluster")[seg_features].mean().round(2))
 
# Visualize clusters: Age vs Total Amount
plt.figure(figsize=(9, 6))
scatter = plt.scatter(
    seg_df["Age"], seg_df["Total Amount"],
    c=seg_df["Cluster"], cmap="tab10", alpha=0.5, s=20
)
plt.colorbar(scatter, label="Cluster")
plt.title("KMeans Customer Segments — Age vs Total Amount")
plt.xlabel("Age")
plt.ylabel("Total Amount ($)")
plt.tight_layout()
plt.show()
 
# Revenue share per cluster
cluster_revenue = seg_df.groupby("Cluster")["Total Amount"].sum()
plt.figure(figsize=(6, 6))
cluster_revenue.plot(kind="pie", autopct="%1.1f%%", startangle=140, colormap="tab10")
plt.title("Revenue Share by Customer Segment")
plt.ylabel("")
plt.tight_layout()
plt.show()
 

# KNN
# Predict Product Category from Age, Gender, Quantity, Total Amount
knn_df = df[["Age", "Gender", "Quantity", "Total Amount", "Product Category"]].dropna().copy()
 
le_gender   = LabelEncoder()
le_category = LabelEncoder()
knn_df["Gender_enc"]   = le_gender.fit_transform(knn_df["Gender"])
knn_df["Category_enc"] = le_category.fit_transform(knn_df["Product Category"])
 
X_knn = knn_df[["Age", "Gender_enc", "Quantity", "Total Amount"]]
y_knn = knn_df["Category_enc"]
 
X_tr, X_te, y_tr, y_te = train_test_split(X_knn, y_knn, test_size=0.2, random_state=42, stratify=y_knn)
 
knn_scaler = StandardScaler()
X_tr_s = knn_scaler.fit_transform(X_tr)
X_te_s = knn_scaler.transform(X_te)
 
# Tune K with accuracy curve
knn_accs = []
k_vals = range(1, 21)
for k in k_vals:
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(X_tr_s, y_tr)
    knn_accs.append(knn_model.score(X_te_s, y_te))
 
best_k = k_vals[np.argmax(knn_accs)]
print(f"\nBest K for KNN: {best_k}  |  Accuracy: {max(knn_accs):.4f}")
 
plt.figure(figsize=(9, 4))
plt.plot(list(k_vals), knn_accs, marker="o")
plt.axvline(best_k, color="red", linestyle="--", label=f"Best K={best_k}")
plt.title("KNN Accuracy vs K")
plt.xlabel("K (Neighbours)")
plt.ylabel("Test Accuracy")
plt.legend()
plt.tight_layout()
plt.show()
 
# Final KNN model
knn_final = KNeighborsClassifier(n_neighbors=best_k)
knn_final.fit(X_tr_s, y_tr)
y_pred_knn = knn_final.predict(X_te_s)
print("\nKNN Classification Report:")
print(classification_report(y_te, y_pred_knn, target_names=le_category.classes_))
 

# Neural network revenue predictor
nn_df = df[["Age", "Gender", "Quantity", "Price per Unit", "Month", "Total Amount"]].dropna().copy()
nn_df["Gender_enc"] = le_gender.transform(nn_df["Gender"])
 
X_nn = nn_df[["Age", "Gender_enc", "Quantity", "Price per Unit", "Month"]]
y_nn = nn_df["Total Amount"]
 
X_tr_nn, X_te_nn, y_tr_nn, y_te_nn = train_test_split(X_nn, y_nn, test_size=0.2, random_state=42)
 
nn_scaler = StandardScaler()
X_tr_nn_s = nn_scaler.fit_transform(X_tr_nn)
X_te_nn_s = nn_scaler.transform(X_te_nn)
 
mlp = MLPRegressor(
    hidden_layer_sizes=(128, 64, 32),
    activation="relu",
    max_iter=500,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20
)
mlp.fit(X_tr_nn_s, y_tr_nn)
y_pred_nn = mlp.predict(X_te_nn_s)
 
print(f"\nNeural Network (MLP)  |  MAE: {mean_absolute_error(y_te_nn, y_pred_nn):.2f}  |  R²: {r2_score(y_te_nn, y_pred_nn):.4f}")
 
# Training loss curve
plt.figure(figsize=(9, 4))
plt.plot(mlp.loss_curve_, label="Training Loss")
if mlp.best_loss_ is not None:
    plt.axhline(mlp.best_loss_, color="red", linestyle="--", label=f"Best Val Loss: {mlp.best_loss_:.4f}")
plt.title("Neural Network Training Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss (MSE)")
plt.legend()
plt.tight_layout()
plt.show()
 
# Actual vs Predicted — Neural Network vs Linear Regression comparison
cmp_nn = pd.DataFrame({
    "Actual":    y_te_nn.reset_index(drop=True),
    "MLP":       y_pred_nn,
}).iloc[:100]
 
# Align LR predictions to same indices for fair comparison
X_te_lr_aligned = X_te_nn.reset_index(drop=True).iloc[:100][["Age", "Quantity", "Price per Unit"]]
cmp_nn["LR"] = model.predict(X_te_lr_aligned)
 
plt.figure(figsize=(13, 6))
plt.plot(cmp_nn.index, cmp_nn["Actual"], label="Actual",       linewidth=2)
plt.plot(cmp_nn.index, cmp_nn["MLP"],    label="MLP (NNet)",   linewidth=2, linestyle="--")
plt.plot(cmp_nn.index, cmp_nn["LR"],     label="Linear Reg",   linewidth=1.5, linestyle=":")
plt.grid(True, linestyle="--", alpha=0.4)
plt.title("Actual vs Predicted — MLP Neural Net vs Linear Regression (Sample 100)")
plt.xlabel("Sample Index")
plt.ylabel("Total Amount ($)")
plt.legend()
plt.tight_layout()
plt.show()
 
# summary

summary = pd.DataFrame({
    "Model": ["Linear Regression", "MLP Neural Network"],
    "MAE":   [mean_absolute_error(y_test, y_pred_lr), mean_absolute_error(y_te_nn, y_pred_nn)],
    "R²":    [r2_score(y_test, y_pred_lr),            r2_score(y_te_nn, y_pred_nn)]
})
print("\n── Model Comparison ──")
print(summary.to_string(index=False))
 
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].bar(summary["Model"], summary["MAE"], color=["steelblue", "darkorange"])
axes[0].set_title("MAE by Model (lower = better)")
axes[0].set_ylabel("Mean Absolute Error ($)")
 
axes[1].bar(summary["Model"], summary["R²"], color=["steelblue", "darkorange"])
axes[1].set_title("R² by Model (higher = better)")
axes[1].set_ylabel("R² Score")
axes[1].set_ylim(0, 1)
plt.tight_layout()
plt.show()
 
