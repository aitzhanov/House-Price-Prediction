# House Price Prediction — Combined Pipeline
# Authors: Makhanbet Alikhan (EDA), Matmusaev Alikhan (Preprocessing), Aitzhanov Alikhan (ML Model), Nurdaulet (reporting, DEMO video)
#
# Pipeline:
#   1. Data Loading
#   2. EDA (Exploratory Data Analysis)
#   3. Preprocessing & Feature Engineering
#   4. ML Model (RandomForestRegressor)
#
# All plots are saved to: graphs/

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings

warnings.filterwarnings("ignore")
sns.set_theme(style="darkgrid", palette="muted")
plt.rcParams["figure.figsize"] = (12, 6)

os.makedirs("graphs", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# 1. Data Loading

db = pd.read_csv("data/dataset_HousePrice.csv")
print(f"Dataset shape: {db.shape}")

# 2. EDA

print("\nFirst 5 rows:")
print(db.head())

print("\nData types:")
print(db.dtypes.value_counts().rename("Count").to_frame())

print("\nDescriptive statistics (top 20 by std):")
print(db.describe().T.sort_values("std", ascending=False).head(20))


def missing_summary(df, name="DataFrame"):
    miss = df.isnull().sum()
    miss = miss[miss > 0].sort_values(ascending=False)
    miss_pct = (miss / len(df) * 100).round(2)
    result = pd.DataFrame({"Missing": miss, "Missing_%": miss_pct})
    print(f"\n{name}: {result.shape[0]} columns with missing values")
    return result


train_miss = missing_summary(db, "DB")
print(train_miss.head(20))

# Plot 1: SalePrice distribution — original, log1p, QQ-plot
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].hist(db["SalePrice"], bins=50, color="steelblue", edgecolor="white")
axes[0].set_title("SalePrice Distribution (Original)")
axes[0].set_xlabel("Price ($)")

axes[1].hist(np.log1p(db["SalePrice"]), bins=50, color="mediumseagreen", edgecolor="white")
axes[1].set_title("SalePrice Distribution (Log1p)")
axes[1].set_xlabel("log(Price)")

stats.probplot(np.log1p(db["SalePrice"]), plot=axes[2])
axes[2].set_title("QQ-Plot — log(SalePrice)")

plt.tight_layout()
plt.savefig("graphs/01_saleprice_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: graphs/01_saleprice_distribution.png")

skew_orig = db["SalePrice"].skew()
skew_log = np.log1p(db["SalePrice"]).skew()
print(f"Skewness (original): {skew_orig:.4f}")
print(f"Skewness (log1p):    {skew_log:.4f}")

# Plot 2: Top 15 features by absolute correlation with SalePrice
num_cols = db.select_dtypes(include="number").columns.tolist()

top_corr = (
    db[num_cols].corr()["SalePrice"]
    .drop("SalePrice")
    .abs()
    .sort_values(ascending=False)
    .head(15)
)

plt.figure(figsize=(10, 7))
top_corr.sort_values().plot(kind="barh", color="steelblue", edgecolor="white")
plt.title("Top 15 Features — Absolute Correlation with SalePrice", fontsize=13)
plt.xlabel("|Pearson Correlation|")
plt.tight_layout()
plt.savefig("graphs/02_top15_correlations.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: graphs/02_top15_correlations.png")

# Plot 3: Correlation heatmap for top 10 features
top10_cols = top_corr.head(10).index.tolist() + ["SalePrice"]
plt.figure(figsize=(12, 9))
sns.heatmap(
    db[top10_cols].corr(),
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    linewidths=0.5,
    square=True,
)
plt.title("Correlation Heatmap — Top 10 Features", fontsize=13)
plt.tight_layout()
plt.savefig("graphs/03_correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: graphs/03_correlation_heatmap.png")

# Plot 4: Scatter plots of key numeric features vs SalePrice
key_features = ["OverallQual", "GrLivArea", "TotalBsmtSF", "GarageArea", "1stFlrSF", "GarageCars"]

fig, axes = plt.subplots(2, 3, figsize=(20, 10))
axes = axes.flatten()

for i, feat in enumerate(key_features):
    axes[i].scatter(db[feat], db["SalePrice"], alpha=0.4, s=15, color="steelblue")
    m, b = np.polyfit(db[feat].fillna(0), db["SalePrice"], 1)
    x_line = np.linspace(db[feat].min(), db[feat].max(), 100)
    axes[i].plot(x_line, m * x_line + b, color="red", lw=1.5)
    axes[i].set_xlabel(feat)
    axes[i].set_ylabel("SalePrice")
    axes[i].set_title(f"{feat} vs SalePrice")

plt.suptitle("Key Numeric Features vs SalePrice", fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig("graphs/04_key_features_scatter.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: graphs/04_key_features_scatter.png")

# Plot 5: Total SF vs SalePrice colored by OverallQual
db["TotalSF"] = (
    db["TotalBsmtSF"].fillna(0) + db["1stFlrSF"].fillna(0) + db["2ndFlrSF"].fillna(0)
)

plt.figure(figsize=(10, 6))
scatter = plt.scatter(
    db["TotalSF"], db["SalePrice"], c=db["OverallQual"], cmap="viridis", alpha=0.6, s=20
)
plt.colorbar(scatter, label="Overall Quality")
plt.xlabel("Total Square Footage")
plt.ylabel("Sale Price ($)")
plt.title("Total SF vs SalePrice (colored by OverallQual)")
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}k"))
plt.tight_layout()
plt.savefig("graphs/05_totalsf_vs_saleprice.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: graphs/05_totalsf_vs_saleprice.png")

# 3. Preprocessing & Feature Engineering

# Remove outliers
db = db[db["GrLivArea"] < 4500]
print(f"\nShape after removing outliers: {db.shape}")

# Fill missing values — numerical
db["LotFrontage"] = db["LotFrontage"].fillna(db["LotFrontage"].median())
db["MasVnrArea"] = db["MasVnrArea"].fillna(0)

# Fill missing values — basement columns
bsmt_cols = ["BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2"]
for col in bsmt_cols:
    db[col] = db[col].fillna("None")

# Fill missing values — garage columns
garage_cols = ["GarageType", "GarageFinish", "GarageQual", "GarageCond"]
for col in garage_cols:
    db[col] = db[col].fillna("None")

# Fill remaining categorical and numerical columns
db["MasVnrType"] = db["MasVnrType"].fillna("None")
db["Electrical"] = db["Electrical"].fillna(db["Electrical"].mode()[0])

for col in db.select_dtypes(include="object").columns:
    db[col] = db[col].fillna("None")

for col in db.select_dtypes(include="number").columns:
    db[col] = db[col].fillna(db[col].median())

print(f"Missing values left: {db.isnull().sum().sum()}")

# Feature engineering
db["HouseAge"] = db["YrSold"] - db["YearBuilt"]
db["RemodAge"] = db["YrSold"] - db["YearRemodAdd"]
db["TotalBathrooms"] = (
    db["FullBath"] + 0.5 * db["HalfBath"] + db["BsmtFullBath"] + 0.5 * db["BsmtHalfBath"]
)
db["HasGarage"] = (db["GarageArea"] > 0).astype(int)
db["HasBasement"] = (db["TotalBsmtSF"] > 0).astype(int)
print("New features created: HouseAge, RemodAge, TotalBathrooms, HasGarage, HasBasement")

# Label encoding for categorical columns
le = LabelEncoder()
for col in db.select_dtypes(include="object").columns:
    db[col] = le.fit_transform(db[col].astype(str))
print("Label encoding completed")

print(f"Final dataset shape: {db.shape}")
print(db.head())

# Save preprocessed dataset
db.to_csv("data/db_preprocessed.csv", index=False)
print("Saved: data/db_preprocessed.csv")

# --- 4. ML Model ---

data = pd.read_csv("data/db_preprocessed.csv")
print(f"\nLoaded preprocessed dataset: {data.shape}")

# Split features and target
y = data["SalePrice"]
X = data.drop(columns=["SalePrice", "Id"])
print(f"Features shape: {X.shape}")
print(f"Target shape:   {y.shape}")

# Train / validation split
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Training rows:   {X_train.shape[0]}")
print(f"Validation rows: {X_valid.shape[0]}")

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
print("Model training finished.")

# Predictions
y_pred = model.predict(X_valid)

comparison = pd.DataFrame({
    "Actual SalePrice": y_valid.values,
    "Predicted SalePrice": y_pred.round(2),
})
print("\nFirst 10 predictions:")
print(comparison.head(10))

# Evaluation metrics
mae = mean_absolute_error(y_valid, y_pred)
rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
r2 = r2_score(y_valid, y_pred)

print(f"\nMAE:      {mae:.2f}")
print(f"RMSE:     {rmse:.2f}")
print(f"R2 Score: {r2:.4f}")

# Feature importance
feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_,
}).sort_values(by="Importance", ascending=False)

print("\nTop 10 important features:")
print(feature_importance.head(10))

# --- 5. Save Results ---

comparison.to_csv("outputs/validation_predictions.csv", index=False)
feature_importance.to_csv("outputs/feature_importance.csv", index=False)

model_metrics = pd.DataFrame({
    "Model": ["RandomForestRegressor"],
    "MAE": [round(mae, 2)],
    "RMSE": [round(rmse, 2)],
    "R2 Score": [round(r2, 4)],
})
model_metrics.to_csv("outputs/model_metrics.csv", index=False)

print("\nAll results saved to outputs/")
print("All plots saved to graphs/")