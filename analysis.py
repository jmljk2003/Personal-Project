import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy as np

# Load dataset
df = pd.read_csv("pop_stats.csv")

# Manually construct clean dataframe (based on known structure)

df = df.iloc[5:21]

# Select ONLY real data columns (ignore empty junk columns)
df = df.iloc[:, [0,1,2,3,4,5,6,7,8,9,10,11]]

# Rename columns (count now matches perfectly)
df.columns = [
    "State",
    "Population_000",
    "Age_0_14_pct",
    "Age_15_64_pct",
    "Age_65_plus_pct",
    "Households_Total_000",
    "Households_Urban_000",
    "Households_Rural_000",
    "Avg_HH_Size_Total",
    "Avg_HH_Size_Urban",
    "Avg_HH_Size_Rural",
    "Urbanisation_Rate"
]

# Convert numeric columns
for col in df.columns[1:]:
    df[col] = (
        df[col]
        .astype(str)
        .str.replace(",", "")
        .replace("n.a", None)
        .astype(float)
    )

print("\nCleaned Data Preview:")
print(df.head())

# =============================
# 2. CORRELATION ANALYSIS
# =============================

corr = df.drop(columns=["State"]).corr()
print("\nCorrelation Matrix (Pearson r):\n")
print(corr)

# =============================
# 3. REGRESSION PLOT HELPER
# =============================

def regression_plot(x, y, xlabel, ylabel, title, filename):
    m, b = np.polyfit(x, y, 1)
    y_pred = m * x + b
    r2 = np.corrcoef(x, y)[0, 1] ** 2

    plt.figure()
    plt.scatter(x, y)
    plt.plot(x, y_pred)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"{title} (R² = {r2:.2f})")
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.show()

# =============================
# 4. REGRESSION ANALYSIS
# =============================

regression_plot(
    df["Age_0_14_pct"],
    df["Avg_HH_Size_Total"],
    "Age 0–14 (%)",
    "Average Household Size",
    "Youth vs Household Size",
    "youth_vs_household_size_regression.png"
)

regression_plot(
    df["Age_65_plus_pct"],
    df["Avg_HH_Size_Total"],
    "Age 65+ (%)",
    "Average Household Size",
    "Elderly vs Household Size",
    "elderly_vs_household_size_regression.png"
)

regression_plot(
    df["Urbanisation_Rate"],
    df["Avg_HH_Size_Total"],
    "Urbanisation Rate (%)",
    "Average Household Size",
    "Urbanisation vs Household Size",
    "urbanisation_vs_household_size_regression.png"
)

# =============================
# 5. CLUSTERING (K-MEANS)
# =============================

features = df[
    ["Age_0_14_pct", "Age_15_64_pct", "Age_65_plus_pct", "Avg_HH_Size_Total"]
]

scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(scaled_features)

print("\nCluster Assignment by State:")
print(df[["State", "Cluster"]])

# =============================
# 6. CLUSTER SUMMARY
# =============================

print("\nCluster Summary Statistics:")
print(
    df.groupby("Cluster")[[
        "Age_0_14_pct",
        "Age_65_plus_pct",
        "Avg_HH_Size_Total",
        "Urbanisation_Rate"
    ]].mean()
)

# =============================
# 7. CLUSTER VISUALISATION + REGRESSION
# =============================

plt.figure()

colors = ["tab:blue", "tab:orange", "tab:green"]

for cluster in sorted(df["Cluster"].unique()):
    cluster_df = df[df["Cluster"] == cluster]

    x = cluster_df["Age_0_14_pct"]
    y = cluster_df["Avg_HH_Size_Total"]

    plt.scatter(x, y, color=colors[cluster], label=f"Cluster {cluster}")

    # Line of best fit PER CLUSTER
    m, b = np.polyfit(x, y, 1)
    plt.plot(x, m * x + b, color=colors[cluster])

# Plot centroids
centroids = scaler.inverse_transform(kmeans.cluster_centers_)
plt.scatter(
    centroids[:, 0],
    centroids[:, 3],
    marker="X",
    s=250,
    color="black",
    label="Centroids"
)

plt.xlabel("Age 0–14 (%)")
plt.ylabel("Average Household Size")
plt.title("Clusters with Cluster-wise Regression Lines")
plt.legend()
plt.savefig("clusters_with_regression.png", dpi=300, bbox_inches="tight")
plt.show()

# =============================
# 8. KEY CORRELATIONS (TEXT OUTPUT)
# =============================

print("\nKey Pearson Correlations:")

pairs = [
    ("Age_0_14_pct", "Avg_HH_Size_Total"),
    ("Age_65_plus_pct", "Avg_HH_Size_Total"),
    ("Urbanisation_Rate", "Avg_HH_Size_Total")
]

for x, y in pairs:
    r = df[x].corr(df[y])
    print(f"{x} vs {y}: r = {r:.3f}")