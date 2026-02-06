import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

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
# 3. VISUAL ANALYSIS (SAVED)
# =============================

# Youth vs Household Size
plt.figure()
plt.scatter(df["Age_0_14_pct"], df["Avg_HH_Size_Total"])
plt.xlabel("Age 0–14 (%)")
plt.ylabel("Average Household Size")
plt.title("Youth Population vs Household Size")
plt.savefig("youth_vs_household_size.png", dpi=300, bbox_inches="tight")
plt.show()

# Elderly vs Household Size
plt.figure()
plt.scatter(df["Age_65_plus_pct"], df["Avg_HH_Size_Total"])
plt.xlabel("Age 65+ (%)")
plt.ylabel("Average Household Size")
plt.title("Elderly Population vs Household Size")
plt.savefig("elderly_vs_household_size.png", dpi=300, bbox_inches="tight")
plt.show()

# Urbanisation vs Household Size
plt.figure()
plt.scatter(df["Urbanisation_Rate"], df["Avg_HH_Size_Total"])
plt.xlabel("Urbanisation Rate (%)")
plt.ylabel("Average Household Size")
plt.title("Urbanisation vs Household Size")
plt.savefig("urbanisation_vs_household_size.png", dpi=300, bbox_inches="tight")
plt.show()

# =============================
# 4. CLUSTERING (K-MEANS)
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
# 5. CLUSTER VISUALISATION
# =============================

plt.figure()
plt.scatter(df["Age_0_14_pct"], df["Avg_HH_Size_Total"])
plt.xlabel("Age 0–14 (%)")
plt.ylabel("Average Household Size")
plt.title("State Clusters: Youth vs Household Size")
plt.savefig("clusters_youth_vs_household_size.png", dpi=300, bbox_inches="tight")
plt.show()