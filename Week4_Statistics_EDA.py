# WEEK 4 ASSIGNMENT
# Statistics, Data Visualization and Exploratory Data Analysis (EDA)
# Project: Retail Sales Analysis
# Dataset: retail_sales.csv

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
os.makedirs("visualizations", exist_ok=True)

# Load dataset
df = pd.read_csv("retail_sales.csv")
df["Date"] = pd.to_datetime(df["Date"])

# =========================================================
# TASK 1: MEAN, MEDIAN AND MODE
# =========================================================
sales = df["Sales"]

mean_sales = sales.mean()
median_sales = sales.median()
mode_sales = sales.mode()

print("TASK 1: MEAN, MEDIAN AND MODE")
print("Mean Sales   :", round(mean_sales, 2))
print("Median Sales :", round(median_sales, 2))
print("Mode Sales   :", mode_sales.tolist())

# Mean: average value of sales.
# Median: middle value after sorting.
# Mode: most frequently occurring sales value.

# =========================================================
# TASK 2: VARIANCE AND STANDARD DEVIATION
# =========================================================
variance_sales = sales.var()
std_sales = sales.std()

print("\nTASK 2: VARIANCE AND STANDARD DEVIATION")
print("Variance:", round(variance_sales, 2))
print("Standard Deviation:", round(std_sales, 2))

# Standard deviation indicates the typical spread of sales
# around the mean. A larger value means greater variation.

# =========================================================
# TASK 3: CORRELATION AND PROBABILITY
# =========================================================
correlation = df["Sales"].corr(df["Profit"])
print("\nTASK 3: CORRELATION")
print("Correlation between Sales and Profit:", round(correlation, 3))

if correlation >= 0.7:
    print("Relationship: Strong positive")
elif correlation >= 0.3:
    print("Relationship: Moderate positive")
elif correlation <= -0.3:
    print("Relationship: Negative")
else:
    print("Relationship: Weak")

# Probability example:
# Probability that a randomly selected order is Electronics.
electronics_probability = (df["Category"] == "Electronics").mean()
print("\nProbability of Electronics order:",
      round(electronics_probability, 3))
print("Percentage:", round(electronics_probability * 100, 2), "%")

# =========================================================
# TASK 4: OUTLIER DETECTION USING IQR
# =========================================================
Q1 = sales.quantile(0.25)
Q3 = sales.quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[(sales < lower_bound) | (sales > upper_bound)]

print("\nTASK 4: OUTLIERS")
print("Q1:", Q1)
print("Q3:", Q3)
print("IQR:", IQR)
print("Lower bound:", lower_bound)
print("Upper bound:", upper_bound)
print("\nPotential outliers:")
print(outliers[["Order_ID", "Product", "Sales"]])

# Outliers may influence mean, variance, correlation and models.
# They should be investigated rather than automatically deleted.

# =========================================================
# TASK 5: MATPLOTLIB VISUALIZATIONS
# =========================================================

# 5.1 Line chart
monthly_sales = df.groupby(df["Date"].dt.to_period("M"))["Sales"].sum()
plt.figure(figsize=(10,5))
plt.plot(monthly_sales.index.astype(str), monthly_sales.values, marker="o")
plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Total Sales")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("visualizations/01_line_chart.png", dpi=200)
plt.show()

# 5.2 Bar chart
category_sales = df.groupby("Category")["Sales"].sum().sort_values(ascending=False)
plt.figure(figsize=(8,5))
plt.bar(category_sales.index, category_sales.values)
plt.title("Total Sales by Category")
plt.xlabel("Category")
plt.ylabel("Total Sales")
plt.tight_layout()
plt.savefig("visualizations/02_bar_chart.png", dpi=200)
plt.show()

# 5.3 Pie chart
region_orders = df["Region"].value_counts()
plt.figure(figsize=(7,7))
plt.pie(region_orders.values, labels=region_orders.index,
        autopct="%1.1f%%", startangle=90)
plt.title("Order Distribution by Region")
plt.tight_layout()
plt.savefig("visualizations/03_pie_chart.png", dpi=200)
plt.show()

# 5.4 Histogram
plt.figure(figsize=(9,5))
plt.hist(df["Sales"], bins=12, edgecolor="black")
plt.title("Distribution of Sales")
plt.xlabel("Sales")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("visualizations/04_histogram.png", dpi=200)
plt.show()

# 5.5 Scatter plot
plt.figure(figsize=(9,5))
plt.scatter(df["Sales"], df["Profit"], alpha=0.75)
plt.title("Sales vs Profit")
plt.xlabel("Sales")
plt.ylabel("Profit")
plt.tight_layout()
plt.savefig("visualizations/05_scatter_plot.png", dpi=200)
plt.show()

# =========================================================
# TASK 6: SEABORN VISUALIZATIONS
# =========================================================

# Count plot
plt.figure(figsize=(8,5))
sns.countplot(data=df, x="Category")
plt.title("Number of Orders by Category")
plt.xlabel("Category")
plt.ylabel("Number of Orders")
plt.tight_layout()
plt.savefig("visualizations/06_count_plot.png", dpi=200)
plt.show()

# Box plot
plt.figure(figsize=(8,5))
sns.boxplot(data=df, x="Category", y="Sales")
plt.title("Sales Distribution by Category")
plt.xlabel("Category")
plt.ylabel("Sales")
plt.tight_layout()
plt.savefig("visualizations/07_box_plot.png", dpi=200)
plt.show()

# Heatmap
numeric_cols = ["Sales", "Quantity", "Discount", "Profit", "Customer_Rating"]
corr_matrix = df[numeric_cols].corr()
plt.figure(figsize=(9,7))
sns.heatmap(corr_matrix, annot=True, fmt=".2f",
            cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("visualizations/08_heatmap.png", dpi=200)
plt.show()

# Pair plot
# Pair plot: seaborn pairplot is used as required by the assignment.
# If a local Seaborn/Matplotlib version has a color-cycle issue, use the
# fallback PairGrid block below.
try:
    pair = sns.pairplot(df[numeric_cols])
    pair.fig.suptitle("Pair Plot of Numerical Variables", y=1.02)
    pair.savefig("visualizations/09_pair_plot.png", dpi=200)
    plt.show()
except ValueError:
    pair = sns.PairGrid(df[numeric_cols])
    pair.map_offdiag(plt.scatter, s=12, alpha=0.65)
    pair.map_diag(plt.hist, bins=10, edgecolor="black")
    pair.fig.suptitle("Pair Plot of Numerical Variables", y=1.02)
    pair.savefig("visualizations/09_pair_plot.png", dpi=200)
    plt.show()

# =========================================================
# TASK 7: EDA – INSPECTION AND CLEANING
# =========================================================
print("\nTASK 7: DATA INSPECTION")
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print("\nData Types:")
print(df.dtypes)
print("\nStatistical Summary:")
print(df.describe())
print("\nMissing Values:")
print(df.isnull().sum())
print("\nDuplicate Records:", df.duplicated().sum())

# Save before-cleaning data
df.to_csv("before_cleaning.csv", index=False)

# Remove duplicates
df_clean = df.drop_duplicates().copy()

# Handle missing values if present
for col in df_clean.select_dtypes(include=np.number).columns:
    if df_clean[col].isnull().any():
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())

for col in df_clean.select_dtypes(include="object").columns:
    if df_clean[col].isnull().any():
        df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])

# Validate inconsistent values
df_clean = df_clean[
    (df_clean["Sales"] > 0) &
    (df_clean["Quantity"] > 0) &
    (df_clean["Discount"].between(0, 1)) &
    (df_clean["Profit"] >= 0) &
    (df_clean["Customer_Rating"].between(1, 5))
].copy()

df_clean.to_csv("cleaned_retail_sales.csv", index=False)

print("\nAfter Cleaning:")
print("Shape:", df_clean.shape)
print("Missing Values:")
print(df_clean.isnull().sum())
print("Duplicates:", df_clean.duplicated().sum())
print(df_clean.head())

# =========================================================
# TASK 8: CORRELATION AND INSIGHTS
# =========================================================
print("\nTASK 8: CORRELATION MATRIX")
print(df_clean[numeric_cols].corr().round(3))

top_category = df_clean.groupby("Category")["Sales"].sum().idxmax()
top_region = df_clean.groupby("Region")["Sales"].sum().idxmax()
best_month = df_clean.groupby(
    df_clean["Date"].dt.to_period("M"))["Sales"].sum().idxmax()

print("\nKEY INSIGHTS")
print("1. Highest-sales category:", top_category)
print("2. Highest-sales region:", top_region)
print("3. Highest-sales month:", best_month)
print("4. Sales and Profit have a positive relationship.")

# =========================================================
# TASK 9: BUSINESS RECOMMENDATIONS
# =========================================================
print("\nBUSINESS RECOMMENDATIONS")
print("1. Increase inventory and promotional focus on the highest-sales category.")
print("2. Strengthen marketing in the highest-sales region.")
print("3. Monitor high-value orders separately because they can strongly affect statistics and forecasting.")

df_clean.to_csv("final_cleaned_dataset.csv", index=False)
print("\nAssignment completed successfully.")
