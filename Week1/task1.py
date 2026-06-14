
import pandas as pd
import numpy as np


df = pd.read_csv("netflixdata.csv")

print("=" * 50)
print("DATASET SIZE")
print("=" * 50)
print(f"Rows    : {df.shape[0]}")
print(f"Columns : {df.shape[1]}")

print("\n" + "=" * 50)
print("FIRST 5 ROWS (df.head())")
print("=" * 50)
print(df.head())

print("\n" + "=" * 50)
print("LAST 5 ROWS (df.tail())")
print("=" * 50)
print(df.tail())

print("\n" + "=" * 50)
print("COLUMN NAMES & DATA TYPES")
print("=" * 50)
print(df.dtypes)

print("\n--- Full Info ---")
df.info()

print("\n" + "=" * 50)
print("STATISTICAL SUMMARY")
print("=" * 50)
print(df.describe())

print("\n" + "=" * 50)
print("MISSING VALUES")
print("=" * 50)
missing = df.isnull().sum()
print(missing)
print(f"\nTotal missing values: {missing.sum()}")

print("\n" + "=" * 50)
print("UNIQUE VALUES PER COLUMN")
print("=" * 50)
print(df.nunique())

print("\n" + "=" * 50)
print("NUMERIC COLUMNS")
print("=" * 50)
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print(numeric_cols)

print("\n" + "=" * 50)
print("CATEGORICAL COLUMNS")
print("=" * 50)
cat_cols = df.select_dtypes(include=['object']).columns.tolist()
print(cat_cols)

print("\n" + "=" * 50)
print("DATASET DESCRIPTION")
print("=" * 50)
print(f"Dataset: Netflix Shows/Movies")
print(f"Total Records : {df.shape[0]}")
print(f"Total Features: {df.shape[1]}")
print(f"Numeric Cols  : {len(numeric_cols)}")
print(f"Categorical Cols: {len(cat_cols)}")