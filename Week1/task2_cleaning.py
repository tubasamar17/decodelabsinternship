
import pandas as pd
import numpy as np

df = pd.read_csv("netflixdata.csv")

print("=" * 50)
print("ORIGINAL DATASET")
print("=" * 50)
print(f"Shape: {df.shape}")
print(df.head())

print("\n" + "=" * 50)
print("MISSING VALUES - BEFORE CLEANING")
print("=" * 50)
print(df.isnull().sum())
print(f"\nTotal missing: {df.isnull().sum().sum()}")

print("\n" + "=" * 50)
print("HANDLING MISSING VALUES")
print("=" * 50)

numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    if df[col].isnull().sum() > 0:
        mean_val = df[col].mean()
        df[col].fillna(mean_val, inplace=True)
        print(f"[Numeric]  '{col}' -> filled with mean: {mean_val:.2f}")

# Categorical columns -> mode se fill karo
cat_cols = df.select_dtypes(include=['object']).columns
for col in cat_cols:
    if df[col].isnull().sum() > 0:
        mode_val = df[col].mode()[0]
        df[col].fillna(mode_val, inplace=True)
        print(f"[Category] '{col}' -> filled with mode: {mode_val}")

print("\n" + "=" * 50)
print("MISSING VALUES - AFTER CLEANING")
print("=" * 50)
print(df.isnull().sum())
print(f"\nTotal missing: {df.isnull().sum().sum()}")

print("\n" + "=" * 50)
print("DUPLICATE ROWS")
print("=" * 50)
before = df.shape[0]
print(f"Rows before: {before}")

df.drop_duplicates(inplace=True)

after = df.shape[0]
print(f"Rows after : {after}")
print(f"Duplicates removed: {before - after}")

print("\n" + "=" * 50)
print("DATA FORMATTING")
print("=" * 50)


for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].str.strip()
    df[col] = df[col].str.lower()
    print(f"Formatted (strip+lower): '{col}'")

# Date columns -> datetime format (agar koi date column ho)
date_keywords = ['date', 'year', 'time', 'added']
for col in df.columns:
    if any(keyword in col.lower() for keyword in date_keywords):
        try:
            df[col] = pd.to_datetime(df[col], errors='coerce')
            print(f"Converted to datetime : '{col}'")
        except:
            pass

df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
print(f"\nCleaned column names: {list(df.columns)}")

print("\n" + "=" * 50)
print("DATA TYPES AFTER FORMATTING")
print("=" * 50)
print(df.dtypes)

print("\n" + "=" * 50)
print("FINAL CLEANED DATASET SUMMARY")
print("=" * 50)
print(f"Final Shape     : {df.shape}")
print(f"Missing Values  : {df.isnull().sum().sum()}")
print(f"Duplicate Rows  : {df.duplicated().sum()}")
print("\nFirst 5 rows of cleaned data:")
print(df.head())

df.to_csv("netflixdata_cleaned.csv", index=False)
print("\n✓ Cleaned dataset saved as: netflixdata_cleaned.csv")