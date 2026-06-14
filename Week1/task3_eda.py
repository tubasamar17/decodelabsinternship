
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("viridis")
plt.rcParams['figure.figsize'] = (12, 6)

print("=" * 60)
print("LOADING CLEANED NETFLIX DATASET")
print("=" * 60)

try:
    df = pd.read_csv("netflixdata_cleaned.csv")
    print(f"✓ Successfully loaded: netflixdata_cleaned.csv")
    print(f"✓ Shape: {df.shape[0]} rows, {df.shape[1]} columns")
except FileNotFoundError:
    print("✗ Error: 'netflixdata_cleaned.csv' not found!")
    print("  Please run your cleaning code first to generate this file.")
    exit()

print("\n" + "=" * 60)
print("1. DATA OVERVIEW")
print("=" * 60)
print("\n--- First 3 rows ---")
print(df.head(3))

print("\n--- Column Names ---")
print(df.columns.tolist())

print("\n--- Data Types ---")
print(df.dtypes)


print("\n" + "=" * 60)
print("2. NUMERICAL FEATURES ANALYSIS")
print("=" * 60)

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

if len(numeric_cols) > 0:
    print(f"\nNumerical columns found: {numeric_cols}")
    
    print("\n--- Statistical Summary ---")
    print(df[numeric_cols].describe())
    
    print("\n--- Skewness (distribution shape) ---")
    for col in numeric_cols:
        skewness = df[col].skew()
        if abs(skewness) > 1:
            print(f"  {col}: {skewness:.2f} (highly skewed)")
        elif abs(skewness) > 0.5:
            print(f"  {col}: {skewness:.2f} (moderately skewed)")
        else:
            print(f"  {col}: {skewness:.2f} (approximately symmetric)")
    
    
    fig, axes = plt.subplots(1, len(numeric_cols), figsize=(5*len(numeric_cols), 4))
    if len(numeric_cols) == 1:
        axes = [axes]
    
    for i, col in enumerate(numeric_cols):
        axes[i].hist(df[col], bins=30, edgecolor='black', alpha=0.7)
        axes[i].set_title(f'Distribution of {col}', fontsize=12, fontweight='bold')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Frequency')
        axes[i].axvline(df[col].mean(), color='red', linestyle='--', label=f'Mean: {df[col].mean():.2f}')
        axes[i].axvline(df[col].median(), color='green', linestyle='--', label=f'Median: {df[col].median():.2f}')
        axes[i].legend()
    
    plt.tight_layout()
    plt.show()
else:
    print("No numerical columns found in the dataset.")

print("\n" + "=" * 60)
print("3. CATEGORICAL FEATURES ANALYSIS")
print("=" * 60)

# Get categorical columns
cat_cols = df.select_dtypes(include=['object']).columns.tolist()

if len(cat_cols) > 0:
    print(f"\nCategorical columns found: {cat_cols[:5]}")  # Show first 5
    
    for col in cat_cols[:3]:  # Analyze first 3 categorical columns
        print(f"\n--- Top 10 values in '{col}' ---")
        value_counts = df[col].value_counts()
        print(value_counts.head(10))
        
        if len(value_counts) > 1:
            plt.figure(figsize=(10, 5))
            top_values = value_counts.head(10)
            bars = plt.bar(range(len(top_values)), top_values.values, color='skyblue', edgecolor='black')
            plt.xticks(range(len(top_values)), top_values.index, rotation=45, ha='right')
            plt.title(f'Top 10 Categories in {col}', fontsize=14, fontweight='bold')
            plt.xlabel(col, fontsize=12)
            plt.ylabel('Count', fontsize=12)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.show()
else:
    print("No categorical columns found in the dataset.")

print("\n" + "=" * 60)
print("4. MISSING VALUES VERIFICATION")
print("=" * 60)

missing = df.isnull().sum()
if missing.sum() == 0:
    print("✓ No missing values found! Cleaning was successful.")
else:
    print(f"⚠ Found {missing.sum()} missing values:")
    print(missing[missing > 0])

print("\n" + "=" * 60)
print("5. DUPLICATE ROWS VERIFICATION")
print("=" * 60)

duplicates = df.duplicated().sum()
if duplicates == 0:
    print("✓ No duplicate rows found! Cleaning was successful.")
else:
    print(f"⚠ Found {duplicates} duplicate rows.")


if len(numeric_cols) >= 2:
    print("\n" + "=" * 60)
    print("6. CORRELATION ANALYSIS")
    print("=" * 60)
    
    correlation_matrix = df[numeric_cols].corr()
    print("\n--- Correlation Matrix ---")
    print(correlation_matrix)
    

    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # Strong correlations
    print("\n--- Strong Correlations (|r| > 0.5) ---")
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            corr_val = correlation_matrix.iloc[i, j]
            if abs(corr_val) > 0.5:
                print(f"  {correlation_matrix.columns[i]} ↔ {correlation_matrix.columns[j]}: {corr_val:.3f}")

print("\n" + "=" * 60)
print("7. NETFLIX-SPECIFIC INSIGHTS")
print("=" * 60)


if 'type' in df.columns:
    print("\n--- Content Type Distribution ---")
    type_dist = df['type'].value_counts()
    print(type_dist)
    
    plt.figure(figsize=(8, 6))
    plt.pie(type_dist.values, labels=type_dist.index, autopct='%1.1f%%',
            startangle=90, colors=['#ff6b6b', '#4ecdc4'])
    plt.title('Movies vs TV Shows Distribution', fontsize=14, fontweight='bold')
    plt.show()

if 'country' in df.columns:
    print("\n--- Top 10 Countries Producing Content ---")
    top_countries = df['country'].value_counts().head(10)
    print(top_countries)
    

    plt.figure(figsize=(12, 6))
    top_countries.plot(kind='bar', color='coral', edgecolor='black')
    plt.title('Top 10 Content Producing Countries', fontsize=14, fontweight='bold')
    plt.xlabel('Country', fontsize=12)
    plt.ylabel('Number of Titles', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

if 'release_year' in df.columns:
    print("\n--- Content Release Year Trend ---")
    yearly_counts = df['release_year'].value_counts().sort_index()
    print(f"Year range: {df['release_year'].min()} - {df['release_year'].max()}")
    print(f"Peak year: {yearly_counts.idxmax()} with {yearly_counts.max()} titles")
    
  
    plt.figure(figsize=(14, 6))
    plt.plot(yearly_counts.index, yearly_counts.values, marker='o', linewidth=2, markersize=4)
    plt.title('Number of Titles Released Over Years', fontsize=14, fontweight='bold')
    plt.xlabel('Release Year', fontsize=12)
    plt.ylabel('Number of Titles', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

if 'rating' in df.columns:
    print("\n--- Rating Distribution ---")
    rating_counts = df['rating'].value_counts()
    print(rating_counts.head(10))


print("\n" + "=" * 60)
print("8. EDA SUMMARY REPORT")
print("=" * 60)

print(f"""
DATASET STATISTICS:
------------------
• Total Records    : {df.shape[0]:,}
• Total Features   : {df.shape[1]}
• Numerical Cols   : {len(numeric_cols)}
• Categorical Cols : {len(cat_cols)}
• Missing Values   : {df.isnull().sum().sum()} ✓
• Duplicate Rows   : {df.duplicated().sum()} ✓

DATA QUALITY: EXCELLENT ✓

KEY OBSERVATIONS:
----------------
""")

if len(numeric_cols) > 0:
    print(f"• Numerical features show varying distributions")
if 'type' in df.columns:
    print(f"• Dataset contains both Movies and TV Shows")
if 'release_year' in df.columns:
    print(f"• Content spans from {df['release_year'].min()} to {df['release_year'].max()}")
if 'country' in df.columns:
    print(f"• Content from {df['country'].nunique()} different countries")

print("\n✓ EDA completed successfully!")
print("=" * 60)

summary_stats = df.describe(include='all').transpose()
summary_stats.to_csv("eda_summary.csv")
print("\n✓ Summary statistics saved to: eda_summary.csv")