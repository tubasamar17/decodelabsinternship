# decodelabsinternship
# Netflix Movies & TV Shows Analysis

##  Project Overview

This project is part of my **Data Science Internship at DecodeLabs (Week 1)**. The goal was to perform end-to-end data analysis on a real-world dataset — from raw data to cleaned data to meaningful insights.

**Dataset:** Netflix Movies and TV Shows (Kaggle)  
**Tasks Completed:** 3 out of 5  
**Tools Used:** Python, Pandas, Matplotlib, Seaborn

---

##  Tasks Completed

###  Task 1: Data Collection & Understanding
- Loaded the Netflix dataset (8,807 rows, 12 columns)
- Explored structure, column types, missing values, and basic statistics
- Documented dataset features including title, director, cast, country, rating, duration, etc.

###  Task 2: Data Cleaning & Preprocessing
- Handled missing values (filled with "Unknown" or mode)
- Removed duplicate rows
- Converted `date_added` to datetime format
- Cleaned the `duration` column (extracted numeric values)
- Created new columns: `duration_type` (Movie/TV Show) and `duration_value`
- Saved cleaned dataset as `netflix_cleaned.csv`

###  Task 3: Exploratory Data Analysis (EDA)
Performed statistical analysis and visualizations to discover patterns:

**Key Findings:**
- **Content Type:** ~68% Movies, ~32% TV Shows
- **Most Common Rating:** TV-MA (Mature Audience)
- **Top Content Country:** United States
- **Top Director:** Rajiv Chilaka (most titles)
- **Top Actor:** Anupam Kher (most appearances)
- **Peak Addition Year:** 2020 (over 1,200 titles added)
- **Average Movie Duration:** ~90 minutes
- **Average TV Show:** ~2 seasons

**Visualizations Included:**
- Distribution of content type (Movies vs TV Shows)
- Top 10 ratings, countries, directors, actors
- Content added per year (trend line)
- Movie duration distribution (histogram)
- TV show seasons distribution
- Monthly content addition pattern

---

##  File Structure
netflix.csv # Raw dataset
 netflix_cleaned.csv # Cleaned dataset (Task 2 output)
 task1_understanding.py # Data loading & understanding
 task2_cleaning.py # Data cleaning & preprocessing
 task3_eda_visual.py # Exploratory data analysis with visuals
 README.md # Project documentation

## How to Run

1. Install required libraries:
   ```bash
   pip install pandas matplotlib seaborn
2. Run each task file:
   python task1_understanding.py
   python task2_cleaning.py
   python task3_eda_visual.py
    
Name: Tooba Samar
Internship: DecodeLabs Data Science Intern
Week: 1
Project: Netflix Data Analysis