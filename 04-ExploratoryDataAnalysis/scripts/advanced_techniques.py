#!/usr/bin/env python3
"""
Advanced EDA Techniques and Visualizations
CMSC 173 - Machine Learning

This script generates advanced EDA visualizations including:
- Feature engineering demonstrations
- Missing data analysis
- Outlier detection methods
- Advanced visualization techniques
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, chi2
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

# Set style for consistent plots
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

def load_sample_data():
    """Load and prepare the sample dataset with intentional patterns."""
    np.random.seed(42)
    n_samples = 891

    data = {
        'age': np.random.normal(29, 14, n_samples),
        'fare': np.random.lognormal(2.5, 1.2, n_samples),
        'pclass': np.random.choice([1, 2, 3], n_samples, p=[0.24, 0.21, 0.55]),
        'sex': np.random.choice(['male', 'female'], n_samples, p=[0.65, 0.35]),
        'survived': np.random.choice([0, 1], n_samples, p=[0.62, 0.38]),
        'embarked': np.random.choice(['C', 'Q', 'S'], n_samples, p=[0.19, 0.09, 0.72]),
        'sibsp': np.random.poisson(0.5, n_samples),
        'parch': np.random.poisson(0.4, n_samples)
    }

    # Add some outliers to fare
    outlier_idx = np.random.choice(n_samples, size=20, replace=False)
    fare_array = np.array(data['fare'])
    fare_array[outlier_idx] = np.random.uniform(200, 512, 20)
    data['fare'] = fare_array

    # Introduce strategic missing values
    missing_idx = np.random.choice(n_samples, size=int(0.2 * n_samples), replace=False)
    data['age'] = pd.Series(data['age'])
    data['age'].iloc[missing_idx] = np.nan

    df = pd.DataFrame(data)
    df['age'] = np.clip(df['age'], 0, 80)
    df['fare'] = np.clip(df['fare'], 0, 512)

    return df

def plot_missing_data_analysis(df, save_path="../figures/"):
    """Generate comprehensive missing data analysis visualizations."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Missing Data Analysis', fontsize=16, fontweight='bold')

    # 1. Missing data heatmap
    missing_data = df.isnull()
    sns.heatmap(missing_data, yticklabels=False, cbar=True,
                cmap='viridis', ax=axes[0, 0])
    axes[0, 0].set_title('Missing Data Heatmap')

    # 2. Missing data bar chart
    missing_counts = df.isnull().sum()
    missing_counts = missing_counts[missing_counts > 0]
    axes[0, 1].bar(range(len(missing_counts)), missing_counts.values,
                   color='coral', alpha=0.7)
    axes[0, 1].set_xticks(range(len(missing_counts)))
    axes[0, 1].set_xticklabels(missing_counts.index, rotation=45)
    axes[0, 1].set_title('Missing Data Counts')
    axes[0, 1].set_ylabel('Number of Missing Values')

    # 3. Missing data percentage
    missing_percent = (df.isnull().sum() / len(df) * 100)
    missing_percent = missing_percent[missing_percent > 0]
    axes[1, 0].barh(range(len(missing_percent)), missing_percent.values,
                    color='lightblue', alpha=0.7)
    axes[1, 0].set_yticks(range(len(missing_percent)))
    axes[1, 0].set_yticklabels(missing_percent.index)
    axes[1, 0].set_title('Missing Data Percentage')
    axes[1, 0].set_xlabel('Percentage Missing (%)')

    # 4. Missing data correlation
    missing_df = df.isnull().astype(int)
    missing_corr = missing_df.corr()
    # Only show if there are correlations to display
    if missing_corr.shape[0] > 1:
        mask = np.triu(np.ones_like(missing_corr, dtype=bool))
        sns.heatmap(missing_corr, mask=mask, annot=True, cmap='coolwarm',
                   center=0, square=True, ax=axes[1, 1])
    axes[1, 1].set_title('Missing Data Correlation')

    plt.tight_layout()
    plt.savefig(f"{save_path}05_missing_data_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_outlier_detection(df, save_path="../figures/"):
    """Generate outlier detection visualizations."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Outlier Detection Methods', fontsize=16, fontweight='bold')

    # Focus on numerical columns with potential outliers
    numerical_cols = ['age', 'fare']

    for i, col in enumerate(numerical_cols):
        # Box plot with outliers highlighted
        df.boxplot(column=col, ax=axes[0, i])
        axes[0, i].set_title(f'{col.capitalize()} - Box Plot Method')

        # Calculate IQR outliers
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]

        # Histogram with outlier boundaries
        axes[1, i].hist(df[col].dropna(), bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[1, i].axvline(lower_bound, color='red', linestyle='--', label=f'Lower Bound: {lower_bound:.2f}')
        axes[1, i].axvline(upper_bound, color='red', linestyle='--', label=f'Upper Bound: {upper_bound:.2f}')
        axes[1, i].set_title(f'{col.capitalize()} - IQR Method ({len(outliers)} outliers)')
        axes[1, i].set_xlabel(col.capitalize())
        axes[1, i].set_ylabel('Frequency')
        axes[1, i].legend()

    plt.tight_layout()
    plt.savefig(f"{save_path}06_outlier_detection.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_feature_engineering(df, save_path="../figures/"):
    """Generate feature engineering demonstration plots."""
    # Create engineered features
    df_eng = df.copy()

    # Age groups
    df_eng['age_group'] = pd.cut(df_eng['age'], bins=[0, 18, 35, 50, 80],
                                labels=['Child', 'Young Adult', 'Adult', 'Senior'])

    # Fare categories
    df_eng['fare_category'] = pd.cut(df_eng['fare'], bins=4,
                                   labels=['Low', 'Medium', 'High', 'Very High'])

    # Family size
    df_eng['family_size'] = df_eng['sibsp'] + df_eng['parch'] + 1

    # Title extraction (simulated)
    titles = ['Mr', 'Mrs', 'Miss', 'Master', 'Dr', 'Rev']
    df_eng['title'] = np.random.choice(titles, size=len(df_eng))

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Feature Engineering Examples', fontsize=16, fontweight='bold')

    # 1. Age groups
    age_group_counts = df_eng['age_group'].value_counts()
    axes[0, 0].bar(range(len(age_group_counts)), age_group_counts.values,
                   color=['lightblue', 'lightgreen', 'orange', 'lightcoral'])
    axes[0, 0].set_xticks(range(len(age_group_counts)))
    axes[0, 0].set_xticklabels(age_group_counts.index, rotation=45)
    axes[0, 0].set_title('Age Groups (Binning)')
    axes[0, 0].set_ylabel('Count')

    # 2. Fare categories
    fare_cat_counts = df_eng['fare_category'].value_counts()
    axes[0, 1].bar(range(len(fare_cat_counts)), fare_cat_counts.values,
                   color=['lightcoral', 'orange', 'lightgreen', 'lightblue'])
    axes[0, 1].set_xticks(range(len(fare_cat_counts)))
    axes[0, 1].set_xticklabels(fare_cat_counts.index, rotation=45)
    axes[0, 1].set_title('Fare Categories (Quantile Binning)')
    axes[0, 1].set_ylabel('Count')

    # 3. Family size distribution
    family_size_counts = df_eng['family_size'].value_counts().sort_index()
    axes[1, 0].bar(family_size_counts.index, family_size_counts.values,
                   color='lightgreen', alpha=0.7)
    axes[1, 0].set_title('Family Size (Combined Features)')
    axes[1, 0].set_xlabel('Family Size')
    axes[1, 0].set_ylabel('Count')

    # 4. Title extraction
    title_counts = df_eng['title'].value_counts()
    axes[1, 1].bar(range(len(title_counts)), title_counts.values,
                   color='lightpink', alpha=0.7)
    axes[1, 1].set_xticks(range(len(title_counts)))
    axes[1, 1].set_xticklabels(title_counts.index, rotation=45)
    axes[1, 1].set_title('Title Extraction (Text Feature)')
    axes[1, 1].set_ylabel('Count')

    plt.tight_layout()
    plt.savefig(f"{save_path}07_feature_engineering.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_normalization_comparison(df, save_path="../figures/"):
    """Generate normalization comparison visualizations."""
    # Select numerical features
    features = ['age', 'fare']
    df_clean = df[features].dropna()

    # Apply different scaling methods
    standard_scaler = StandardScaler()
    minmax_scaler = MinMaxScaler()
    robust_scaler = RobustScaler()

    df_standard = pd.DataFrame(standard_scaler.fit_transform(df_clean),
                              columns=[f'{col}_standard' for col in features])
    df_minmax = pd.DataFrame(minmax_scaler.fit_transform(df_clean),
                            columns=[f'{col}_minmax' for col in features])
    df_robust = pd.DataFrame(robust_scaler.fit_transform(df_clean),
                            columns=[f'{col}_robust' for col in features])

    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    fig.suptitle('Normalization Methods Comparison', fontsize=16, fontweight='bold')

    scaling_methods = [
        (df_clean, 'Original', 'lightblue'),
        (df_standard, 'Standard (Z-score)', 'lightgreen'),
        (df_minmax, 'Min-Max', 'lightcoral'),
        (df_robust, 'Robust', 'lightyellow')
    ]

    # Plot distributions for age
    for i, (data, method, color) in enumerate(scaling_methods[:3]):
        col = 'age' if method == 'Original' else f'age_{method.split()[0].lower()}'
        if col in data.columns:
            axes[i, 0].hist(data[col], bins=30, alpha=0.7, color=color, edgecolor='black')
            axes[i, 0].set_title(f'Age - {method}')
            axes[i, 0].set_xlabel('Age')
            axes[i, 0].set_ylabel('Frequency')

    # Plot distributions for fare
    for i, (data, method, color) in enumerate(scaling_methods[:3]):
        col = 'fare' if method == 'Original' else f'fare_{method.split()[0].lower()}'
        if col in data.columns:
            axes[i, 1].hist(data[col], bins=30, alpha=0.7, color=color, edgecolor='black')
            axes[i, 1].set_title(f'Fare - {method}')
            axes[i, 1].set_xlabel('Fare')
            axes[i, 1].set_ylabel('Frequency')

    plt.tight_layout()
    plt.savefig(f"{save_path}08_normalization_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_feature_selection(df, save_path="../figures/"):
    """Generate feature selection visualization."""
    # Prepare data for feature selection
    df_clean = df.dropna()

    # Encode categorical variables
    df_encoded = pd.get_dummies(df_clean, columns=['sex', 'embarked'])

    # Separate features and target
    X = df_encoded.drop('survived', axis=1)
    y = df_encoded['survived']

    # Statistical feature selection
    selector = SelectKBest(score_func=f_classif, k=8)
    X_selected = selector.fit_transform(X, y)

    # Get feature scores
    feature_scores = pd.DataFrame({
        'feature': X.columns,
        'score': selector.scores_
    }).sort_values('score', ascending=False)

    # Random Forest feature importance
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)

    rf_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle('Feature Selection Methods', fontsize=16, fontweight='bold')

    # Statistical test scores
    top_features_stat = feature_scores.head(10)
    axes[0].barh(range(len(top_features_stat)), top_features_stat['score'],
                color='lightblue', alpha=0.7)
    axes[0].set_yticks(range(len(top_features_stat)))
    axes[0].set_yticklabels(top_features_stat['feature'])
    axes[0].set_title('F-Test Statistical Scores')
    axes[0].set_xlabel('F-Score')

    # Random Forest importance
    top_features_rf = rf_importance.head(10)
    axes[1].barh(range(len(top_features_rf)), top_features_rf['importance'],
                color='lightgreen', alpha=0.7)
    axes[1].set_yticks(range(len(top_features_rf)))
    axes[1].set_yticklabels(top_features_rf['feature'])
    axes[1].set_title('Random Forest Feature Importance')
    axes[1].set_xlabel('Importance')

    plt.tight_layout()
    plt.savefig(f"{save_path}09_feature_selection.png", dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Generate all advanced EDA technique visualizations."""
    print("Generating Advanced EDA Technique Visualizations...")

    # Load data
    df = load_sample_data()

    # Generate all plots
    plot_missing_data_analysis(df)
    plot_outlier_detection(df)
    plot_feature_engineering(df)
    plot_normalization_comparison(df)
    plot_feature_selection(df)

    print("✓ Advanced techniques visualizations generated successfully!")
    print(f"  - Missing data analysis")
    print(f"  - Outlier detection methods")
    print(f"  - Feature engineering examples")
    print(f"  - Normalization comparison")
    print(f"  - Feature selection methods")

if __name__ == "__main__":
    main()