#!/usr/bin/env python3
"""
Core EDA Methods and Visualizations
CMSC 173 - Machine Learning

This script generates basic EDA visualizations including:
- Data type distributions
- Univariate analysis (histograms, box plots)
- Basic correlation analysis
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Set style for consistent plots
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

def load_sample_data():
    """Load and prepare the Titanic dataset for EDA demonstrations."""
    # Create a sample dataset similar to Titanic for demonstration
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

    # Introduce some missing values
    missing_idx = np.random.choice(n_samples, size=int(0.2 * n_samples), replace=False)
    data['age'] = pd.Series(data['age'])
    data['age'].iloc[missing_idx[:len(missing_idx)//2]] = np.nan

    df = pd.DataFrame(data)
    df['age'] = np.clip(df['age'], 0, 80)  # Realistic age bounds
    df['fare'] = np.clip(df['fare'], 0, 512)  # Realistic fare bounds

    return df

def plot_data_types_overview(df, save_path="../figures/"):
    """Generate overview of data types and structure."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Data Types and Structure Overview', fontsize=16, fontweight='bold')

    # 1. Data types bar chart
    type_counts = df.dtypes.value_counts()
    axes[0, 0].bar(range(len(type_counts)), type_counts.values,
                   color=['skyblue', 'lightcoral', 'lightgreen'])
    axes[0, 0].set_xticks(range(len(type_counts)))
    axes[0, 0].set_xticklabels([str(t) for t in type_counts.index], rotation=45)
    axes[0, 0].set_title('Data Types Distribution')
    axes[0, 0].set_ylabel('Count')

    # 2. Missing data heatmap
    missing_data = df.isnull()
    sns.heatmap(missing_data, yticklabels=False, cbar=True,
                cmap='viridis', ax=axes[0, 1])
    axes[0, 1].set_title('Missing Data Pattern')

    # 3. Dataset shape info
    info_text = f"""Dataset Shape: {df.shape[0]} rows × {df.shape[1]} columns

Numerical Features: {df.select_dtypes(include=[np.number]).shape[1]}
Categorical Features: {df.select_dtypes(include=['object']).shape[1]}

Memory Usage: {df.memory_usage(deep=True).sum() / 1024:.1f} KB
Missing Values: {df.isnull().sum().sum()} ({df.isnull().sum().sum()/df.size*100:.1f}%)"""

    axes[1, 0].text(0.1, 0.5, info_text, fontsize=11, verticalalignment='center',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    axes[1, 0].set_xlim(0, 1)
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].axis('off')
    axes[1, 0].set_title('Dataset Summary')

    # 4. Feature types classification
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

    feature_info = f"""Numerical Features:
{', '.join(numerical_cols)}

Categorical Features:
{', '.join(categorical_cols)}

Binary Features:
{', '.join([col for col in df.columns if df[col].nunique() == 2])}"""

    axes[1, 1].text(0.1, 0.5, feature_info, fontsize=10, verticalalignment='center',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].axis('off')
    axes[1, 1].set_title('Feature Classification')

    plt.tight_layout()
    plt.savefig(f"{save_path}01_data_types_overview.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_univariate_numerical(df, save_path="../figures/"):
    """Generate univariate analysis plots for numerical variables."""
    numerical_cols = ['age', 'fare']

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Univariate Analysis - Numerical Variables', fontsize=16, fontweight='bold')

    for i, col in enumerate(numerical_cols):
        # Histogram
        axes[i, 0].hist(df[col].dropna(), bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[i, 0].set_title(f'{col.capitalize()} Distribution')
        axes[i, 0].set_xlabel(col.capitalize())
        axes[i, 0].set_ylabel('Frequency')

        # Box plot
        df.boxplot(column=col, ax=axes[i, 1])
        axes[i, 1].set_title(f'{col.capitalize()} Box Plot')
        axes[i, 1].set_ylabel(col.capitalize())

        # Statistical summary text
        stats = df[col].describe()
        stats_text = f"""Count: {stats['count']:.0f}
Mean: {stats['mean']:.2f}
Std: {stats['std']:.2f}
Min: {stats['min']:.2f}
25%: {stats['25%']:.2f}
50%: {stats['50%']:.2f}
75%: {stats['75%']:.2f}
Max: {stats['max']:.2f}
Missing: {df[col].isnull().sum()}"""

        axes[i, 2].text(0.1, 0.5, stats_text, fontsize=10, verticalalignment='center',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
        axes[i, 2].set_xlim(0, 1)
        axes[i, 2].set_ylim(0, 1)
        axes[i, 2].axis('off')
        axes[i, 2].set_title(f'{col.capitalize()} Statistics')

    plt.tight_layout()
    plt.savefig(f"{save_path}02_univariate_numerical.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_univariate_categorical(df, save_path="../figures/"):
    """Generate univariate analysis plots for categorical variables."""
    categorical_cols = ['sex', 'pclass', 'embarked']

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Univariate Analysis - Categorical Variables', fontsize=16, fontweight='bold')

    for i, col in enumerate(categorical_cols):
        # Bar chart
        value_counts = df[col].value_counts()
        axes[0, i].bar(range(len(value_counts)), value_counts.values,
                      color=['lightcoral', 'lightblue', 'lightgreen'][:len(value_counts)])
        axes[0, i].set_xticks(range(len(value_counts)))
        axes[0, i].set_xticklabels(value_counts.index, rotation=45)
        axes[0, i].set_title(f'{col.capitalize()} Distribution')
        axes[0, i].set_ylabel('Count')

        # Pie chart
        axes[1, i].pie(value_counts.values, labels=value_counts.index, autopct='%1.1f%%',
                      colors=['lightcoral', 'lightblue', 'lightgreen'][:len(value_counts)])
        axes[1, i].set_title(f'{col.capitalize()} Proportions')

    plt.tight_layout()
    plt.savefig(f"{save_path}03_univariate_categorical.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_correlation_analysis(df, save_path="../figures/"):
    """Generate correlation analysis visualizations."""
    # Select numerical columns for correlation
    numerical_df = df.select_dtypes(include=[np.number])

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Correlation Analysis', fontsize=16, fontweight='bold')

    # Correlation matrix heatmap
    corr_matrix = numerical_df.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
               square=True, ax=axes[0])
    axes[0].set_title('Correlation Matrix Heatmap')

    # Correlation with target variable (survived)
    target_corr = numerical_df.corr()['survived'].sort_values(ascending=False)
    target_corr = target_corr[target_corr.index != 'survived']

    colors = ['green' if x > 0 else 'red' for x in target_corr.values]
    axes[1].barh(range(len(target_corr)), target_corr.values, color=colors, alpha=0.7)
    axes[1].set_yticks(range(len(target_corr)))
    axes[1].set_yticklabels(target_corr.index)
    axes[1].set_xlabel('Correlation with Survival')
    axes[1].set_title('Feature Correlation with Target')
    axes[1].axvline(x=0, color='black', linestyle='-', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{save_path}04_correlation_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Generate all core EDA method visualizations."""
    print("Generating Core EDA Method Visualizations...")

    # Load data
    df = load_sample_data()

    # Generate all plots
    plot_data_types_overview(df)
    plot_univariate_numerical(df)
    plot_univariate_categorical(df)
    plot_correlation_analysis(df)

    print("✓ Core methods visualizations generated successfully!")
    print(f"  - Data types overview")
    print(f"  - Univariate numerical analysis")
    print(f"  - Univariate categorical analysis")
    print(f"  - Correlation analysis")

if __name__ == "__main__":
    main()