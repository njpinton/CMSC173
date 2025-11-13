#!/usr/bin/env python3
"""
Real-World Cross Validation Examples
CMSC 173 - Machine Learning

This script generates visualizations demonstrating cross validation
in real-world scenarios including:
- Model selection with different algorithms
- Learning curves with cross validation
- Validation curve analysis
- Bias-variance tradeoff visualization
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, validation_curve, learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification, load_breast_cancer
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set style for consistent plots
plt.style.use('default')
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 11,
    'figure.titlesize': 16
})

def create_model_comparison():
    """Create cross-validated model performance comparison."""
    # Load dataset
    data = load_breast_cancer()
    X, y = data.data, data.target

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Define models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
        'SVM (RBF)': SVC(random_state=42, gamma='scale'),
        'SVM (Linear)': SVC(random_state=42, kernel='linear')
    }

    # Perform cross-validation
    cv_scores = {}
    for name, model in models.items():
        scores = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy')
        cv_scores[name] = scores

    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Box plot of CV scores
    model_names = list(cv_scores.keys())
    scores_list = [cv_scores[name] for name in model_names]

    box_plot = ax1.boxplot(scores_list, labels=model_names, patch_artist=True)
    colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow']

    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    ax1.set_ylabel('Cross-Validation Accuracy', fontweight='bold')
    ax1.set_title('Model Comparison using 5-Fold CV', fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.tick_params(axis='x', rotation=45)

    # Mean scores with error bars
    means = [np.mean(scores) for scores in scores_list]
    stds = [np.std(scores) for scores in scores_list]

    bars = ax2.bar(model_names, means, yerr=stds, capsize=5, color=colors,
                   alpha=0.8, edgecolor='black')
    ax2.set_ylabel('Cross-Validation Accuracy', fontweight='bold')
    ax2.set_title('Mean CV Accuracy with Standard Deviation', fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.tick_params(axis='x', rotation=45)

    # Add value labels on bars
    for bar, mean, std in zip(bars, means, stds):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.005,
                f'{mean:.3f}±{std:.3f}', ha='center', va='bottom',
                fontweight='bold', fontsize=10)

    plt.tight_layout()
    plt.savefig('../figures/model_comparison_cv.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_learning_curves():
    """Create learning curves with cross validation."""
    # Create synthetic dataset
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=10,
                              n_redundant=10, n_clusters_per_class=1,
                              random_state=42)

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Define models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
        'SVM (RBF)': SVC(random_state=42, gamma='scale')
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    colors = ['blue', 'green', 'red']

    for idx, (name, model) in enumerate(models.items()):
        ax = axes[idx]

        # Generate learning curve
        train_sizes, train_scores, val_scores = learning_curve(
            model, X_scaled, y, cv=5, n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10),
            scoring='accuracy', random_state=42
        )

        # Calculate means and standard deviations
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)

        # Plot learning curves
        ax.plot(train_sizes, train_mean, 'o-', color=colors[idx],
               label='Training Score', alpha=0.8, linewidth=2)
        ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                       alpha=0.2, color=colors[idx])

        ax.plot(train_sizes, val_mean, 'o-', color='orange',
               label='Validation Score', alpha=0.8, linewidth=2)
        ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                       alpha=0.2, color='orange')

        ax.set_xlabel('Training Set Size', fontweight='bold')
        ax.set_ylabel('Accuracy Score', fontweight='bold')
        ax.set_title(f'{name}', fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0.6, 1.02)

    plt.tight_layout()
    plt.savefig('../figures/learning_curves_cv.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_validation_curves():
    """Create validation curves for hyperparameter tuning."""
    # Load dataset
    X, y = make_classification(n_samples=500, n_features=20, n_informative=10,
                              random_state=42)

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # SVM C parameter
    param_range = np.logspace(-2, 2, 10)
    train_scores, val_scores = validation_curve(
        SVC(gamma='scale', random_state=42), X_scaled, y,
        param_name='C', param_range=param_range,
        cv=5, scoring='accuracy', n_jobs=-1
    )

    ax = axes[0, 0]
    train_mean, train_std = np.mean(train_scores, axis=1), np.std(train_scores, axis=1)
    val_mean, val_std = np.mean(val_scores, axis=1), np.std(val_scores, axis=1)

    ax.semilogx(param_range, train_mean, 'o-', color='blue', label='Training Score')
    ax.fill_between(param_range, train_mean - train_std, train_mean + train_std,
                   alpha=0.2, color='blue')
    ax.semilogx(param_range, val_mean, 'o-', color='red', label='Validation Score')
    ax.fill_between(param_range, val_mean - val_std, val_mean + val_std,
                   alpha=0.2, color='red')

    ax.set_xlabel('C Parameter', fontweight='bold')
    ax.set_ylabel('Accuracy Score', fontweight='bold')
    ax.set_title('SVM: C Parameter Validation Curve', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # SVM gamma parameter
    param_range = np.logspace(-4, 0, 10)
    train_scores, val_scores = validation_curve(
        SVC(C=1.0, random_state=42), X_scaled, y,
        param_name='gamma', param_range=param_range,
        cv=5, scoring='accuracy', n_jobs=-1
    )

    ax = axes[0, 1]
    train_mean, train_std = np.mean(train_scores, axis=1), np.std(train_scores, axis=1)
    val_mean, val_std = np.mean(val_scores, axis=1), np.std(val_scores, axis=1)

    ax.semilogx(param_range, train_mean, 'o-', color='blue', label='Training Score')
    ax.fill_between(param_range, train_mean - train_std, train_mean + train_std,
                   alpha=0.2, color='blue')
    ax.semilogx(param_range, val_mean, 'o-', color='red', label='Validation Score')
    ax.fill_between(param_range, val_mean - val_std, val_mean + val_std,
                   alpha=0.2, color='red')

    ax.set_xlabel('Gamma Parameter', fontweight='bold')
    ax.set_ylabel('Accuracy Score', fontweight='bold')
    ax.set_title('SVM: Gamma Parameter Validation Curve', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Random Forest n_estimators
    param_range = range(10, 201, 20)
    train_scores, val_scores = validation_curve(
        RandomForestClassifier(random_state=42), X, y,
        param_name='n_estimators', param_range=param_range,
        cv=5, scoring='accuracy', n_jobs=-1
    )

    ax = axes[1, 0]
    train_mean, train_std = np.mean(train_scores, axis=1), np.std(train_scores, axis=1)
    val_mean, val_std = np.mean(val_scores, axis=1), np.std(val_scores, axis=1)

    ax.plot(param_range, train_mean, 'o-', color='blue', label='Training Score')
    ax.fill_between(param_range, train_mean - train_std, train_mean + train_std,
                   alpha=0.2, color='blue')
    ax.plot(param_range, val_mean, 'o-', color='red', label='Validation Score')
    ax.fill_between(param_range, val_mean - val_std, val_mean + val_std,
                   alpha=0.2, color='red')

    ax.set_xlabel('Number of Estimators', fontweight='bold')
    ax.set_ylabel('Accuracy Score', fontweight='bold')
    ax.set_title('Random Forest: n_estimators Validation Curve', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Random Forest max_depth
    param_range = range(1, 21)
    train_scores, val_scores = validation_curve(
        RandomForestClassifier(n_estimators=100, random_state=42), X, y,
        param_name='max_depth', param_range=param_range,
        cv=5, scoring='accuracy', n_jobs=-1
    )

    ax = axes[1, 1]
    train_mean, train_std = np.mean(train_scores, axis=1), np.std(train_scores, axis=1)
    val_mean, val_std = np.mean(val_scores, axis=1), np.std(val_scores, axis=1)

    ax.plot(param_range, train_mean, 'o-', color='blue', label='Training Score')
    ax.fill_between(param_range, train_mean - train_std, train_mean + train_std,
                   alpha=0.2, color='blue')
    ax.plot(param_range, val_mean, 'o-', color='red', label='Validation Score')
    ax.fill_between(param_range, val_mean - val_std, val_mean + val_std,
                   alpha=0.2, color='red')

    ax.set_xlabel('Max Depth', fontweight='bold')
    ax.set_ylabel('Accuracy Score', fontweight='bold')
    ax.set_title('Random Forest: max_depth Validation Curve', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('../figures/validation_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_bias_variance_illustration():
    """Create bias-variance tradeoff visualization."""
    np.random.seed(42)

    # Create true function
    def true_function(x):
        return 0.5 * x**2 + 0.3 * x + 0.1

    # Generate data
    n_samples = 50
    noise_level = 0.3
    x_true = np.linspace(-2, 2, 100)
    y_true = true_function(x_true)

    # Generate multiple datasets
    n_datasets = 50
    models_low_complexity = []
    models_high_complexity = []

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Low complexity model (linear)
    ax = axes[0, 0]
    predictions_low = []

    for i in range(n_datasets):
        # Generate noisy training data
        x_train = np.random.uniform(-2, 2, n_samples)
        y_train = true_function(x_train) + np.random.normal(0, noise_level, n_samples)

        # Fit linear model (low complexity)
        coeffs = np.polyfit(x_train, y_train, 1)
        y_pred = np.polyval(coeffs, x_true)
        predictions_low.append(y_pred)

        if i < 20:  # Show first 20 models
            ax.plot(x_true, y_pred, color='blue', alpha=0.3, linewidth=1)

    ax.plot(x_true, y_true, 'r-', linewidth=3, label='True Function')
    ax.plot(x_true, np.mean(predictions_low, axis=0), 'b-', linewidth=3,
           label='Average Prediction')
    ax.set_xlabel('x', fontweight='bold')
    ax.set_ylabel('y', fontweight='bold')
    ax.set_title('Low Complexity Model (High Bias, Low Variance)', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # High complexity model (degree 10 polynomial)
    ax = axes[0, 1]
    predictions_high = []

    for i in range(n_datasets):
        # Generate noisy training data
        x_train = np.random.uniform(-2, 2, n_samples)
        y_train = true_function(x_train) + np.random.normal(0, noise_level, n_samples)

        # Fit high degree polynomial (high complexity)
        try:
            coeffs = np.polyfit(x_train, y_train, min(10, len(x_train)-1))
            y_pred = np.polyval(coeffs, x_true)
            predictions_high.append(y_pred)

            if i < 20:  # Show first 20 models
                ax.plot(x_true, y_pred, color='green', alpha=0.3, linewidth=1)
        except np.linalg.LinAlgError:
            continue

    ax.plot(x_true, y_true, 'r-', linewidth=3, label='True Function')
    if predictions_high:
        ax.plot(x_true, np.mean(predictions_high, axis=0), 'g-', linewidth=3,
               label='Average Prediction')
    ax.set_xlabel('x', fontweight='bold')
    ax.set_ylabel('y', fontweight='bold')
    ax.set_title('High Complexity Model (Low Bias, High Variance)', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-2, 4)

    # Bias computation
    ax = axes[1, 0]
    if predictions_low and predictions_high:
        bias_low = (np.mean(predictions_low, axis=0) - y_true)**2
        bias_high = (np.mean(predictions_high, axis=0) - y_true)**2

        ax.plot(x_true, bias_low, 'b-', linewidth=2, label='Low Complexity')
        ax.plot(x_true, bias_high, 'g-', linewidth=2, label='High Complexity')
        ax.set_xlabel('x', fontweight='bold')
        ax.set_ylabel('Bias²', fontweight='bold')
        ax.set_title('Squared Bias', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Variance computation
    ax = axes[1, 1]
    if predictions_low and predictions_high:
        var_low = np.var(predictions_low, axis=0)
        var_high = np.var(predictions_high, axis=0)

        ax.plot(x_true, var_low, 'b-', linewidth=2, label='Low Complexity')
        ax.plot(x_true, var_high, 'g-', linewidth=2, label='High Complexity')
        ax.set_xlabel('x', fontweight='bold')
        ax.set_ylabel('Variance', fontweight='bold')
        ax.set_title('Prediction Variance', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('../figures/bias_variance_tradeoff.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    print("Generating real-world cross validation examples...")

    create_model_comparison()
    print("✓ Model comparison with CV created")

    create_learning_curves()
    print("✓ Learning curves created")

    create_validation_curves()
    print("✓ Validation curves created")

    create_bias_variance_illustration()
    print("✓ Bias-variance tradeoff illustration created")

    print("All real-world example visualizations completed!")