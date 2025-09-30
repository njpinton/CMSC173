"""
Real-World Parameter Estimation Examples
========================================

This script generates visualizations for practical parameter estimation applications
in machine learning and statistics using real datasets and scenarios.

Author: CMSC 173 Machine Learning Course
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats, optimize
from sklearn.datasets import make_regression, fetch_california_housing, make_classification
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set style for consistent plots
plt.style.use('default')
sns.set_palette("husl")

def setup_plots():
    """Setup matplotlib for high-quality plots"""
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 11
    plt.rcParams['figure.dpi'] = 150

def linear_regression_estimation():
    """
    Real-world linear regression parameter estimation
    """
    setup_plots()

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

    # Generate realistic regression data
    np.random.seed(42)
    n_samples = 100
    X, y, true_coef = make_regression(n_samples=n_samples, n_features=1, noise=10,
                                      coef=True, random_state=42)
    X = X.flatten()

    # Add some realistic context
    X = X * 10 + 50  # Scale to represent something like "study hours"
    y = y * 0.5 + 75  # Scale to represent something like "test scores"

    # 1. Data and fitted line
    ax1.scatter(X, y, alpha=0.6, color='lightblue', s=30, label='Data Points')

    # Manual calculation of regression parameters
    X_mean = np.mean(X)
    y_mean = np.mean(y)

    # β₁ = Σ(xi - x̄)(yi - ȳ) / Σ(xi - x̄)²
    beta1 = np.sum((X - X_mean) * (y - y_mean)) / np.sum((X - X_mean)**2)
    beta0 = y_mean - beta1 * X_mean

    # Fitted line
    X_line = np.linspace(X.min(), X.max(), 100)
    y_pred = beta0 + beta1 * X_line

    ax1.plot(X_line, y_pred, 'r-', linewidth=2,
             label=f'Fitted: y = {beta0:.1f} + {beta1:.2f}x')

    # Show residuals for a few points
    y_fitted = beta0 + beta1 * X
    for i in range(0, len(X), 10):  # Show every 10th residual
        ax1.plot([X[i], X[i]], [y[i], y_fitted[i]], 'gray', alpha=0.5, linewidth=1)

    ax1.set_xlabel('Study Hours per Week')
    ax1.set_ylabel('Test Score')
    ax1.set_title('Linear Regression: Test Score vs Study Hours')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Likelihood surface for linear regression
    # For fixed σ², explore β₀ and β₁
    beta0_range = np.linspace(beta0 - 20, beta0 + 20, 30)
    beta1_range = np.linspace(beta1 - 1, beta1 + 1, 30)

    B0, B1 = np.meshgrid(beta0_range, beta1_range)
    log_likelihood = np.zeros_like(B0)

    # Estimate σ² from residuals
    residuals = y - (beta0 + beta1 * X)
    sigma_sq = np.mean(residuals**2)

    for i in range(len(beta0_range)):
        for j in range(len(beta1_range)):
            y_pred_ij = B0[j, i] + B1[j, i] * X
            # Normal likelihood
            log_likelihood[j, i] = -0.5 * n_samples * np.log(2 * np.pi * sigma_sq) - \
                                   0.5 * np.sum((y - y_pred_ij)**2) / sigma_sq

    contour = ax2.contour(B0, B1, log_likelihood, levels=20, colors='blue', alpha=0.6)
    ax2.clabel(contour, inline=True, fontsize=8)

    # Mark MLE point
    ax2.plot(beta0, beta1, 'ro', markersize=8, label='MLE')

    ax2.set_xlabel('β₀ (Intercept)')
    ax2.set_ylabel('β₁ (Slope)')
    ax2.set_title('Log-Likelihood Surface')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Residual analysis
    residuals = y - y_fitted

    ax3.scatter(y_fitted, residuals, alpha=0.6, color='green', s=30)
    ax3.axhline(0, color='red', linestyle='--', linewidth=2)

    # Add some statistics
    residual_std = np.std(residuals)
    ax3.axhline(2*residual_std, color='orange', linestyle=':', alpha=0.7,
                label=f'±2σ = ±{2*residual_std:.1f}')
    ax3.axhline(-2*residual_std, color='orange', linestyle=':', alpha=0.7)

    ax3.set_xlabel('Fitted Values')
    ax3.set_ylabel('Residuals')
    ax3.set_title('Residual Plot')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Bootstrap confidence intervals
    np.random.seed(42)
    n_bootstrap = 1000
    bootstrap_slopes = []
    bootstrap_intercepts = []

    for _ in range(n_bootstrap):
        # Bootstrap sample
        indices = np.random.choice(len(X), len(X), replace=True)
        X_boot = X[indices]
        y_boot = y[indices]

        # Calculate parameters
        X_boot_mean = np.mean(X_boot)
        y_boot_mean = np.mean(y_boot)

        beta1_boot = np.sum((X_boot - X_boot_mean) * (y_boot - y_boot_mean)) / \
                     np.sum((X_boot - X_boot_mean)**2)
        beta0_boot = y_boot_mean - beta1_boot * X_boot_mean

        bootstrap_slopes.append(beta1_boot)
        bootstrap_intercepts.append(beta0_boot)

    # Plot bootstrap distribution
    ax4.hist(bootstrap_slopes, bins=30, alpha=0.7, color='lightcoral',
             edgecolor='black', label='Bootstrap Distribution')

    # Confidence intervals
    ci_lower = np.percentile(bootstrap_slopes, 2.5)
    ci_upper = np.percentile(bootstrap_slopes, 97.5)

    ax4.axvline(beta1, color='red', linestyle='-', linewidth=2, label=f'Original β₁ = {beta1:.3f}')
    ax4.axvline(ci_lower, color='green', linestyle='--', linewidth=2)
    ax4.axvline(ci_upper, color='green', linestyle='--', linewidth=2)

    ax4.fill_betweenx([0, ax4.get_ylim()[1]], ci_lower, ci_upper,
                      alpha=0.2, color='green', label=f'95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]')

    ax4.set_xlabel('Bootstrap Slope Estimates')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Bootstrap Confidence Interval for Slope')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('../figures/linear_regression_estimation.png', dpi=300, bbox_inches='tight')
    plt.close()

def logistic_regression_estimation():
    """
    Logistic regression parameter estimation example
    """
    setup_plots()

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

    # Generate binary classification data
    np.random.seed(42)
    n_samples = 200
    X, y = make_classification(n_samples=n_samples, n_features=1, n_redundant=0,
                              n_informative=1, n_clusters_per_class=1,
                              random_state=42)
    X = X.flatten()

    # Scale to meaningful range
    X = X * 2 + 5  # Hours of preparation

    # 1. Data and fitted logistic curve
    ax1.scatter(X[y==0], y[y==0], alpha=0.6, color='red', s=30, label='Fail (y=0)')
    ax1.scatter(X[y==1], y[y==1], alpha=0.6, color='blue', s=30, label='Pass (y=1)')

    # Fit logistic regression
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression()
    lr.fit(X.reshape(-1, 1), y)

    # Get parameters
    beta0 = lr.intercept_[0]
    beta1 = lr.coef_[0][0]

    # Plot logistic curve
    X_curve = np.linspace(X.min(), X.max(), 200)
    probabilities = 1 / (1 + np.exp(-(beta0 + beta1 * X_curve)))

    ax1.plot(X_curve, probabilities, 'green', linewidth=2,
             label=f'Logistic: p = 1/(1+exp(-({beta0:.2f}+{beta1:.2f}x)))')

    ax1.set_xlabel('Hours of Preparation')
    ax1.set_ylabel('Probability of Passing')
    ax1.set_title('Logistic Regression: Pass/Fail Prediction')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.1, 1.1)

    # 2. Log-likelihood function (1D slice)
    beta1_range = np.linspace(-2, 2, 100)
    log_likelihood = []

    for b1 in beta1_range:
        # Fix beta0, vary beta1
        linear_combination = beta0 + b1 * X
        probabilities = 1 / (1 + np.exp(-linear_combination))

        # Avoid log(0)
        probabilities = np.clip(probabilities, 1e-15, 1-1e-15)

        # Bernoulli likelihood
        ll = np.sum(y * np.log(probabilities) + (1-y) * np.log(1-probabilities))
        log_likelihood.append(ll)

    ax2.plot(beta1_range, log_likelihood, 'b-', linewidth=2, label='Log-Likelihood')
    ax2.axvline(beta1, color='red', linestyle='--', linewidth=2,
                label=f'MLE β₁ = {beta1:.3f}')

    # Mark maximum
    max_idx = np.argmax(log_likelihood)
    ax2.plot(beta1_range[max_idx], log_likelihood[max_idx], 'go', markersize=8,
             label='Maximum')

    ax2.set_xlabel('β₁ (Slope Parameter)')
    ax2.set_ylabel('Log-Likelihood')
    ax2.set_title('Log-Likelihood Function')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Newton-Raphson convergence
    def logistic_log_likelihood(beta, X, y):
        linear_combination = beta[0] + beta[1] * X
        probabilities = 1 / (1 + np.exp(-linear_combination))
        probabilities = np.clip(probabilities, 1e-15, 1-1e-15)
        return np.sum(y * np.log(probabilities) + (1-y) * np.log(1-probabilities))

    def logistic_gradient(beta, X, y):
        linear_combination = beta[0] + beta[1] * X
        probabilities = 1 / (1 + np.exp(-linear_combination))

        grad_beta0 = np.sum(y - probabilities)
        grad_beta1 = np.sum(X * (y - probabilities))

        return np.array([grad_beta0, grad_beta1])

    # Simulate Newton-Raphson steps
    beta_current = np.array([0.0, 0.0])  # Start at origin
    steps = [beta_current.copy()]
    likelihoods = [logistic_log_likelihood(beta_current, X, y)]

    for i in range(10):
        grad = logistic_gradient(beta_current, X, y)

        # Simple gradient ascent step (simplified Newton-Raphson)
        learning_rate = 0.01
        beta_current = beta_current + learning_rate * grad

        steps.append(beta_current.copy())
        likelihoods.append(logistic_log_likelihood(beta_current, X, y))

        if np.linalg.norm(grad) < 0.1:  # Convergence
            break

    steps = np.array(steps)

    ax3.plot(range(len(likelihoods)), likelihoods, 'bo-', linewidth=2, markersize=4)
    ax3.axhline(logistic_log_likelihood([beta0, beta1], X, y), color='red',
                linestyle='--', label='True MLE')

    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Log-Likelihood')
    ax3.set_title('Newton-Raphson Convergence')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Classification performance
    # Predicted probabilities
    X_test = np.linspace(X.min(), X.max(), 100)
    probs_test = lr.predict_proba(X_test.reshape(-1, 1))[:, 1]

    ax4.plot(X_test, probs_test, 'green', linewidth=2, label='Predicted Probability')

    # Decision boundary (p = 0.5)
    decision_boundary = -beta0 / beta1
    ax4.axvline(decision_boundary, color='orange', linestyle='--', linewidth=2,
                label=f'Decision Boundary: x = {decision_boundary:.2f}')

    # Show some predictions
    X_sample = np.array([2, 4, 6, 8, 10])
    probs_sample = lr.predict_proba(X_sample.reshape(-1, 1))[:, 1]
    predictions = (probs_sample > 0.5).astype(int)

    for i, (x_val, prob, pred) in enumerate(zip(X_sample, probs_sample, predictions)):
        color = 'blue' if pred == 1 else 'red'
        ax4.plot(x_val, prob, 'o', color=color, markersize=8)
        ax4.text(x_val, prob + 0.05, f'p={prob:.2f}', ha='center', fontsize=9)

    ax4.set_xlabel('Hours of Preparation')
    ax4.set_ylabel('Probability of Passing')
    ax4.set_title('Classification Probabilities and Decision Boundary')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig('../figures/logistic_regression_estimation.png', dpi=300, bbox_inches='tight')
    plt.close()

def gaussian_mixture_estimation():
    """
    Gaussian Mixture Model parameter estimation using EM algorithm
    """
    setup_plots()

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

    # Generate mixture data
    np.random.seed(42)
    n1, n2, n3 = 100, 150, 75

    # Three components
    comp1 = np.random.normal(-2, 0.8, n1)
    comp2 = np.random.normal(1, 1.2, n2)
    comp3 = np.random.normal(4, 0.7, n3)

    data = np.concatenate([comp1, comp2, comp3])
    np.random.shuffle(data)

    # True parameters
    true_weights = [n1, n2, n3]
    true_weights = [w/sum(true_weights) for w in true_weights]
    true_means = [-2, 1, 4]
    true_stds = [0.8, 1.2, 0.7]

    # 1. Data and true mixture
    ax1.hist(data, bins=30, density=True, alpha=0.7, color='lightgray',
             edgecolor='black', label='Mixed Data')

    x_range = np.linspace(data.min(), data.max(), 200)

    # True mixture
    true_mixture = sum(w * stats.norm.pdf(x_range, mu, sigma)
                      for w, mu, sigma in zip(true_weights, true_means, true_stds))
    ax1.plot(x_range, true_mixture, 'r-', linewidth=2, label='True Mixture')

    # True components
    colors = ['blue', 'green', 'orange']
    for i, (w, mu, sigma, color) in enumerate(zip(true_weights, true_means, true_stds, colors)):
        component_pdf = w * stats.norm.pdf(x_range, mu, sigma)
        ax1.plot(x_range, component_pdf, '--', color=color, alpha=0.7,
                 label=f'Component {i+1}: w={w:.2f}, μ={mu}, σ={sigma}')

    ax1.set_xlabel('Value')
    ax1.set_ylabel('Density')
    ax1.set_title('True Gaussian Mixture Model')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. EM Algorithm simulation
    # Initialize parameters (poorly to show convergence)
    K = 3
    weights = np.array([0.33, 0.33, 0.34])
    means = np.array([0.0, 2.0, 3.0])  # Poor initialization
    stds = np.array([1.0, 1.0, 1.0], dtype=float)

    # Track convergence
    log_likelihoods = []

    for iteration in range(20):
        # E-step: compute responsibilities
        responsibilities = np.zeros((len(data), K))

        for k in range(K):
            responsibilities[:, k] = weights[k] * stats.norm.pdf(data, means[k], stds[k])

        # Normalize
        responsibilities = responsibilities / responsibilities.sum(axis=1, keepdims=True)

        # M-step: update parameters
        N_k = responsibilities.sum(axis=0)

        # Update weights
        weights = N_k / len(data)

        # Update means
        means = (responsibilities * data.reshape(-1, 1)).sum(axis=0) / N_k

        # Update standard deviations
        for k in range(K):
            variance = np.sum(responsibilities[:, k] * (data - means[k])**2) / N_k[k]
            stds[k] = np.sqrt(variance)

        # Compute log-likelihood
        mixture_probs = sum(weights[k] * stats.norm.pdf(data, means[k], stds[k])
                           for k in range(K))
        log_likelihood = np.sum(np.log(mixture_probs + 1e-15))
        log_likelihoods.append(log_likelihood)

    ax2.plot(log_likelihoods, 'bo-', linewidth=2, markersize=4)
    ax2.set_xlabel('EM Iteration')
    ax2.set_ylabel('Log-Likelihood')
    ax2.set_title('EM Algorithm Convergence')
    ax2.grid(True, alpha=0.3)

    # 3. Final fitted mixture
    ax3.hist(data, bins=30, density=True, alpha=0.7, color='lightgray',
             edgecolor='black', label='Data')

    # Fitted mixture
    fitted_mixture = sum(weights[k] * stats.norm.pdf(x_range, means[k], stds[k])
                        for k in range(K))
    ax3.plot(x_range, fitted_mixture, 'g-', linewidth=2, label='Fitted Mixture')

    # Fitted components
    for k in range(K):
        component_pdf = weights[k] * stats.norm.pdf(x_range, means[k], stds[k])
        ax3.plot(x_range, component_pdf, '--', color=colors[k], alpha=0.7,
                 label=f'Comp {k+1}: w={weights[k]:.2f}, μ={means[k]:.1f}, σ={stds[k]:.1f}')

    ax3.set_xlabel('Value')
    ax3.set_ylabel('Density')
    ax3.set_title('EM Fitted Gaussian Mixture')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Component assignments
    # Final responsibilities
    final_responsibilities = np.zeros((len(data), K))
    for k in range(K):
        final_responsibilities[:, k] = weights[k] * stats.norm.pdf(data, means[k], stds[k])
    final_responsibilities = final_responsibilities / final_responsibilities.sum(axis=1, keepdims=True)

    # Assign each point to most likely component
    assignments = np.argmax(final_responsibilities, axis=1)

    for k in range(K):
        mask = assignments == k
        ax4.scatter(data[mask], np.zeros_like(data[mask]) + k,
                   alpha=0.6, color=colors[k], s=20, label=f'Component {k+1}')

    # Show component means
    for k in range(K):
        ax4.axvline(means[k], color=colors[k], linestyle='--', alpha=0.7)
        ax4.text(means[k], k + 0.1, f'μ̂={means[k]:.1f}', ha='center', fontsize=9)

    ax4.set_xlabel('Value')
    ax4.set_ylabel('Assigned Component')
    ax4.set_title('Final Component Assignments')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_yticks([0, 1, 2])

    plt.tight_layout()
    plt.savefig('../figures/gaussian_mixture_estimation.png', dpi=300, bbox_inches='tight')
    plt.close()

def arima_estimation():
    """
    Time series parameter estimation for ARIMA models
    """
    setup_plots()

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

    # Generate AR(1) time series
    np.random.seed(42)
    n_obs = 200
    true_phi = 0.7
    true_sigma = 1.0

    # Generate AR(1): X_t = φ*X_{t-1} + ε_t
    ts = np.zeros(n_obs)
    noise = np.random.normal(0, true_sigma, n_obs)

    for t in range(1, n_obs):
        ts[t] = true_phi * ts[t-1] + noise[t]

    # 1. Time series plot
    time = np.arange(n_obs)
    ax1.plot(time, ts, 'b-', linewidth=1, alpha=0.8, label='Observed Time Series')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Value')
    ax1.set_title('AR(1) Time Series: X_t = φX_{t-1} + ε_t')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Autocorrelation function
    def autocorrelation(x, max_lag=20):
        n = len(x)
        x = x - np.mean(x)
        autocorr = np.correlate(x, x, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr = autocorr / autocorr[0]  # Normalize
        return autocorr[:max_lag+1]

    lags = np.arange(21)
    sample_acf = autocorrelation(ts, 20)

    # Theoretical ACF for AR(1): ρ(k) = φ^k
    theoretical_acf = true_phi ** lags

    ax2.bar(lags, sample_acf, alpha=0.7, color='lightblue',
            edgecolor='black', label='Sample ACF')
    ax2.plot(lags, theoretical_acf, 'ro-', markersize=4,
             label=f'Theoretical ACF: φ^k, φ={true_phi}')

    # Confidence bands (approximately ±1.96/√n)
    conf_band = 1.96 / np.sqrt(n_obs)
    ax2.axhline(conf_band, color='red', linestyle='--', alpha=0.5)
    ax2.axhline(-conf_band, color='red', linestyle='--', alpha=0.5)

    ax2.set_xlabel('Lag')
    ax2.set_ylabel('Autocorrelation')
    ax2.set_title('Autocorrelation Function')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Parameter estimation methods
    # Method of Moments: φ̂ = r₁ (lag-1 autocorrelation)
    phi_mom = sample_acf[1]

    # Maximum Likelihood (conditional)
    def ar1_log_likelihood(phi, ts):
        if abs(phi) >= 1:
            return -np.inf

        residuals = ts[1:] - phi * ts[:-1]
        sigma_sq = np.var(residuals)

        log_lik = -0.5 * (len(residuals) * np.log(2 * np.pi * sigma_sq) +
                         np.sum(residuals**2) / sigma_sq)
        return log_lik

    phi_range = np.linspace(-0.95, 0.95, 100)
    log_likelihoods = [ar1_log_likelihood(phi, ts) for phi in phi_range]

    ax3.plot(phi_range, log_likelihoods, 'b-', linewidth=2, label='Log-Likelihood')

    # Find MLE
    max_idx = np.argmax(log_likelihoods)
    phi_mle = phi_range[max_idx]

    ax3.axvline(true_phi, color='red', linestyle='--', linewidth=2,
                label=f'True φ = {true_phi}')
    ax3.axvline(phi_mom, color='green', linestyle=':', linewidth=2,
                label=f'MoM φ̂ = {phi_mom:.3f}')
    ax3.axvline(phi_mle, color='orange', linestyle='-.', linewidth=2,
                label=f'MLE φ̂ = {phi_mle:.3f}')

    ax3.set_xlabel('φ (AR Parameter)')
    ax3.set_ylabel('Log-Likelihood')
    ax3.set_title('AR(1) Parameter Estimation')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Residual analysis
    phi_est = phi_mle
    residuals = ts[1:] - phi_est * ts[:-1]

    # Q-Q plot for normality check
    from scipy import stats
    (osm, osr), (slope, intercept, r) = stats.probplot(residuals, dist="norm", plot=None)

    ax4.scatter(osm, osr, alpha=0.6, color='blue', s=20)
    ax4.plot(osm, slope * osm + intercept, 'r-', linewidth=2,
             label=f'Normal Q-Q Line (R² = {r**2:.3f})')

    ax4.set_xlabel('Theoretical Quantiles')
    ax4.set_ylabel('Sample Quantiles')
    ax4.set_title('Residual Normality Check (Q-Q Plot)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Add text with estimation results
    textstr = f'Estimation Results:\nTrue φ = {true_phi}\nMoM φ̂ = {phi_mom:.3f}\nMLE φ̂ = {phi_mle:.3f}'
    ax4.text(0.05, 0.95, textstr, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig('../figures/arima_estimation.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Generate all real-world example figures"""
    print("Generating real-world parameter estimation examples...")

    linear_regression_estimation()
    print("✓ Linear regression estimation")

    logistic_regression_estimation()
    print("✓ Logistic regression estimation")

    gaussian_mixture_estimation()
    print("✓ Gaussian mixture model estimation")

    arima_estimation()
    print("✓ ARIMA parameter estimation")

    print("Real-world examples figures completed!")

if __name__ == "__main__":
    main()