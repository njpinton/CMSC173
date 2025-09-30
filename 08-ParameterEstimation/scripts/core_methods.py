"""
Core Methods for Parameter Estimation
=====================================

This script generates visualizations for basic parameter estimation concepts,
including Method of Moments and Maximum Likelihood Estimation examples.

Author: CMSC 173 Machine Learning Course
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import minimize_scalar
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

def parameter_estimation_concept():
    """
    Illustrate the basic concept of parameter estimation
    """
    setup_plots()

    # Generate true data from known distribution
    np.random.seed(42)
    true_mu, true_sigma = 2.5, 1.2
    n_samples = 100
    data = np.random.normal(true_mu, true_sigma, n_samples)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left plot: Data histogram with true and estimated distributions
    ax1.hist(data, bins=20, density=True, alpha=0.7, color='lightblue',
             edgecolor='black', label='Observed Data')

    # True distribution
    x = np.linspace(data.min() - 1, data.max() + 1, 100)
    true_pdf = stats.norm.pdf(x, true_mu, true_sigma)
    ax1.plot(x, true_pdf, 'r-', linewidth=2, label=f'True: μ={true_mu}, σ={true_sigma}')

    # MoM estimates
    mom_mu = np.mean(data)
    mom_sigma = np.sqrt(np.var(data, ddof=0))
    mom_pdf = stats.norm.pdf(x, mom_mu, mom_sigma)
    ax1.plot(x, mom_pdf, 'g--', linewidth=2,
             label=f'MoM: μ̂={mom_mu:.2f}, σ̂={mom_sigma:.2f}')

    # MLE estimates (same as MoM for normal)
    mle_mu = np.mean(data)
    mle_sigma = np.sqrt(np.var(data, ddof=0))
    mle_pdf = stats.norm.pdf(x, mle_mu, mle_sigma)
    ax1.plot(x, mle_pdf, 'b:', linewidth=2,
             label=f'MLE: μ̂={mle_mu:.2f}, σ̂={mle_sigma:.2f}')

    ax1.set_xlabel('Value')
    ax1.set_ylabel('Density')
    ax1.set_title('Parameter Estimation: Normal Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right plot: Parameter space and likelihood
    mu_range = np.linspace(1.5, 3.5, 50)
    log_likelihood = []

    for mu in mu_range:
        ll = np.sum(stats.norm.logpdf(data, mu, true_sigma))
        log_likelihood.append(ll)

    ax2.plot(mu_range, log_likelihood, 'b-', linewidth=2)
    ax2.axvline(true_mu, color='red', linestyle='-', linewidth=2, label=f'True μ = {true_mu}')
    ax2.axvline(mle_mu, color='green', linestyle='--', linewidth=2, label=f'MLE μ̂ = {mle_mu:.2f}')

    ax2.set_xlabel('μ (parameter)')
    ax2.set_ylabel('Log-Likelihood')
    ax2.set_title('Likelihood Function')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('../figures/parameter_estimation_concept.png', dpi=300, bbox_inches='tight')
    plt.close()

def ml_applications():
    """
    Show parameter estimation applications in machine learning
    """
    setup_plots()

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Linear Regression
    np.random.seed(42)
    n = 50
    true_beta0, true_beta1 = 1.5, 0.8
    x = np.linspace(0, 10, n)
    y_true = true_beta0 + true_beta1 * x
    noise = np.random.normal(0, 0.5, n)
    y = y_true + noise

    # Estimate parameters
    beta1_hat = np.sum((x - np.mean(x)) * (y - np.mean(y))) / np.sum((x - np.mean(x))**2)
    beta0_hat = np.mean(y) - beta1_hat * np.mean(x)

    ax1.scatter(x, y, alpha=0.6, label='Data')
    ax1.plot(x, y_true, 'r-', linewidth=2, label=f'True: y = {true_beta0} + {true_beta1}x')
    ax1.plot(x, beta0_hat + beta1_hat * x, 'g--', linewidth=2,
             label=f'Estimated: y = {beta0_hat:.2f} + {beta1_hat:.2f}x')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Linear Regression Parameter Estimation')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Poisson Distribution
    np.random.seed(42)
    true_lambda = 3.5
    poisson_data = np.random.poisson(true_lambda, 200)
    lambda_hat = np.mean(poisson_data)

    counts = np.bincount(poisson_data)
    x_vals = np.arange(len(counts))

    ax2.bar(x_vals, counts / len(poisson_data), alpha=0.7, label='Observed')

    # Theoretical probabilities
    x_theory = np.arange(0, max(poisson_data) + 1)
    true_pmf = stats.poisson.pmf(x_theory, true_lambda)
    estimated_pmf = stats.poisson.pmf(x_theory, lambda_hat)

    ax2.plot(x_theory, true_pmf, 'ro-', markersize=4, label=f'True λ = {true_lambda}')
    ax2.plot(x_theory, estimated_pmf, 'g^-', markersize=4, label=f'Est. λ̂ = {lambda_hat:.2f}')

    ax2.set_xlabel('Value')
    ax2.set_ylabel('Probability')
    ax2.set_title('Poisson Parameter Estimation')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Exponential Distribution
    np.random.seed(42)
    true_rate = 2.0
    exp_data = np.random.exponential(1/true_rate, 100)
    rate_hat = 1 / np.mean(exp_data)

    ax3.hist(exp_data, bins=20, density=True, alpha=0.7, color='lightcoral',
             edgecolor='black', label='Data')

    x_exp = np.linspace(0, max(exp_data), 100)
    true_pdf = stats.expon.pdf(x_exp, scale=1/true_rate)
    est_pdf = stats.expon.pdf(x_exp, scale=1/rate_hat)

    ax3.plot(x_exp, true_pdf, 'r-', linewidth=2, label=f'True λ = {true_rate}')
    ax3.plot(x_exp, est_pdf, 'g--', linewidth=2, label=f'MLE λ̂ = {rate_hat:.2f}')

    ax3.set_xlabel('Value')
    ax3.set_ylabel('Density')
    ax3.set_title('Exponential Parameter Estimation')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Gaussian Mixture
    np.random.seed(42)
    n1, n2 = 80, 120
    mu1, sigma1 = -2, 1
    mu2, sigma2 = 3, 1.5

    data1 = np.random.normal(mu1, sigma1, n1)
    data2 = np.random.normal(mu2, sigma2, n2)
    mixture_data = np.concatenate([data1, data2])

    ax4.hist(mixture_data, bins=30, density=True, alpha=0.7, color='lightgreen',
             edgecolor='black', label='Mixture Data')

    x_mix = np.linspace(mixture_data.min(), mixture_data.max(), 200)

    # True mixture
    true_mixture = (n1/(n1+n2)) * stats.norm.pdf(x_mix, mu1, sigma1) + \
                   (n2/(n1+n2)) * stats.norm.pdf(x_mix, mu2, sigma2)
    ax4.plot(x_mix, true_mixture, 'r-', linewidth=2, label='True Mixture')

    # Components
    ax4.plot(x_mix, (n1/(n1+n2)) * stats.norm.pdf(x_mix, mu1, sigma1),
             'b--', alpha=0.7, label=f'Component 1: μ={mu1}')
    ax4.plot(x_mix, (n2/(n1+n2)) * stats.norm.pdf(x_mix, mu2, sigma2),
             'g--', alpha=0.7, label=f'Component 2: μ={mu2}')

    ax4.set_xlabel('Value')
    ax4.set_ylabel('Density')
    ax4.set_title('Gaussian Mixture Model')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('../figures/ml_applications.png', dpi=300, bbox_inches='tight')
    plt.close()

def estimator_properties():
    """
    Visualize properties of estimators: bias, variance, consistency
    """
    setup_plots()

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

    # Simulation parameters
    true_mu = 5.0
    true_sigma = 2.0
    sample_sizes = [10, 30, 100, 300]
    n_simulations = 1000

    # 1. Bias demonstration
    np.random.seed(42)
    biased_estimates = []
    unbiased_estimates = []

    for n in sample_sizes:
        biased_vars = []
        unbiased_vars = []

        for _ in range(n_simulations):
            sample = np.random.normal(true_mu, true_sigma, n)
            # Biased variance estimator (divide by n)
            biased_vars.append(np.var(sample, ddof=0))
            # Unbiased variance estimator (divide by n-1)
            unbiased_vars.append(np.var(sample, ddof=1))

        biased_estimates.append(np.mean(biased_vars))
        unbiased_estimates.append(np.mean(unbiased_vars))

    ax1.plot(sample_sizes, biased_estimates, 'ro-', label='Biased (÷n)')
    ax1.plot(sample_sizes, unbiased_estimates, 'go-', label='Unbiased (÷n-1)')
    ax1.axhline(true_sigma**2, color='black', linestyle='--', label=f'True σ² = {true_sigma**2}')

    ax1.set_xlabel('Sample Size')
    ax1.set_ylabel('Average Estimate')
    ax1.set_title('Bias: Variance Estimation')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Variance comparison
    mom_vars = []
    mle_vars = []

    np.random.seed(42)
    for n in sample_sizes:
        mom_estimates = []
        mle_estimates = []

        for _ in range(n_simulations):
            sample = np.random.exponential(1/true_mu, n)  # Using exponential for difference
            # MoM estimator
            mom_estimates.append(1/np.mean(sample))
            # MLE estimator (same as MoM for exponential)
            mle_estimates.append(1/np.mean(sample))

        mom_vars.append(np.var(mom_estimates))
        mle_vars.append(np.var(mle_estimates))

    ax2.plot(sample_sizes, mom_vars, 'bo-', label='Method of Moments')
    ax2.plot(sample_sizes, mle_vars, 'ro-', label='Maximum Likelihood')

    ax2.set_xlabel('Sample Size')
    ax2.set_ylabel('Variance of Estimator')
    ax2.set_title('Efficiency: Estimator Variance')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')

    # 3. Consistency demonstration
    sample_sizes_large = np.logspace(1, 3, 20).astype(int)
    mean_estimates = []
    std_estimates = []

    np.random.seed(42)
    for n in sample_sizes_large:
        estimates = []
        for _ in range(200):
            sample = np.random.normal(true_mu, true_sigma, n)
            estimates.append(np.mean(sample))

        mean_estimates.append(np.mean(estimates))
        std_estimates.append(np.std(estimates))

    ax3.errorbar(sample_sizes_large, mean_estimates, yerr=std_estimates,
                 fmt='bo-', capsize=5, alpha=0.7)
    ax3.axhline(true_mu, color='red', linestyle='--', label=f'True μ = {true_mu}')

    ax3.set_xlabel('Sample Size')
    ax3.set_ylabel('Sample Mean Estimate')
    ax3.set_title('Consistency: Convergence to True Value')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xscale('log')

    # 4. MSE decomposition
    sample_sizes_mse = [10, 20, 50, 100, 200, 500]
    mse_values = []
    bias_squared_values = []
    variance_values = []

    np.random.seed(42)
    for n in sample_sizes_mse:
        estimates = []
        for _ in range(n_simulations):
            sample = np.random.normal(true_mu, true_sigma, n)
            # Using slightly biased estimator for demonstration
            estimates.append(np.mean(sample) + 0.5/n)  # Small bias that decreases with n

        estimates = np.array(estimates)
        bias = np.mean(estimates) - true_mu
        variance = np.var(estimates)
        mse = np.mean((estimates - true_mu)**2)

        bias_squared_values.append(bias**2)
        variance_values.append(variance)
        mse_values.append(mse)

    ax4.plot(sample_sizes_mse, mse_values, 'ko-', label='MSE')
    ax4.plot(sample_sizes_mse, bias_squared_values, 'ro-', label='Bias²')
    ax4.plot(sample_sizes_mse, variance_values, 'bo-', label='Variance')

    ax4.set_xlabel('Sample Size')
    ax4.set_ylabel('Value')
    ax4.set_title('MSE Decomposition: MSE = Bias² + Variance')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale('log')

    plt.tight_layout()
    plt.savefig('../figures/estimator_properties.png', dpi=300, bbox_inches='tight')
    plt.close()

def moments_illustration():
    """
    Illustrate moments and their relationship to distribution parameters
    """
    setup_plots()

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

    # Different distributions to show moments
    x = np.linspace(-4, 8, 1000)

    # Normal distributions with different parameters
    distributions = [
        ('Normal(0,1)', stats.norm(0, 1), 'blue'),
        ('Normal(2,1)', stats.norm(2, 1), 'red'),
        ('Normal(0,2)', stats.norm(0, 2), 'green'),
    ]

    # Plot distributions
    for name, dist, color in distributions:
        y = dist.pdf(x)
        ax1.plot(x, y, color=color, linewidth=2, label=name)

        # Mark mean
        mean = dist.mean()
        ax1.axvline(mean, color=color, linestyle='--', alpha=0.7)

    ax1.set_xlabel('Value')
    ax1.set_ylabel('Density')
    ax1.set_title('Different Distributions and Their Means (1st Moment)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Show how moments change with parameters
    means = []
    variances = []
    skewness = []
    kurtosis = []

    sigma_values = np.linspace(0.5, 3, 20)

    for sigma in sigma_values:
        dist = stats.norm(2, sigma)  # Fixed mean, varying std
        means.append(dist.mean())
        variances.append(dist.var())
        skewness.append(dist.stats(moments='s'))
        kurtosis.append(dist.stats(moments='k'))

    ax2.plot(sigma_values, means, 'b-', linewidth=2, label='Mean (1st moment)')
    ax2.plot(sigma_values, variances, 'r-', linewidth=2, label='Variance (2nd central moment)')

    ax2.set_xlabel('Standard Deviation (σ)')
    ax2.set_ylabel('Moment Value')
    ax2.set_title('How Moments Change with Parameters')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Sample moments vs theoretical moments
    np.random.seed(42)
    true_mu, true_sigma = 3, 1.5
    sample_sizes = np.logspace(1, 3, 20).astype(int)

    theoretical_mean = true_mu
    theoretical_var = true_sigma**2

    sample_means = []
    sample_vars = []

    for n in sample_sizes:
        sample = np.random.normal(true_mu, true_sigma, n)
        sample_means.append(np.mean(sample))
        sample_vars.append(np.var(sample, ddof=0))

    ax3.semilogx(sample_sizes, sample_means, 'bo-', alpha=0.7, label='Sample Mean')
    ax3.axhline(theoretical_mean, color='blue', linestyle='--',
                label=f'Theoretical Mean = {theoretical_mean}')

    ax3_twin = ax3.twinx()
    ax3_twin.semilogx(sample_sizes, sample_vars, 'ro-', alpha=0.7, label='Sample Variance')
    ax3_twin.axhline(theoretical_var, color='red', linestyle='--',
                     label=f'Theoretical Variance = {theoretical_var}')

    ax3.set_xlabel('Sample Size')
    ax3.set_ylabel('Sample Mean', color='blue')
    ax3_twin.set_ylabel('Sample Variance', color='red')
    ax3.set_title('Convergence of Sample Moments')
    ax3.grid(True, alpha=0.3)

    # Add legends
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_twin.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, loc='center right')

    # Moment matching illustration
    # Generate data from gamma distribution
    np.random.seed(42)
    true_alpha, true_beta = 2, 1.5
    gamma_data = np.random.gamma(true_alpha, true_beta, 200)

    # Calculate sample moments
    m1 = np.mean(gamma_data)
    m2 = np.mean(gamma_data**2)
    sample_var = np.var(gamma_data, ddof=0)

    # Method of moments estimates
    beta_hat = sample_var / m1
    alpha_hat = m1 / beta_hat

    # Plot comparison
    x_gamma = np.linspace(0, max(gamma_data), 100)

    ax4.hist(gamma_data, bins=25, density=True, alpha=0.7, color='lightblue',
             edgecolor='black', label='Data')

    # True distribution
    true_pdf = stats.gamma.pdf(x_gamma, true_alpha, scale=true_beta)
    ax4.plot(x_gamma, true_pdf, 'r-', linewidth=2,
             label=f'True: α={true_alpha}, β={true_beta}')

    # MoM fitted distribution
    mom_pdf = stats.gamma.pdf(x_gamma, alpha_hat, scale=beta_hat)
    ax4.plot(x_gamma, mom_pdf, 'g--', linewidth=2,
             label=f'MoM: α̂={alpha_hat:.2f}, β̂={beta_hat:.2f}')

    ax4.set_xlabel('Value')
    ax4.set_ylabel('Density')
    ax4.set_title('Method of Moments: Gamma Distribution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('../figures/moments_illustration.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Generate all core method figures"""
    print("Generating core parameter estimation figures...")

    parameter_estimation_concept()
    print("✓ Parameter estimation concept")

    ml_applications()
    print("✓ ML applications")

    estimator_properties()
    print("✓ Estimator properties")

    moments_illustration()
    print("✓ Moments illustration")

    print("Core methods figures completed!")

if __name__ == "__main__":
    main()