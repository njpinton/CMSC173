"""
Advanced Parameter Estimation Techniques
========================================

This script generates visualizations for advanced parameter estimation methods,
including MLE examples, numerical optimization, and comparison techniques.

Author: CMSC 173 Machine Learning Course
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats, optimize
from scipy.special import gamma as gamma_func
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

def likelihood_concept():
    """
    Illustrate the likelihood concept and MLE
    """
    setup_plots()

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

    # Generate sample data
    np.random.seed(42)
    true_mu = 3.0
    true_sigma = 1.5
    data = np.random.normal(true_mu, true_sigma, 50)

    # 1. Likelihood vs parameter (normal distribution, known sigma)
    mu_range = np.linspace(1, 5, 100)
    likelihood = []
    log_likelihood = []

    for mu in mu_range:
        ll = np.sum(stats.norm.logpdf(data, mu, true_sigma))
        log_likelihood.append(ll)
        likelihood.append(np.exp(ll))

    # Normalize likelihood for plotting
    likelihood = np.array(likelihood)
    likelihood = likelihood / np.max(likelihood)

    ax1.plot(mu_range, likelihood, 'b-', linewidth=2, label='Likelihood')
    ax1.axvline(true_mu, color='red', linestyle='--', linewidth=2, label=f'True μ = {true_mu}')
    ax1.axvline(np.mean(data), color='green', linestyle=':', linewidth=2,
                label=f'MLE μ̂ = {np.mean(data):.2f}')

    ax1.set_xlabel('μ (Mean Parameter)')
    ax1.set_ylabel('Normalized Likelihood')
    ax1.set_title('Likelihood Function')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Log-likelihood (easier to work with)
    ax2.plot(mu_range, log_likelihood, 'b-', linewidth=2, label='Log-Likelihood')
    ax2.axvline(true_mu, color='red', linestyle='--', linewidth=2, label=f'True μ = {true_mu}')
    ax2.axvline(np.mean(data), color='green', linestyle=':', linewidth=2,
                label=f'MLE μ̂ = {np.mean(data):.2f}')

    ax2.set_xlabel('μ (Mean Parameter)')
    ax2.set_ylabel('Log-Likelihood')
    ax2.set_title('Log-Likelihood Function')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. 2D likelihood surface (mu and sigma)
    mu_vals = np.linspace(2, 4, 30)
    sigma_vals = np.linspace(0.8, 2.2, 30)
    MU, SIGMA = np.meshgrid(mu_vals, sigma_vals)

    log_lik_surface = np.zeros_like(MU)
    for i in range(len(mu_vals)):
        for j in range(len(sigma_vals)):
            log_lik_surface[j, i] = np.sum(stats.norm.logpdf(data, MU[j, i], SIGMA[j, i]))

    contour = ax3.contour(MU, SIGMA, log_lik_surface, levels=20, colors='blue', alpha=0.6)
    ax3.clabel(contour, inline=True, fontsize=8)

    # Mark true and estimated values
    ax3.plot(true_mu, true_sigma, 'ro', markersize=8, label='True Parameters')
    ax3.plot(np.mean(data), np.std(data, ddof=0), 'go', markersize=8, label='MLE Estimates')

    ax3.set_xlabel('μ')
    ax3.set_ylabel('σ')
    ax3.set_title('2D Log-Likelihood Surface')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Exponential distribution MLE
    true_lambda = 2.0
    exp_data = np.random.exponential(1/true_lambda, 100)

    lambda_range = np.linspace(0.5, 4, 100)
    exp_log_likelihood = []

    for lam in lambda_range:
        ll = np.sum(stats.expon.logpdf(exp_data, scale=1/lam))
        exp_log_likelihood.append(ll)

    ax4.plot(lambda_range, exp_log_likelihood, 'b-', linewidth=2, label='Log-Likelihood')
    ax4.axvline(true_lambda, color='red', linestyle='--', linewidth=2,
                label=f'True λ = {true_lambda}')
    ax4.axvline(1/np.mean(exp_data), color='green', linestyle=':', linewidth=2,
                label=f'MLE λ̂ = {1/np.mean(exp_data):.2f}')

    ax4.set_xlabel('λ (Rate Parameter)')
    ax4.set_ylabel('Log-Likelihood')
    ax4.set_title('MLE: Exponential Distribution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('../figures/likelihood_concept.png', dpi=300, bbox_inches='tight')
    plt.close()

def mom_poisson_example():
    """
    Detailed Method of Moments example with Poisson distribution
    """
    setup_plots()

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

    # Generate Poisson data
    np.random.seed(42)
    true_lambda = 2.8
    n_samples = 200
    data = np.random.poisson(true_lambda, n_samples)

    # Method of Moments estimate
    lambda_hat = np.mean(data)

    # 1. Data histogram with fitted distributions
    counts = np.bincount(data)
    x_vals = np.arange(len(counts))

    ax1.bar(x_vals, counts/n_samples, alpha=0.7, color='lightblue',
            edgecolor='black', label='Observed Data')

    # Theoretical and estimated PMFs
    x_theory = np.arange(0, max(data) + 1)
    true_pmf = stats.poisson.pmf(x_theory, true_lambda)
    est_pmf = stats.poisson.pmf(x_theory, lambda_hat)

    ax1.plot(x_theory, true_pmf, 'ro-', markersize=4, linewidth=2,
             label=f'True λ = {true_lambda}')
    ax1.plot(x_theory, est_pmf, 'g^-', markersize=4, linewidth=2,
             label=f'MoM λ̂ = {lambda_hat:.2f}')

    ax1.set_xlabel('Value')
    ax1.set_ylabel('Probability')
    ax1.set_title('Poisson Distribution: MoM Estimation')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Sample size effect
    sample_sizes = np.logspace(1, 3, 20).astype(int)
    estimates = []
    std_errors = []

    np.random.seed(42)
    for n in sample_sizes:
        est_list = []
        for _ in range(100):  # Multiple simulations for each sample size
            sample = np.random.poisson(true_lambda, n)
            est_list.append(np.mean(sample))

        estimates.append(np.mean(est_list))
        std_errors.append(np.std(est_list))

    ax2.errorbar(sample_sizes, estimates, yerr=std_errors, fmt='bo-',
                 capsize=5, alpha=0.7, label='MoM Estimates')
    ax2.axhline(true_lambda, color='red', linestyle='--', linewidth=2,
                label=f'True λ = {true_lambda}')

    ax2.set_xlabel('Sample Size')
    ax2.set_ylabel('Estimated λ')
    ax2.set_title('Consistency: Effect of Sample Size')
    ax2.set_xscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Distribution of estimates
    np.random.seed(42)
    n_simulations = 1000
    mom_estimates = []

    for _ in range(n_simulations):
        sample = np.random.poisson(true_lambda, 100)
        mom_estimates.append(np.mean(sample))

    ax3.hist(mom_estimates, bins=30, density=True, alpha=0.7, color='lightgreen',
             edgecolor='black', label='MoM Estimates')

    # Theoretical distribution of sample mean
    theoretical_mean = true_lambda
    theoretical_var = true_lambda / 100  # Var(X̄) = λ/n for Poisson
    x_theory = np.linspace(min(mom_estimates), max(mom_estimates), 100)
    theoretical_pdf = stats.norm.pdf(x_theory, theoretical_mean, np.sqrt(theoretical_var))

    ax3.plot(x_theory, theoretical_pdf, 'r-', linewidth=2,
             label=f'Theoretical N({theoretical_mean}, {np.sqrt(theoretical_var):.3f})')
    ax3.axvline(true_lambda, color='red', linestyle='--', alpha=0.7)

    ax3.set_xlabel('Estimated λ')
    ax3.set_ylabel('Density')
    ax3.set_title('Sampling Distribution of MoM Estimator')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Comparison with different true values
    true_lambdas = [0.5, 1.0, 2.0, 4.0, 8.0]
    sample_size = 50

    x_max = 15
    x_vals = np.arange(0, x_max + 1)

    for i, true_lam in enumerate(true_lambdas):
        np.random.seed(42 + i)
        sample = np.random.poisson(true_lam, sample_size)
        est_lam = np.mean(sample)

        # Plot theoretical vs estimated
        true_pmf = stats.poisson.pmf(x_vals, true_lam)
        est_pmf = stats.poisson.pmf(x_vals, est_lam)

        ax4.plot(x_vals, true_pmf, '-', alpha=0.7, linewidth=1,
                 color=f'C{i}', label=f'True λ={true_lam}')
        ax4.plot(x_vals, est_pmf, '--', alpha=0.7, linewidth=1,
                 color=f'C{i}', label=f'Est λ̂={est_lam:.1f}')

    ax4.set_xlabel('Value')
    ax4.set_ylabel('Probability')
    ax4.set_title('MoM Performance Across Different λ Values')
    ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('../figures/mom_poisson_example.png', dpi=300, bbox_inches='tight')
    plt.close()

def mle_exponential():
    """
    Detailed MLE example with exponential distribution
    """
    setup_plots()

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

    # Generate exponential data
    np.random.seed(42)
    true_lambda = 1.5
    n_samples = 100
    data = np.random.exponential(1/true_lambda, n_samples)

    # MLE estimate
    lambda_mle = 1 / np.mean(data)

    # 1. Data and fitted distributions
    ax1.hist(data, bins=20, density=True, alpha=0.7, color='lightcoral',
             edgecolor='black', label=f'Data (n={n_samples})')

    x_range = np.linspace(0, max(data), 200)
    true_pdf = stats.expon.pdf(x_range, scale=1/true_lambda)
    mle_pdf = stats.expon.pdf(x_range, scale=1/lambda_mle)

    ax1.plot(x_range, true_pdf, 'r-', linewidth=2, label=f'True λ = {true_lambda}')
    ax1.plot(x_range, mle_pdf, 'g--', linewidth=2, label=f'MLE λ̂ = {lambda_mle:.2f}')

    ax1.set_xlabel('Value')
    ax1.set_ylabel('Density')
    ax1.set_title('Exponential Distribution: MLE')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Log-likelihood function
    lambda_range = np.linspace(0.5, 3, 100)
    log_likelihood = []

    for lam in lambda_range:
        ll = np.sum(stats.expon.logpdf(data, scale=1/lam))
        log_likelihood.append(ll)

    ax2.plot(lambda_range, log_likelihood, 'b-', linewidth=2, label='Log-Likelihood')
    ax2.axvline(true_lambda, color='red', linestyle='--', linewidth=2,
                label=f'True λ = {true_lambda}')
    ax2.axvline(lambda_mle, color='green', linestyle=':', linewidth=2,
                label=f'MLE λ̂ = {lambda_mle:.2f}')

    # Mark maximum
    max_idx = np.argmax(log_likelihood)
    ax2.plot(lambda_range[max_idx], log_likelihood[max_idx], 'go', markersize=8,
             label='Maximum')

    ax2.set_xlabel('λ (Rate Parameter)')
    ax2.set_ylabel('Log-Likelihood')
    ax2.set_title('Log-Likelihood Function')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Score function (derivative of log-likelihood)
    # For exponential: d/dλ log L(λ) = n/λ - Σxi
    score_values = []
    for lam in lambda_range:
        score = len(data)/lam - np.sum(data)
        score_values.append(score)

    ax3.plot(lambda_range, score_values, 'purple', linewidth=2, label='Score Function')
    ax3.axhline(0, color='black', linestyle='-', alpha=0.5)
    ax3.axvline(lambda_mle, color='green', linestyle=':', linewidth=2,
                label=f'Root: λ̂ = {lambda_mle:.2f}')

    ax3.set_xlabel('λ')
    ax3.set_ylabel('Score S(λ)')
    ax3.set_title('Score Function: S(λ) = dℓ/dλ')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Sampling distribution of MLE
    np.random.seed(42)
    n_simulations = 1000
    mle_estimates = []

    for _ in range(n_simulations):
        sample = np.random.exponential(1/true_lambda, n_samples)
        mle_estimates.append(1/np.mean(sample))

    ax4.hist(mle_estimates, bins=30, density=True, alpha=0.7, color='lightgreen',
             edgecolor='black', label='MLE Estimates')

    # Theoretical distribution (approximately normal for large n)
    # For exponential MLE: asymptotically normal with variance λ²/n
    theoretical_mean = true_lambda
    theoretical_var = (true_lambda**2) / n_samples
    x_theory = np.linspace(min(mle_estimates), max(mle_estimates), 100)
    theoretical_pdf = stats.norm.pdf(x_theory, theoretical_mean, np.sqrt(theoretical_var))

    ax4.plot(x_theory, theoretical_pdf, 'r-', linewidth=2,
             label=f'Asymptotic N({theoretical_mean}, {np.sqrt(theoretical_var):.3f})')
    ax4.axvline(true_lambda, color='red', linestyle='--', alpha=0.7)

    ax4.set_xlabel('Estimated λ')
    ax4.set_ylabel('Density')
    ax4.set_title('Sampling Distribution of MLE')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('../figures/mle_exponential.png', dpi=300, bbox_inches='tight')
    plt.close()

def numerical_mle():
    """
    Demonstrate numerical methods for MLE when no closed form exists
    """
    setup_plots()

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

    # Example: Weibull distribution (requires numerical optimization)
    np.random.seed(42)
    true_shape = 2.5
    true_scale = 3.0
    n_samples = 100
    # Use scipy stats to generate data for consistency
    data = stats.weibull_min.rvs(true_shape, scale=true_scale, size=n_samples)

    # 1. Data and true distribution
    ax1.hist(data, bins=20, density=True, alpha=0.7, color='lightblue',
             edgecolor='black', label=f'Data (n={n_samples})')

    x_range = np.linspace(0, max(data), 200)
    # Weibull PDF: (k/λ)(x/λ)^(k-1) * exp(-(x/λ)^k)
    true_pdf = stats.weibull_min.pdf(x_range, true_shape, scale=true_scale)
    ax1.plot(x_range, true_pdf, 'r-', linewidth=2,
             label=f'True: k={true_shape}, λ={true_scale}')

    ax1.set_xlabel('Value')
    ax1.set_ylabel('Density')
    ax1.set_title('Weibull Distribution Data')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. 2D log-likelihood surface
    k_vals = np.linspace(1.5, 3.5, 30)
    lambda_vals = np.linspace(2.0, 4.0, 30)
    K, LAM = np.meshgrid(k_vals, lambda_vals)

    log_lik_surface = np.zeros_like(K)
    for i in range(len(k_vals)):
        for j in range(len(lambda_vals)):
            try:
                log_lik_surface[j, i] = np.sum(stats.weibull_min.logpdf(data, K[j, i], scale=LAM[j, i]))
            except:
                log_lik_surface[j, i] = -np.inf

    # Mask invalid values
    log_lik_surface = np.ma.masked_invalid(log_lik_surface)

    contour = ax2.contour(K, LAM, log_lik_surface, levels=20, colors='blue', alpha=0.6)
    ax2.clabel(contour, inline=True, fontsize=8)

    # Find MLE numerically
    def neg_log_likelihood(params):
        k, lam = params
        if k <= 0 or lam <= 0:
            return np.inf
        try:
            return -np.sum(stats.weibull_min.logpdf(data, k, scale=lam))
        except:
            return np.inf

    # Initial guess
    initial_guess = [2.0, 2.5]
    result = optimize.minimize(neg_log_likelihood, initial_guess, method='Nelder-Mead')
    k_mle, lam_mle = result.x

    ax2.plot(true_shape, true_scale, 'ro', markersize=8, label='True Parameters')
    ax2.plot(k_mle, lam_mle, 'go', markersize=8, label=f'MLE: k̂={k_mle:.2f}, λ̂={lam_mle:.2f}')

    ax2.set_xlabel('Shape (k)')
    ax2.set_ylabel('Scale (λ)')
    ax2.set_title('2D Log-Likelihood Surface')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Optimization path (simplified 1D example)
    # Fix scale, optimize shape
    shape_range = np.linspace(1.5, 3.5, 100)
    log_lik_1d = []

    for k in shape_range:
        try:
            ll = np.sum(stats.weibull_min.logpdf(data, k, scale=true_scale))
            log_lik_1d.append(ll)
        except:
            log_lik_1d.append(-np.inf)

    ax3.plot(shape_range, log_lik_1d, 'b-', linewidth=2, label='Log-Likelihood')

    # Show optimization steps (Newton-Raphson simulation)
    def shape_log_likelihood(k):
        try:
            return np.sum(stats.weibull_min.logpdf(data, k, scale=true_scale))
        except:
            return -np.inf

    # Simple gradient ascent simulation
    k_current = 2.0
    learning_rate = 0.1
    steps = []
    values = []

    for i in range(20):
        # Numerical gradient
        epsilon = 0.01
        grad = (shape_log_likelihood(k_current + epsilon) - shape_log_likelihood(k_current - epsilon)) / (2 * epsilon)

        steps.append(k_current)
        values.append(shape_log_likelihood(k_current))

        k_current += learning_rate * grad

        if abs(grad) < 0.001:  # Convergence
            break

    ax3.plot(steps, values, 'ro-', markersize=4, alpha=0.7, label='Optimization Steps')
    ax3.axvline(true_shape, color='red', linestyle='--', label=f'True k = {true_shape}')

    ax3.set_xlabel('Shape Parameter (k)')
    ax3.set_ylabel('Log-Likelihood')
    ax3.set_title('Numerical Optimization Path')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Compare optimization methods
    methods = ['Nelder-Mead', 'BFGS', 'Powell']
    results = {}

    for method in methods:
        try:
            res = optimize.minimize(neg_log_likelihood, initial_guess, method=method)
            results[method] = {
                'params': res.x,
                'likelihood': -res.fun,
                'iterations': res.nit if hasattr(res, 'nit') else res.nfev,
                'success': res.success
            }
        except:
            results[method] = None

    # Plot fitted distributions
    for i, (method, result) in enumerate(results.items()):
        if result and result['success']:
            k_est, lam_est = result['params']
            fitted_pdf = stats.weibull_min.pdf(x_range, k_est, scale=lam_est)
            ax4.plot(x_range, fitted_pdf, linestyle='--', linewidth=2,
                     label=f'{method}: k̂={k_est:.2f}, λ̂={lam_est:.2f}')

    # Original data and true distribution
    ax4.hist(data, bins=20, density=True, alpha=0.5, color='lightgray',
             edgecolor='black')
    ax4.plot(x_range, true_pdf, 'r-', linewidth=3, alpha=0.8,
             label=f'True: k={true_shape}, λ={true_scale}')

    ax4.set_xlabel('Value')
    ax4.set_ylabel('Density')
    ax4.set_title('Comparison of Optimization Methods')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('../figures/numerical_mle.png', dpi=300, bbox_inches='tight')
    plt.close()

def mle_properties():
    """
    Demonstrate theoretical properties of MLE estimators
    """
    setup_plots()

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Consistency demonstration
    true_mu = 3.0
    true_sigma = 1.5
    sample_sizes = np.logspace(1, 3, 20).astype(int)

    np.random.seed(42)
    mean_estimates = []
    std_estimates = []

    for n in sample_sizes:
        estimates = []
        for _ in range(200):
            sample = np.random.normal(true_mu, true_sigma, n)
            estimates.append(np.mean(sample))  # MLE for normal mean

        mean_estimates.append(np.mean(estimates))
        std_estimates.append(np.std(estimates))

    ax1.errorbar(sample_sizes, mean_estimates, yerr=std_estimates,
                 fmt='bo-', capsize=5, alpha=0.7, label='MLE Estimates')
    ax1.axhline(true_mu, color='red', linestyle='--', linewidth=2,
                label=f'True μ = {true_mu}')

    # Theoretical standard error
    theoretical_se = true_sigma / np.sqrt(sample_sizes)
    ax1.fill_between(sample_sizes, true_mu - 1.96*theoretical_se, true_mu + 1.96*theoretical_se,
                     alpha=0.2, color='red', label='95% Theoretical CI')

    ax1.set_xlabel('Sample Size')
    ax1.set_ylabel('Estimated μ')
    ax1.set_title('Consistency: MLE Convergence')
    ax1.set_xscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Asymptotic normality
    n_large = 100
    n_simulations = 1000

    np.random.seed(42)
    mle_estimates = []

    for _ in range(n_simulations):
        sample = np.random.normal(true_mu, true_sigma, n_large)
        mle_estimates.append(np.mean(sample))

    # Standardized estimates
    standardized = (np.array(mle_estimates) - true_mu) / (true_sigma / np.sqrt(n_large))

    ax2.hist(standardized, bins=30, density=True, alpha=0.7, color='lightgreen',
             edgecolor='black', label='Standardized MLE')

    # Standard normal overlay
    x_norm = np.linspace(-4, 4, 100)
    ax2.plot(x_norm, stats.norm.pdf(x_norm), 'r-', linewidth=2,
             label='Standard Normal')

    ax2.set_xlabel('Standardized Estimate')
    ax2.set_ylabel('Density')
    ax2.set_title('Asymptotic Normality')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Efficiency (Cramér-Rao bound)
    # Compare MLE with other estimators for exponential distribution
    true_lambda = 2.0
    sample_size = 50
    n_simulations = 1000

    np.random.seed(42)
    mle_estimates = []
    mom_estimates = []
    median_estimates = []

    for _ in range(n_simulations):
        sample = np.random.exponential(1/true_lambda, sample_size)

        # MLE
        mle_estimates.append(1/np.mean(sample))

        # Method of Moments (same as MLE for exponential)
        mom_estimates.append(1/np.mean(sample))

        # Median-based estimator
        median_estimates.append(np.log(2)/np.median(sample))

    estimators = {
        'MLE': mle_estimates,
        'Median-based': median_estimates
    }

    positions = [1, 2]
    bp = ax3.boxplot([estimators[name] for name in estimators.keys()],
                     positions=positions, patch_artist=True)

    colors = ['lightblue', 'lightcoral']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    ax3.axhline(true_lambda, color='red', linestyle='--', linewidth=2,
                label=f'True λ = {true_lambda}')

    ax3.set_xticklabels(estimators.keys())
    ax3.set_ylabel('Estimated λ')
    ax3.set_title('Efficiency Comparison')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Add variance comparison
    for i, (name, estimates) in enumerate(estimators.items()):
        var = np.var(estimates)
        ax3.text(positions[i], max(estimates), f'Var={var:.3f}',
                 ha='center', va='bottom', fontsize=10)

    # 4. Invariance property
    # If θ̂ is MLE of θ, then g(θ̂) is MLE of g(θ)
    np.random.seed(42)
    scale_param = 2.5
    n_samples = 100

    # Generate exponential data
    exp_data = np.random.exponential(scale_param, n_samples)

    # MLE of scale parameter
    scale_mle = np.mean(exp_data)

    # Rate parameter = 1/scale
    rate_true = 1/scale_param
    rate_mle = 1/scale_mle  # Invariance: MLE of g(θ) = g(MLE of θ)

    # Verify by direct optimization of rate parameter
    def neg_log_lik_rate(rate):
        if rate <= 0:
            return np.inf
        return -np.sum(stats.expon.logpdf(exp_data, scale=1/rate))

    rate_range = np.linspace(0.1, 1.0, 100)
    log_lik_rate = [-neg_log_lik_rate(r) for r in rate_range]

    ax4.plot(rate_range, log_lik_rate, 'b-', linewidth=2, label='Log-Likelihood')
    ax4.axvline(rate_true, color='red', linestyle='--', linewidth=2,
                label=f'True rate = {rate_true:.3f}')
    ax4.axvline(rate_mle, color='green', linestyle=':', linewidth=2,
                label=f'MLE rate = {rate_mle:.3f}')

    # Direct MLE
    direct_mle = rate_range[np.argmax(log_lik_rate)]
    ax4.axvline(direct_mle, color='orange', linestyle='-.', linewidth=2,
                label=f'Direct MLE = {direct_mle:.3f}')

    ax4.set_xlabel('Rate Parameter')
    ax4.set_ylabel('Log-Likelihood')
    ax4.set_title('Invariance Property: g(θ̂) = ĝ(θ)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('../figures/mle_properties.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Generate all advanced technique figures"""
    print("Generating advanced parameter estimation figures...")

    likelihood_concept()
    print("✓ Likelihood concept")

    mom_poisson_example()
    print("✓ Method of Moments Poisson example")

    mle_exponential()
    print("✓ MLE exponential example")

    numerical_mle()
    print("✓ Numerical MLE methods")

    mle_properties()
    print("✓ MLE theoretical properties")

    print("Advanced techniques figures completed!")

if __name__ == "__main__":
    main()