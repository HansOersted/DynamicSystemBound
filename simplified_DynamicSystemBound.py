
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
from sklearn.neighbors import KernelDensity

# ---------- params ----------
sigma2 = 0.02
N = 100
delta = 0.05
lambda_val = np.sqrt(N)
theta_star = np.array([1.0, -1.0])
np.random.seed(0)

# ---------- loss ----------
def empirical_loss(theta):
    return np.sum((theta - theta_star)**2)

def unnormalized_log_posterior(theta):
    log_prior = -np.sum(theta**2) / (2 * sigma2)
    loss = empirical_loss(theta)
    return log_prior - lambda_val * loss

# ---------- Metropolis-Hastings ----------
def metropolis_hastings(log_prob_func, init_theta, num_samples, step_size=0.1):
    samples = []
    theta = init_theta
    current_log_prob = log_prob_func(theta)
    for _ in range(num_samples):
        proposal = theta + np.random.normal(0, step_size, size=theta.shape)
        proposal_log_prob = log_prob_func(proposal)
        accept_ratio = np.exp(proposal_log_prob - current_log_prob)
        if np.random.rand() < accept_ratio:
            theta = proposal
            current_log_prob = proposal_log_prob
        samples.append(theta.copy())
    return np.array(samples)

samples = metropolis_hastings(unnormalized_log_posterior, init_theta=np.zeros(2), num_samples=5000)
empirical_losses = np.array([empirical_loss(theta) for theta in samples])
estimated_empirical_loss = np.mean(empirical_losses)

# ---------- KDE to estimate KL divergence ----------
kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(samples)
log_rho = kde.score_samples(samples)
log_pi = stats.multivariate_normal(mean=np.zeros(2), cov=sigma2 * np.eye(2)).logpdf(samples)
kl_estimate = np.mean(log_rho - log_pi)

# ---------- G1, G2 parameters ----------
G1 = 0.0  # you can change this
G2 = 0.0  # you can change this

# ---------- PAC-Bayes Bound (Theorem 5.1 style) ----------
bound_term = (kl_estimate + np.log(1 / delta) + (lambda_val**2 / N) * G1 + (lambda_val / N) * G2) / lambda_val
pac_bayes_bound = estimated_empirical_loss + bound_term

# ---------- output ----------
print("Estimated empirical loss =", estimated_empirical_loss)
print("KL divergence =", kl_estimate)
print("PAC-Bayes Bound (Theorem 5.1 form) =", pac_bayes_bound)

# ---------- visualization ----------
plt.figure(figsize=(8, 6))
sns.kdeplot(x=samples[:, 0], y=samples[:, 1], fill=True, cmap="Blues", levels=20, thresh=0.05)
plt.scatter(theta_star[0], theta_star[1], color='red', label='Target θ = [1, -1]', zorder=5)
plt.title("Posterior Samples & Density Contours")
plt.xlabel("w")
plt.ylabel("b")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("simplified_DynamicSystemBound.png")
print("figure saved 'simplified_DynamicSystemBound.png'")