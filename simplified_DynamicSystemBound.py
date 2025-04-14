import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
from sklearn.neighbors import KernelDensity

# ---------- params ----------
sigma2 = 0.02
lambda_N = np.sqrt(100)
theta_star = np.array([1.0, -1.0])
np.random.seed(0)

# ---------- loss ----------
def empirical_loss(theta):
    return np.sum((theta - theta_star)**2)

def unnormalized_log_posterior(theta):
    log_prior = -np.sum(theta**2) / (2 * sigma2)
    loss = empirical_loss(theta)
    return log_prior - lambda_N * loss

# ---------- Metropolis-Hastings (MCMC design) ----------
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

# ---------- MCMC ----------
samples = metropolis_hastings(unnormalized_log_posterior, init_theta=np.zeros(2), num_samples=5000)
estimated_empirical_loss = np.mean([empirical_loss(theta) for theta in samples])

# ---------- kl_estimate ----------
kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(samples)
grid_x, grid_y = np.meshgrid(np.linspace(-2, 4, 100), np.linspace(-4, 2, 100))
grid_points = np.vstack([grid_x.ravel(), grid_y.ravel()]).T

log_rho = kde.score_samples(grid_points)
rho = np.exp(log_rho)
log_pi = stats.multivariate_normal(mean=[0, 0], cov=sigma2 * np.eye(2)).logpdf(grid_points)
pi = np.exp(log_pi)

eps = 1e-12
rho = np.clip(rho, eps, None)
pi = np.clip(pi, eps, None)
kl_estimate = np.sum(rho * (log_rho - log_pi)) * (6 / 100)**2 

# ---------- PAC-Bayes Bound with and without Psi ----------
N = 100
delta = 0.05
log_term = np.log(2 * np.sqrt(N) / delta)
pac_bayes_bound = estimated_empirical_loss + np.sqrt((kl_estimate + log_term) / (2 * N))

# Add Psi correction term
G1 = 0.5
G2 = 1.0
lambda_val = np.sqrt(N)
psi_hat = (lambda_val**2 * G1) / N + (lambda_val * G2) / N
pac_bayes_bound_with_psi = pac_bayes_bound + psi_hat

# ---------- output ----------
print("Estimated empirical loss =", estimated_empirical_loss)
print("KL =", kl_estimate)
print("PAC-Bayes bound (without Ψ) =", pac_bayes_bound)
print("PAC-Bayes bound (with Ψ)    =", pac_bayes_bound_with_psi)

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
plt.show()
