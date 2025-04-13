import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KernelDensity
import scipy.stats as stats

# ---------- 参数设置 ----------
np.random.seed(0)
T = 100  # 时间步长
sigma2 = 0.02
lambda_N = np.sqrt(T)
delta = 0.05

# ---------- 构造数据生成器 S_g ----------
# 参数设定（来自论文 appendix B 中 equation 214）
A_g = np.array([[0.8, 0.1], [0.05, 0.9]])
B_g = np.array([[0.1], [0.2]])
b_sg = np.array([0.1, -0.1])
C_g = np.array([[1.0, -1.0]])
D_g = np.array([[0.5]])
b_yg = np.array([0.0])

n_s = 2  # 状态维度
n_x = 1
n_y = 1

s_g = np.zeros((T + 1, n_s))
e_g = np.clip(np.random.normal(0, 0.2, size=(T, 1)), -1, 1)  # 截断高斯输入

y_data = np.zeros((T, n_y))
x_data = np.zeros((T, n_x))

for t in range(T):
    s_g[t + 1] = np.maximum(0, A_g @ s_g[t] + B_g @ e_g[t] + b_sg)  # ReLU
    combined = C_g @ s_g[t] + D_g @ e_g[t] + b_yg
    output = np.tanh(combined)
    y_data[t] = output[0]
    x_data[t] = output[0]

# ---------- 构造预测器结构（同生成器） ----------
def forward_predictor(e, theta, s0):
    A, B, b, C, D, c = theta
    s = s0.copy()
    y_hat = []
    for t in range(len(e)):
        s = np.maximum(0, A @ s + B @ e[t] + b)
        y = np.tanh(C @ s + D @ e[t] + c)
        y_hat.append(y)
    return np.array(y_hat)

# ---------- 参数展开与重构 ----------
def unpack_theta(theta_vec):
    A = theta_vec[0:4].reshape(2, 2)
    B = theta_vec[4:6].reshape(2, 1)
    b = theta_vec[6:8]
    C = theta_vec[8:10].reshape(1, 2)
    D = theta_vec[10:11].reshape(1, 1)
    c = theta_vec[11:12]
    return A, B, b, C, D, c

def empirical_loss(theta_vec):
    theta = unpack_theta(theta_vec)
    y_pred = forward_predictor(e_g, theta, np.zeros(2))
    return np.mean((y_pred[:, 0] - y_data[:, 0]) ** 2)

def unnormalized_log_posterior(theta_vec):
    log_prior = -np.sum(theta_vec**2) / (2 * sigma2)
    return log_prior - lambda_N * empirical_loss(theta_vec)

# ---------- Metropolis-Hastings 采样 ----------
def metropolis_hastings(log_prob_func, init_theta, num_samples, step_size=0.05):
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

init_theta = np.random.randn(12) * 0.1
samples = metropolis_hastings(unnormalized_log_posterior, init_theta, num_samples=2000)

# ---------- 计算经验损失 ----------
empirical_losses = np.array([empirical_loss(theta) for theta in samples])
estimated_empirical_loss = np.mean(empirical_losses)

# ---------- KDE 拟合后验，估计 KL ----------
kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(samples)
log_rho = kde.score_samples(samples)
rho_vals = np.exp(log_rho)
log_pi = stats.multivariate_normal(mean=np.zeros(12), cov=sigma2 * np.eye(12)).logpdf(samples)
pi_vals = np.exp(log_pi)

eps = 1e-12
rho_vals = np.clip(rho_vals, eps, None)
pi_vals = np.clip(pi_vals, eps, None)
kl_estimate = np.mean(log_rho - log_pi)

# ---------- PAC-Bayes Bound ----------
log_term = np.log(2 * np.sqrt(T) / delta)
bound = estimated_empirical_loss + np.sqrt((kl_estimate + log_term) / (2 * T))

import pandas as pd
import matplotlib.pyplot as plt

# ---------- 可视化预测误差分布 ----------
plt.figure(figsize=(8, 5))
sns.histplot(empirical_losses, bins=40, kde=True, color="skyblue")
plt.axvline(x=estimated_empirical_loss, color='red', linestyle='--', label="Avg Loss")
plt.title("Empirical Loss Distribution (Gibbs Posterior)")
plt.xlabel("Empirical Loss")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

import pprint
result_summary = {
    "平均经验误差": estimated_empirical_loss,
    "KL 散度": kl_estimate,
    "PAC-Bayes 泛化误差上界": bound
}

pprint.pprint(result_summary)
