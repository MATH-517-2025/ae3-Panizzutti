import numpy as np
import matplotlib.pyplot as plt
from math import floor
import os

seed = 42
R = 200
sigma_2 = 1.0
OUTPUT = "output"
np.random.seed(seed)

# actual regression function m(x)
def m(x):
    return np.sin((x / 3 + 0.1) ** -1)


# simulate one dataset
def simulate_data(n, alpha, beta, sigma2 = sigma_2):
    X = np.random.beta(alpha, beta, size=n)
    eps = np.random.normal(0.0, np.sqrt(sigma2), size=n)
    Y = m(X) + eps
    return X, Y

# quantile N blocks of the data
def quantile_blocks(X, N):
    n = len(X)
    order = np.argsort(X)
    base = n // N
    rem = n % N
    counts = np.full(N, base, dtype=int)
    if rem > 0:
        counts[:rem] += 1
    codes = np.empty(n, dtype=int)
    start = 0
    for j in range(N):
        end = start + counts[j]
        idx = order[start:end]
        codes[idx] = j
        start = end
    return codes, N, counts

# quartic OLS in each quantile block and pieces for h
def blocked_quartic_fit_and_estimates(X, Y, N):
    n = len(X)
    codes, K, counts = quantile_blocks(X, N)

    X1 = X
    X2 = X1 * X1
    X3 = X2 * X1
    X4 = X3 * X1

    resid_sq_sum = 0.0
    theta22_sum = 0.0

    for j in range(K):
        idx = np.where(codes == j)[0]
        if len(idx) == 0:
            continue
        Xb = np.column_stack([np.ones(len(idx)), X1[idx], X2[idx], X3[idx], X4[idx]])
        yb = Y[idx]
        beta_hat, _, _, _ = np.linalg.lstsq(Xb, yb, rcond=None)
        b0, b1, b2, b3, b4 = beta_hat
        # m''(x) = 2*b2 + 6*b3*x + 12*b4*x^2
        mpp = 2.0 * b2 + 6.0 * b3 * X1[idx] + 12.0 * b4 * X2[idx]
        theta22_sum += np.sum(mpp ** 2)
        yhat = Xb @ beta_hat
        resid = yb - yhat
        resid_sq_sum += np.sum(resid ** 2)

    theta22_hat = theta22_sum / n
    RSS = resid_sq_sum
    sigma2_hat = RSS / (n - 5 * K)
    return theta22_hat, sigma2_hat, RSS, counts

# h_AMISE bandwidth from estimates
def h_AMISE_from_estimates(n, theta22_hat, sigma2_hat, supp_len=1.0):
    return (n ** (-1.0 / 5.0)) * ((35.0 * sigma2_hat * supp_len) / theta22_hat) ** (1.0 / 5.0)

# choose best N by Mallows Cp 
def choose_N_by_Cp(X, Y):
    n = len(X)
    N_max = max(min(floor(n / 20), 5), 1)
    cand_N = list(range(1, N_max + 1))

    _, _, RSS_Nmax, _ = blocked_quartic_fit_and_estimates(X, Y, N_max)
    denom = RSS_Nmax / (n - 5 * N_max)

    Ns = []
    Cps = []
    for N in cand_N:
        _, _, RSS_N, _ = blocked_quartic_fit_and_estimates(X, Y, N)
        Cp = RSS_N / denom - (n - 10 * N)
        Ns.append(N)
        Cps.append(Cp)

    Ns = np.array(Ns)
    Cps = np.array(Cps)
    N_opt = int(Ns[np.argmin(Cps)])
    return N_opt, Ns, Cps



# simulate one dataset and get one h (optionally choose N)
def estimate_h(n, alpha, beta, sigma2=1.0, N=None, chooseN=False):
    X, Y = simulate_data(n, alpha, beta, sigma2=sigma2)
    if chooseN:
        N_use, _, _ = choose_N_by_Cp(X, Y)
    else:
        N_use = max(1, 1 if N is None else int(N))
        if n <= 5 * N_use:
            N_use = max(1, n // 6)
    theta22_hat, sigma2_hat, RSS, counts = blocked_quartic_fit_and_estimates(X, Y, N_use)
    h_hat = h_AMISE_from_estimates(n, theta22_hat, sigma2_hat, supp_len=1.0)
    return h_hat, N_use, X, Y








# make output folders
os.makedirs(OUTPUT, exist_ok=True)


# Mean and sd of h vs N for three Beta shapes
n_q1 = 2000
N_candidates_q1 = np.arange(1, 6, dtype=int)
shapes_q1 = [(2.0, 2.0, "Beta(2,2)"),
             (1.0, 4.0, "Beta(1,4)"),
             (4.0, 1.0, "Beta(4,1)")]

valid_Ns = [int(N) for N in N_candidates_q1 if n_q1 > 5 * N]
series = []
for a, b, lab in shapes_q1:
    h_means, h_sds = [], []
    for N in valid_Ns:
        hs = [estimate_h(n_q1, a, b, sigma2=sigma_2, N=int(N), chooseN=False)[0]
              for _ in range(R)]
        h_means.append(float(np.mean(hs)))
        h_sds.append(float(np.std(hs, ddof=1)))
    series.append((lab, np.array(h_means), np.array(h_sds)))

plt.figure(figsize=(7,4))
markers = ["o", "s", "D"]
for (lab, means, sds), mk in zip(series, markers):
    plt.errorbar(valid_Ns, means, yerr=sds, fmt=f"{mk}-", capsize=4, label=lab)
plt.xlabel("N")
plt.ylabel("estimated h (mean and sd)")
plt.title(f"Mean and sd of h vs N (n={n_q1})")
plt.grid(True, linestyle="--", alpha=0.4)
plt.xlim(min(valid_Ns)-0.5, max(valid_Ns)+0.5)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT, "h_vs_N_mean.png"), dpi=150, bbox_inches="tight")
plt.close()






# Average N chosen by Cp vs n for three Beta shapes
#slow
ns_q2 = np.array([200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000], dtype=int)
#ns_q2 = np.array([200, 500, 1000, 2000, 5000, 10000, 20000, 50000], dtype=int)
shapes = [(2.0, 2.0, "Beta(2,2)"),
          (1.0, 4.0, "Beta(1,4)"),
          (4.0, 1.0, "Beta(4,1)")]

results = []
for alpha_q2, beta_q2, label in shapes:
    avg_N, sd_N = [], []
    for nval in ns_q2:
        chosen = []
        for _ in range(R):
            Xn, Yn = simulate_data(int(nval), alpha_q2, beta_q2, sigma2=sigma_2)
            N_ch, _, _ = choose_N_by_Cp(Xn, Yn)
            chosen.append(int(N_ch))
        avg_N.append(float(np.mean(chosen)))
        sd_N.append(float(np.std(chosen, ddof=1)))
    results.append((label, np.array(avg_N), np.array(sd_N)))

plt.figure(figsize=(8,4))
markers = ["o", "s", "D"]
for (label, avgN, sdN), mk in zip(results, markers):
    plt.errorbar(ns_q2, avgN, yerr=sdN, fmt=f"{mk}-", capsize=4, label=label)

plt.xlabel("sample size n (log scale)")
plt.ylabel("average N chosen by Cp (with sd)")
plt.title("Cp selected N vs sample size n for different Beta shapes")
plt.grid(True, linestyle="--", alpha=0.4)
plt.xscale("log")
plt.xticks(ns_q2, [str(x) for x in ns_q2], rotation=30)
ymax = max([np.max(avgN + sdN) for _, avgN, sdN in results])
plt.ylim(0.8, ymax + 0.6)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT, "avgN_vs_n.png"), dpi=150, bbox_inches="tight")
plt.close()





# Heatmap of mean h over a grid of alpha and beta
n_q3 = 1000
alpha_grid = np.array([0.5, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)
beta_grid = np.array([0.5, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)

H_mean = np.zeros((len(beta_grid), len(alpha_grid)), dtype=float)
for i, b in enumerate(beta_grid):
    for j, a in enumerate(alpha_grid):
        hs = []
        for _ in range(R):
            h_val, _, _, _ = estimate_h(n_q3, float(a), float(b), sigma2=sigma_2, N=None, chooseN=True)
            hs.append(h_val)
        H_mean[i, j] = float(np.mean(hs))

plt.figure(figsize=(7,5))
im = plt.imshow(H_mean, origin="lower", aspect="auto", cmap="viridis")
plt.xticks(ticks=np.arange(len(alpha_grid)), labels=[f"{a:g}" for a in alpha_grid])
plt.yticks(ticks=np.arange(len(beta_grid)),  labels=[f"{b:g}" for b in beta_grid])
plt.xlabel("alpha")
plt.ylabel("beta")
plt.title(f"Mean h across Beta(alpha, beta) (n={n_q3})")
cbar = plt.colorbar(im)
cbar.set_label("mean estimated h")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT, "heatmap_h_by_alpha_beta.png"), dpi=150, bbox_inches="tight")
plt.close()


# plot m(x) and its second derivative
x = np.linspace(0, 1, 4000)
y = m(x)
mpp_num = np.gradient(np.gradient(y, x), x)

fig, ax = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
ax[0].plot(x, y)
ax[0].set_title("m(x) on [0,1]")
ax[0].set_xlabel("x")
ax[0].set_ylabel("m(x)")
ax[0].set_xlim(0, 1)

ax[1].plot(x, mpp_num)
ax[1].axhline(0, lw=1, alpha=0.4)
ax[1].set_title("m''(x) on [0,1]")
ax[1].set_xlabel("x")
ax[1].set_ylabel("m''(x)")
ax[1].set_xlim(0, 1)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT, "m_and_mpp.png"), dpi=150, bbox_inches="tight")
plt.close()





