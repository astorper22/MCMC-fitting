import numpy as np
import emcee
from scipy import integrate
import matplotlib.pyplot as plt

############ GROWTH RATE ANALYSIS #############

# Time points - in hours
time = np.array([0, 2, 4, 6, 8])

# Bacterial population data (log CFU/mL)
control_pop_025 = np.array([1.333333333, 0, 4.683072674, 4.492373752, 9])  # MIC = 0.25
control_pop_8 = np.array([2.034543, 1.333333333, 4.200686664, 4.842113092, 9])  # MIC = 8

# Carrying capacity function
def popK(pop):
    return max(max(pop), 1.0)  # Ensure K is at least 1 to avoid zero capacity

# Differential equation for bacterial growth
def dBdt(t, B, r, K):
    return r * B * (1 - B / K)

# Log-prior function
def log_prior(params):
    r, sigma = params
    if 0 < r < 1 and 0 < sigma < 3.6:  # Reasonable bounds for r and sigma
        return 0.0
    return -np.inf

# Log-likelihood function
def log_likelihood(params, t, pop):
    r, sigma = params
    if sigma <= 0:
        return -np.inf

    K = popK(pop)
    B0 = max(pop[0], 1e-3)  # Ensure initial population is non-zero

    sol = integrate.solve_ivp(
        lambda t, B: dBdt(t, B, r, K),
        [t[0], t[-1]],
        [B0],
        t_eval=t,
        method="RK45",
        rtol=1e-5,
        atol=1e-8
    )
    B_pred = sol.y[0]
    if np.any(B_pred < 0):  # Avoid unphysical negative values
        return -np.inf
    residuals = pop - B_pred
    log_likelihood_value = -0.5 * np.sum(
        np.log(2 * np.pi * sigma**2) + (residuals**2) / sigma**2
    )
    return log_likelihood_value

# Log-posterior combines prior and likelihood
def log_posterior(params, t, pop):
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(params, t, pop)

# MCMC function for growth rate
def run_mcmc(pop, nwalkers=100, nsteps=1000):
    ndim = 2  # Parameters: r (growth rate), sigma (noise level)
    initial_pos = [
        np.array([0.5, 0.6]) + 0.05 * np.random.randn(ndim) for _ in range(nwalkers)
    ]  # More realistic initial guesses

    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, log_posterior, args=(time, pop)
    )
    sampler.run_mcmc(initial_pos, nsteps, progress=True)
    return sampler


def compute_rhat_and_ess(chain_param):
    """
    chain_param: array of shape (n_steps, n_walkers)
    returns: R̂, ESS
    """
    n, m = chain_param.shape
    # per‐walker means & variances
    means = chain_param.mean(axis=0)
    vars_ = chain_param.var(axis=0, ddof=1)
    # within‐chain and between‐chain
    W = vars_.mean()
    grand_mean = means.mean()
    B = n * np.sum((means - grand_mean)**2) / (m - 1)
    # pooled var
    var_plus = (n - 1)/n * W + B/n
    # R‐hat & ESS
    Rhat = np.sqrt(var_plus / W)
    ESS  = m * n * W / var_plus
    return Rhat, ESS


# Growth rate analysis function
def analyze_growth_rate(pop):
    sampler = run_mcmc(pop)
    samples = sampler.get_chain(flat=True)
    r_samples, sigma_samples = samples[:, 0], samples[:, 1]

    r_median = np.median(r_samples)
    sigma_median = np.median(sigma_samples)

    print(f'Estimated growth rate (r): {r_median:.3f} ± {np.std(r_samples):.3f} log CFU/mL/hr')
    print(f'Estimated noise level (sigma): {sigma_median:.3f} ± {np.std(sigma_samples):.3f}')


     # ——— INSERT HERE ———
    # get the full 3D chain: shape (nsteps, nwalkers, ndim)
    full_chain = sampler.get_chain(discard=0, flat=False)
    for idx, name in enumerate(["r","σ"]):
        Rhat, ESS = compute_rhat_and_ess(full_chain[:,:,idx])
        print(f"  → {name:>2s}: R̂ = {Rhat:.3f}, ESS ≈ {ESS:.0f}")

    # Solve ODE for plotting
    K = popK(pop)
    B0 = max(pop[0], 1e-3)
    t_fit = np.linspace(0, 8, 300)  # Higher resolution for smooth curves
    sol = integrate.solve_ivp(
        lambda t, B: dBdt(t, B, r_median, K),
        [0, 8],
        [B0],
        t_eval=t_fit,
        method="RK45",
        rtol=1e-5,
        atol=1e-8
    )

    # Plot observed data and fitted curve
    plt.scatter(time, pop, color='red', label='Observed Data')
    plt.plot(t_fit, sol.y[0], color='blue', label="Fitted Growth Curve")
    plt.title('Fitted Growth Curve')
    plt.xlabel('Time (hours)')
    plt.ylabel('Log CFU/mL')
    plt.legend()
    plt.show()

    return r_median

print("\nResults for MRSA MIC = 0.25:")
r_025 = analyze_growth_rate(control_pop_025)

print("\nResults for MRSA MIC = 8:")
r_8 = analyze_growth_rate(control_pop_8)


############ KILLING RATE ANALYSIS ###############

def dBdt2(t, B, r, K, mu):
    return r * B * (1 - B / K) - mu * B

# Log-prior function for killing
def log_prior_killing(params):
    sigma, mu = params
    if 0 < sigma < 3.6 and 0 < mu < 2:  # Reasonable bounds for sigma and mu
        return 0.0
    return -np.inf

# Log-likelihood for killing
def log_likelihood_killing(params, t, pop, r_fixed):
    sigma, mu = params
    if sigma <= 0 or mu < 0:
        return -np.inf

    K = popK(pop)
    B0 = max(pop[0], 0.00001)
    sol = integrate.solve_ivp(
    lambda t, B: dBdt2(t, B, r_fixed, K, mu),
    [t[0], t[-1]],  # Ensure integration from start to end
    [B0],
    t_eval=t,  # Force evaluation at these points
    method="RK45",
    rtol=1e-5,
    atol=1e-8
    )
    B_pred = sol.y[0]
    if len(B_pred) != len(pop):
        print(f"Shape mismatch! Expected {len(pop)} points but got {len(B_pred)}")
        print(f"t_eval provided: {t}")
        print(f"t from solver: {sol.t}")  # Check if solver is skipping points
    residuals = pop - B_pred
    log_likelihood_value = -0.5 * np.sum(
        np.log(2 * np.pi * sigma**2) + (residuals**2) / sigma**2
    )
    return log_likelihood_value

# Log-posterior for killing
def log_posterior_killing(params, t, pop, r_fixed):
    lp = log_prior_killing(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_killing(params, t, pop, r_fixed)

def run_mcmc_killing(pop, r_fixed, nwalkers=50, nsteps=1000):
    ndim = 2  # Parameters: sigma, mu
    initial_pos = [np.array([0.6, 1]) + 0.01 * np.random.randn(ndim) for _ in range(nwalkers)]

    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, log_posterior_killing, args=(time, pop, r_fixed)
    )
    sampler.run_mcmc(initial_pos, nsteps, progress=True)
    return sampler

def analyze_killing_rate(pop, r_fixed):
    sampler = run_mcmc_killing(pop, r_fixed)
    samples = sampler.get_chain(flat=True)
    sigma_samples, mu_samples = samples[:, 0], samples[:, 1]

    mu_median = np.median(mu_samples)
    sigma_median = np.median(sigma_samples)

    print(f'Estimated killing rate (mu): {mu_median:.3f} ± {np.std(mu_samples):.3f} log CFU/mL/hr')
    print(f'Estimated noise level (sigma): {sigma_median:.3f} ± {np.std(sigma_samples):.3f}')


    
    # ——— INSERT HERE ———
    full_chain = sampler.get_chain(discard=0, flat=False)
    for idx, name in enumerate(["σ","μ"]):
        Rhat, ESS = compute_rhat_and_ess(full_chain[:,:,idx])
        print(f"  → {name:>2s}: R̂ = {Rhat:.3f}, ESS ≈ {ESS:.0f}")

    # Solve ODE for plotting
    K = popK(pop)
    B0 = max(pop[0], 0.00001)
    t_fit = np.linspace(0, 8, 300)
    sol = integrate.solve_ivp(
        lambda t, B: dBdt2(t, B, r_fixed, K, mu_median),
        [0, 8],
        [B0],
        t_eval=t_fit
    )

    # Plot observed data
    plt.scatter(time, pop, color='red', label='Observed Data')
    plt.plot(t_fit, sol.y[0], color='green', label="Fitted Killing Curve")
    plt.title('Fitted Killing Curve')
    plt.xlabel('Time (hours)')
    plt.ylabel('Log CFU/mL')
    plt.legend()
    plt.show()

    return mu_median, np.std(mu_samples)

# MOXI conc = 0.125 ug/mL at MIC = 0.25
moxi_0125_pop_025 = np.array([1.333333333, 0, 4.232990001, 4.460070414, 5.129603804])
moxi_05_pop_025 = np.array([2.842113092, 2.893747079, 2.865898395, 1.433676665, 0])
moxi_2_pop_025 = np.array([2.725363753, 2.359727082, 0, 1.433676665, 0])
moxi_8_pop_025 = np.array([2.666666667, 1.6666666667, 0, 0, 0])
moxi_32_pop_025 = np.array([2.592717083, 0.333333333, 0, 0, 0])


# MIC = 8
moxi_0125_pop_8 = np.array([1.333333333, 0, 2.767009999, 4.984747503, 9])
moxi_05_pop_8 = np.array([2.133333333, 0, 2.367009999, 4.924747503, 9])
moxi_2_pop_8 = np.array([0.633333333, 0, 2.167009999, 4.384747503, 9])
moxi_8_pop_8 = np.array([2.460070414, 2.660070414, 2.815376012, 3.059507265, 3.339566833])
moxi_32_pop_8 = np.array([2.764085357, 1.6666666667, 0, 0, 0])


print("\nResults for MOXI on MRSA MIC = 0.25:")
mu_0125_025, std_0125_025 = analyze_killing_rate(moxi_0125_pop_025, r_025)
mu_05_025, std_05_025 = analyze_killing_rate(moxi_05_pop_025, r_025)
mu_2_025, std_2_025 = analyze_killing_rate(moxi_2_pop_025, r_025)
mu_8_025, std_8_025 = analyze_killing_rate(moxi_8_pop_025, r_025)
mu_32_025, std_32_025 = analyze_killing_rate(moxi_32_pop_025, r_025)

mu_025_values = np.array([mu_0125_025, mu_05_025, mu_2_025, mu_8_025, mu_32_025])
#std_025_values = np.array([std_0125_025, std_05_025, std_2_025, std_8_025, std_32_025])
 
print("\nResults for MOXI on MRSA MIC = 8:")
mu_0125_8, std_0125_8 = analyze_killing_rate(moxi_0125_pop_8, r_8)
mu_05_8, std_05_8 = analyze_killing_rate(moxi_05_pop_8, r_8)
mu_2_8, std_2_8 = analyze_killing_rate(moxi_2_pop_8, r_8)
mu_8_8, std_8_8 = analyze_killing_rate(moxi_8_pop_8, r_8)
mu_32_8, std_32_8 = analyze_killing_rate(moxi_32_pop_8, r_8)

mu_8_values = np.array([mu_0125_8, mu_05_8, mu_2_8, mu_8_8, mu_32_8])
#std_025_values = np.array([std_0125_025, std_05_025, std_2_025, std_8_025, std_32_025


concentration_values = np.array([0.125, 0.5, 2, 8, 32])


# Killing rate model function
def mu_model_killing_graph(A, mumax, k, MIC, r):
    term = (A / MIC) ** k
    if np.any(term <= 1 - (mumax / r)):  # Prevent invalid math
        return np.inf
    return (mumax * term) / (term - 1 + (mumax / r))

# Log-prior function
def log_prior_killing_graph(params):
    mumax, k = params
    if 0.1 < mumax < 5   and 0.05 < k < 3:
        return 0.0
    return -np.inf

# Log-likelihood function
def log_likelihood_killing_graph(params, A, mu_obs, sigma_obs, MIC, r):
    mumax, k = params
    mu_pred = mu_model_killing_graph(A, mumax, k, MIC, r)
    if np.any(np.isnan(mu_pred) | np.isinf(mu_pred)):
        return -np.inf
    residuals = mu_obs - mu_pred
    return -0.5 * np.sum((residuals / sigma_obs) ** 2)

# Log-posterior function
def log_posterior_killing_graph(params, A, mu_obs, sigma_obs, MIC, r):
    lp = log_prior_killing_graph(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_killing_graph(params, A, mu_obs, sigma_obs, MIC, r)

# MCMC function for parameter estimation
def run_mcmc_killing_graph(A, mu_obs, sigma_obs, MIC, r, nwalkers=500, nsteps=10000):
    ndim = 2  # Parameters: mumax, k
    # Start with initial guess for mumax and k, adjust for better exploration
    initial_pos = [np.array([1.2, 0.4]) + 0.01 * np.random.randn(ndim) for _ in range(nwalkers)]
    print(f"Initial positions: {initial_pos[:5]}")  # Debugging initial positions
    
    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, log_posterior_killing_graph, args=(A, mu_obs, sigma_obs, MIC, r)
    )

    sampler.run_mcmc(initial_pos, nsteps, progress=True)

    print(f"Acceptance fraction: {np.mean(sampler.acceptance_fraction):.3f}")  # Debug acceptance
    print(f"Chain shape: {sampler.get_chain().shape}")  # Debug chain shape
    return sampler

def killing_rate_graph_mcmc(mu_values, conc_values, MIC, r, nwalkers=500, nsteps=10000):
    """
    Run MCMC to fit the Hill‐type killing model μ(A) and plot:
      1. Observed μ vs concentration with fitted curve
      2. Posterior histograms for μmax and k
    Assumes the following functions/variables are already defined in scope:
      - run_mcmc_killing_graph(A, mu_obs, sigma_obs, MIC, r, nwalkers, nsteps)
      - mu_model_killing_graph(A, mumax, k, MIC, r)
      - plt (matplotlib.pyplot)
    """

    # 1) Validate input arrays
    if len(mu_values) == 0 or len(conc_values) == 0:
        raise ValueError("Input data arrays are empty!")

    # 2) Estimate observational noise (sigma_obs) from the spread of mu_values
    sigma_obs = np.std(mu_values) * np.ones_like(mu_values)
    # If all mu_values are identical, avoid zero‐sigma
    if np.any(sigma_obs == 0):
        sigma_obs += 1e-6

    # 3) Run the MCMC sampler
    sampler = run_mcmc_killing_graph(
        conc_values,
        mu_values,
        sigma_obs,
        MIC,
        r,
        nwalkers=nwalkers,
        nsteps=nsteps
    )

    # 4) Discard burn‐in and thin the chain
    burnin = min(500, nsteps // 2)
    chain = sampler.get_chain(discard=burnin, flat=False)  # shape (nsteps‐burnin, nwalkers, 2)
    # Flattened samples shape: ( (nsteps‐burnin) * nwalkers, 2 )
    flat_samples = sampler.get_chain(discard=burnin, flat=True)

    mumax_samples = flat_samples[:, 0]
    k_samples     = flat_samples[:, 1]

    if mumax_samples.size == 0 or k_samples.size == 0:
        raise ValueError("No valid posterior samples were returned.")

    # 5) Summarize posterior statistics
    mumax_median = np.median(mumax_samples)
    mumax_std    = np.std(mumax_samples)
    k_median     = np.median(k_samples)
    k_std        = np.std(k_samples)

    print("=================================")
    print("Fitted Parameters:")
    print(f"  μmax = {mumax_median:.3f} ± {mumax_std:.3f}")
    print(f"     k = {k_median:.3f} ± {k_std:.3f}")
    
    # after you define `chain = sampler.get_chain(..., flat=False)`
    for idx, name in enumerate(["μmax","k"]):
        Rhat, ESS = compute_rhat_and_ess(chain[:,:,idx])
        print(f"  → {name:6s}: R̂ = {Rhat:.3f}, ESS ≈ {ESS:.0f}")


    # 6) Plot observed data and fitted dose–response curve
    plt.figure(figsize=(6, 4))
    plt.scatter(conc_values, mu_values, color='red', label='Observed Data', zorder=3)

    # Create a fine grid of concentrations for the fitted curve
    x_fit = np.linspace(conc_values.min(), conc_values.max(), 200)
    y_fit = mu_model_killing_graph(x_fit, mumax_median, k_median, MIC, r)

    plt.plot(x_fit, y_fit, color='blue', lw=2, label='Fitted Curve', zorder=2)
    plt.xscale('log')
    plt.xlabel('MOXI Concentration (µg/mL)')
    plt.ylabel('Killing Rate μ (log CFU/mL/hr)')
    plt.title(f'Dose–Response Fit (MIC = {MIC})')
    plt.legend()
    plt.grid(which='both', ls='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

    # 7) Plot posterior histograms for μmax and k
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    ax[0].hist(mumax_samples, bins=30, color='skyblue', edgecolor='black', density=True)
    ax[0].set_title("Posterior of μmax")
    ax[0].set_xlabel("μmax")
    ax[0].set_ylabel("Density")
    ax[0].axvline(mumax_median, color='red', linestyle='--', label=f"Median = {mumax_median:.2f}")
    ax[0].legend()

    ax[1].hist(k_samples, bins=30, color='lightgreen', edgecolor='black', density=True)
    ax[1].set_title("Posterior of k")
    ax[1].set_xlabel("k")
    ax[1].set_ylabel("Density")
    ax[1].axvline(k_median, color='red', linestyle='--', label=f"Median = {k_median:.2f}")
    ax[1].legend()

    plt.tight_layout()
    plt.show()



# For MIC=0.25
killing_rate_graph_mcmc(mu_025_values, concentration_values, 0.25, r_025)

# For MIC=8
killing_rate_graph_mcmc(mu_8_values, concentration_values, 8, r_8)



import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.optimize import curve_fit

# -----------------------------------------------------------------------------
# 1) Analytic & numerical growth‐curve fitting
# -----------------------------------------------------------------------------

def plot_growth_curve(time, pop, r):
    """
    Scatter observed log CFU and overlay logistic growth fit with rate r.
    """
    K  = popK(pop)
    B0 = max(pop[0], 1e-3)
    t_fit = np.linspace(time.min(), time.max(), 200)
    sol = integrate.solve_ivp(
        lambda t, B: dBdt(t, B, r, K),
        [time.min(), time.max()],
        [B0],
        t_eval=t_fit,
        rtol=1e-6, atol=1e-9
    )
    plt.figure()
    plt.scatter(time, pop, marker='o', color='red', label='Observed')
    plt.plot(t_fit, sol.y[0], color='blue', lw=2, label=f'Fit (r={r:.2f})')
    plt.xlabel('Time (h)')
    plt.ylabel('Log CFU/mL')
    plt.title('Bacterial Growth Curve')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# -----------------------------------------------------------------------------
# 2) Killing‐curve fitting for each concentration
# -----------------------------------------------------------------------------

def plot_killing_curve(time, pop, r_fixed, mu):
    """
    Scatter observed log CFU and overlay constant‐kill fit with rate μ.
    """
    K  = popK(pop)
    B0 = max(pop[0], 1e-5)
    t_fit = np.linspace(time.min(), time.max(), 200)
    sol = integrate.solve_ivp(
        lambda t, B: dBdt2(t, B, r_fixed, K, mu),
        [time.min(), time.max()],
        [B0],
        t_eval=t_fit,
        rtol=1e-6, atol=1e-9
    )
    plt.figure()
    plt.scatter(time, pop, marker='o', color='black', label='Observed')
    plt.plot(t_fit, sol.y[0], color='green', lw=2, label=f'Kill fit (μ={mu:.2f})')
    plt.xlabel('Time (h)')
    plt.ylabel('Log CFU/mL')
    plt.title(f'Killing Curve (r fixed={r_fixed:.2f})')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------------------------
# 4) Run analyses with your real data
# -----------------------------------------------------------------------------

# a) Growth‐rate estimation
sam_025 = run_mcmc(control_pop_025)
flat_025 = sam_025.get_chain(flat=True)
r_025 = np.median(flat_025[:, 0])
print(f"Estimated r (MIC=0.25): {r_025:.3f}")

sam_8 = run_mcmc(control_pop_8)
flat_8 = sam_8.get_chain(flat=True)
r_8 = np.median(flat_8[:, 0])
print(f"Estimated r (MIC=8): {r_8:.3f}")

# b) Plot growth curves
plot_growth_curve(time, control_pop_025, r_025)
plot_growth_curve(time, control_pop_8,   r_8)

# c) Killing‐rate estimation & plots (MIC=0.25)
mu_vals_025 = []
for pop in [moxi_0125_pop_025, moxi_05_pop_025, moxi_2_pop_025, moxi_8_pop_025, moxi_32_pop_025]:
    mu_med, _ = analyze_killing_rate(pop, r_025)
    mu_vals_025.append(mu_med)

for pop, mu in zip(
    [moxi_0125_pop_025, moxi_05_pop_025, moxi_2_pop_025, moxi_8_pop_025, moxi_32_pop_025],
    mu_vals_025
):
    plot_killing_curve(time, pop, r_025, mu)
    
    
# c) Killing‐rate estimation & plots (MIC=0.25)
mu_vals_8 = []
for pop in [moxi_0125_pop_8, moxi_05_pop_8, moxi_2_pop_8, moxi_8_pop_8, moxi_32_pop_8]:
    mu_med, _ = analyze_killing_rate(pop, r_8)
    mu_vals_8.append(mu_med)

for pop, mu in zip(
    [moxi_0125_pop_8, moxi_05_pop_8, moxi_2_pop_8, moxi_8_pop_8, moxi_32_pop_8],
    mu_vals_8
):
    plot_killing_curve(time, pop, r_8, mu)



import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import corner

# -----------------------------------------------------------------------------
# ASSUME previous definitions and MCMC runs:
#   time, control_pop_025, r_025, sampler_025, flat_025
#   popK, dBdt, run_mcmc, analyze_killing_rate, concentration_values, mu_vals_025
# -----------------------------------------------------------------------------




import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import corner

# -----------------------------------------------------------------------------
# 0) Run growth‐rate MCMC so sampler_025 & flat_025 are defined
# -----------------------------------------------------------------------------
sampler_025 = run_mcmc(control_pop_025)       # uses your previously defined run_mcmc()
flat_025    = sampler_025.get_chain(flat=True)

sampler_8 = run_mcmc(control_pop_8)
flat_8    = sampler_8.get_chain(flat=True)


# -----------------------------------------------------------------------------
# 1) Posterior predictive bands for growth
# -----------------------------------------------------------------------------
idx    = np.random.choice(flat_025.shape[0], 100, replace=False)
all_B  = []
t_pred = np.linspace(time.min(), time.max(), 200)
for i in idx:
    r_i = flat_025[i, 0]
    sol = integrate.solve_ivp(
        lambda t, B: dBdt(t, B, r_i, popK(control_pop_025)),
        [time[0], time[-1]],
        [control_pop_025[0]],
        t_eval=t_pred
    )
    all_B.append(sol.y[0])
all_B = np.array(all_B)
low, high = np.percentile(all_B, [5, 95], axis=0)

plt.figure()
plt.fill_between(t_pred, low, high, color='C2', alpha=0.3, label='90% PI')
plt.plot(t_pred, np.median(all_B, axis=0), 'C2', lw=2, label='Median Sim')
plt.scatter(time, control_pop_025, c='k', label='Observed')
plt.xlabel("Time (h)")
plt.ylabel("Log CFU/mL")
plt.title("Posterior Predictive: Growth (MIC=0.25)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


idx    = np.random.choice(flat_8.shape[0], 100, replace=False)
all_B  = []
t_pred = np.linspace(time.min(), time.max(), 200)
for i in idx:
    r_i = flat_8[i, 0]
    sol = integrate.solve_ivp(
        lambda t, B: dBdt(t, B, r_i, popK(control_pop_8)),
        [time[0], time[-1]],
        [control_pop_8[0]],
        t_eval=t_pred
    )
    all_B.append(sol.y[0])
all_B = np.array(all_B)
low, high = np.percentile(all_B, [5, 95], axis=0)

plt.figure()
plt.fill_between(t_pred, low, high, color='C2', alpha=0.3, label='90% PI')
plt.plot(t_pred, np.median(all_B, axis=0), 'C2', lw=2, label='Median Sim')
plt.scatter(time, control_pop_8, c='k', label='Observed')
plt.xlabel("Time (h)")
plt.ylabel("Log CFU/mL")
plt.title("Posterior Predictive: Growth (MIC=8)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# -----------------------------------------------------------------------------
# 2) Violin plot of posterior marginals (r and σ)
# -----------------------------------------------------------------------------
plt.figure(figsize=(4,4))
_ = plt.violinplot([flat_025[:,0], flat_025[:,1]],
                   showmeans=True, showextrema=False)
plt.xticks([1,2], ["r","σ"])
plt.ylabel("Value")
plt.title("Posterior Marginals (MIC=0.25)")
plt.grid(True, axis='y')
plt.tight_layout()
plt.show()

plt.figure(figsize=(4,4))
_ = plt.violinplot([flat_8[:,0], flat_8[:,1]],
                   showmeans=True, showextrema=False)
plt.xticks([1,2], ["r","σ"])
plt.ylabel("Value")
plt.title("Posterior Marginals (MIC=8)")
plt.grid(True, axis='y')
plt.tight_layout()
plt.show()

# -----------------------------------------------------------------------------
# 3) Trace plots of the chains
# -----------------------------------------------------------------------------
chain = sampler_025.get_chain()  # shape (nsteps, nwalkers, ndim)
nsteps, nwalkers, ndim = chain.shape

fig, axes = plt.subplots(2,1, figsize=(6,4), sharex=True)
for w in range(nwalkers):
    axes[0].plot(chain[:,w,0], color='C0', alpha=0.3)
    axes[1].plot(chain[:,w,1], color='C1', alpha=0.3)

axes[0].set_ylabel("r")
axes[0].set_title("MCMC Trace")
axes[1].set_ylabel("σ")
axes[1].set_xlabel("Step")

for ax in axes:
    ax.grid(True)

plt.tight_layout()
plt.show()


chain = sampler_8.get_chain()  # shape (nsteps, nwalkers, ndim)
nsteps, nwalkers, ndim = chain.shape

fig, axes = plt.subplots(2,1, figsize=(6,4), sharex=True)
for w in range(nwalkers):
    axes[0].plot(chain[:,w,0], color='C0', alpha=0.3)
    axes[1].plot(chain[:,w,1], color='C1', alpha=0.3)

axes[0].set_ylabel("r")
axes[0].set_title("MCMC Trace")
axes[1].set_ylabel("σ")
axes[1].set_xlabel("Step")

for ax in axes:
    ax.grid(True)

plt.tight_layout()
plt.show()


import numpy as np
import matplotlib.pyplot as plt
import corner

# -----------------------------------------------------------------------------
# 4) Corner plot for growth‐rate posterior (r, σ)
# -----------------------------------------------------------------------------
flat_025 = sampler_025.get_chain(flat=True)
if corner is not None:
    fig = corner.corner(
        flat_025,
        labels=["r", "σ"],
        show_titles=True,
        title_fmt=".2f"
    )
    fig.suptitle("Posterior of Growth‐Rate Parameters mic=0.25", y=1.02)
    plt.tight_layout()
    plt.show()

flat_8 = sampler_8.get_chain(flat=True)
if corner is not None:
    fig = corner.corner(
        flat_8,
        labels=["r", "σ"],
        show_titles=True,
        title_fmt=".2f"
    )
    fig.suptitle("Posterior of Growth‐Rate Parameters mic=8", y=1.02)
    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------------------------
# 5) Killing‐curve parameter posterior (μmax, k) via MCMC
# -----------------------------------------------------------------------------
# Prepare data for MIC = 0.25
A = concentration_values
mu_obs = np.array(mu_vals_025)
sigma_obs = np.std(mu_obs) * np.ones_like(mu_obs)

# Run MCMC for Hill‐type Emax model
sampler_kill = run_mcmc_killing_graph(
    A, mu_obs, sigma_obs, 0.25, r_025,
    nwalkers=200, nsteps=5000
)

# Flatten the chain after discarding burn‐in
burnin = 1000
chain_kill = sampler_kill.get_chain(discard=burnin, flat=True)
if corner is not None:
    fig = corner.corner(
        chain_kill,
        labels=[r"$\mu_{\max}$", "k"],
        show_titles=True,
        title_fmt=".2f"
    )
    fig.suptitle("Posterior of Killing‐Curve Parameters mic=0.25", y=1.02)
    plt.tight_layout()
    plt.show()
    
    
    
import numpy as np
import matplotlib.pyplot as plt
import corner

# -----------------------------------------------------------------------------
# 4) Corner plot for growth‐rate posterior (r, σ) at MIC = 8
# -----------------------------------------------------------------------------
# Assume sampler_8 was created via: sampler_8 = run_mcmc(control_pop_8)
flat_8 = sampler_8.get_chain(flat=True)  # shape (nwalkers*nsteps, 2)

fig = corner.corner(
    flat_8,
    labels=["r", "σ"],
    show_titles=True,
    title_fmt=".2f"
)
fig.suptitle("Posterior of Growth‐Rate Parameters (MIC = 8)", y=1.02)
plt.tight_layout()
plt.show()

# -----------------------------------------------------------------------------
# 5) Corner plot for killing‐curve parameters (μmax, k) at MIC = 8
# -----------------------------------------------------------------------------
# Prepare the dose–response data at MIC=8
A       = concentration_values           # e.g. np.array([0.125, 0.5, 2, 8, 32])
mu_obs8 = np.array(mu_8_values)          # μ medians for MIC = 8
sigma8  = np.std(mu_obs8) * np.ones_like(mu_obs8)

# Run MCMC for the Hill‐type Emax model at MIC=8
sampler_kill8 = run_mcmc_killing_graph(
    A, mu_obs8, sigma8, 8, r_8,
    nwalkers=200, nsteps=5000
)

# Discard burn‐in and flatten
burnin     = 1000
flat_kill8 = sampler_kill8.get_chain(discard=burnin, flat=True)  # shape (nwalkers*(nsteps−burnin), 2)

fig = corner.corner(
    flat_kill8,
    labels=[r"$\mu_{\max}$", "k"],
    show_titles=True,
    title_fmt=".2f"
)
fig.suptitle("Posterior of Killing‐Curve Parameters (MIC = 8)", y=1.02)
plt.tight_layout()
plt.show()


import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# 5) Autocorrelation for each parameter
# -----------------------------------------------------------------------------
def plot_autocorr(chain, max_lag=50):
    """
    Plot the mean autocorrelation function (ACF) for each parameter
    across all walkers.
    
    Parameters
    ----------
    chain : ndarray, shape (n_steps, n_walkers, n_dim)
        The MCMC chain.
    max_lag : int
        How many lags to display.
    """
    nsteps, nwalkers, ndim = chain.shape
    
    # Parameter names (extendable if ndim > 2)
    names = [r"$r$", r"$\sigma$"]
    if ndim > len(names):
        names += [f"$p_{{{i}}}$" for i in range(len(names)+1, ndim+1)]
    
    # Create a wider figure
    fig, axes = plt.subplots(1, ndim, figsize=(8, 3), sharey=True)
    
    for i in range(ndim):
        acfs = []
        for w in range(nwalkers):
            x = chain[:, w, i]
            x = x - x.mean()
            acf_full = np.correlate(x, x, mode="full")
            # take lags 0..max_lag-1
            acf = acf_full[nsteps-1 : nsteps-1 + max_lag] / acf_full[nsteps-1]
            acfs.append(acf)
        
        mean_acf = np.mean(acfs, axis=0)
        lag = np.arange(max_lag)
        
        axes[i].plot(lag, mean_acf, lw=1.5, color=f"C{i}")
        axes[i].set_title(names[i], fontsize=12)
        axes[i].set_xlabel("Lag", fontsize=10)
        axes[i].grid(ls="--", alpha=0.5)
    
    axes[0].set_ylabel("Autocorrelation", fontsize=10)
    
    # Put the main title in the top margin
    fig.suptitle("Autocorrelation Functions", fontsize=14)
    fig.subplots_adjust(top=0.85, wspace=0.3)
    plt.tight_layout(rect=[0, 0, 1, 0.85])
    plt.show()


# Assuming `chain` is your array of shape (n_steps, n_walkers, 2)
plot_autocorr(chain, max_lag=50)


import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, simps
from SALib.sample import saltelli
from SALib.analyze import sobol




# -----------------------------------------------------------------------------
# Prepare killing‐curve posterior samples for MIC=0.25 and MIC=8
# -----------------------------------------------------------------------------
# (Re‐run MCMC to get full posterior chains if not stored)
sigma_025_obs = np.std(mu_025_values) * np.ones_like(mu_025_values)
sampler_kill25 = run_mcmc_killing_graph(
    concentration_values, mu_025_values, sigma_025_obs, 0.25, r_025,
    nwalkers=200, nsteps=5000
)
flat_kill25 = sampler_kill25.get_chain(discard=1000, flat=True)  # shape (n_samples, 2)

sigma_8_obs = np.std(mu_8_values) * np.ones_like(mu_8_values)
sampler_kill8 = run_mcmc_killing_graph(
    concentration_values, mu_8_values, sigma_8_obs, 8, r_8,
    nwalkers=200, nsteps=5000
)
flat_kill8 = sampler_kill8.get_chain(discard=1000, flat=True)      # shape (n_samples, 2)

# Determine initial populations for clearance simulations
B0_025 = control_pop_025[0]
B0_8   = control_pop_8[0]

# -----------------------------------------------------------------------------
# A) Doubling Time Bar Chart
# -----------------------------------------------------------------------------
T_double_025 = np.log(2) / r_025
T_double_8   = np.log(2) / r_8

plt.figure(figsize=(4,3))
plt.bar(["MIC=0.25", "MIC=8"], [T_double_025, T_double_8], color=["C0","C1"])
plt.ylabel("Doubling time (h)")
plt.title("Bacterial Doubling Times")
plt.grid(axis="y", ls="--", alpha=0.6)
plt.tight_layout()
plt.show()


# -----------------------------------------------------------------------------
# B) Log‐Kill Slopes 
# -----------------------------------------------------------------------------
def compute_log_kill_slope(time, pop):
    mask = pop > 0
    y = pop[mask]     # log CFU
    x = time[mask]    # hours
    coef = np.polyfit(x, y, 1)
    return coef[0]    # slope

# MIC = 0.25
kill_sets_025 = [
    (moxi_0125_pop_025, "0.125 µg/mL"),
    (moxi_05_pop_025,   "0.5 µg/mL"),
    (moxi_2_pop_025,    "2 µg/mL"),
    (moxi_8_pop_025,    "8 µg/mL"),
    (moxi_32_pop_025,   "32 µg/mL")
]
slopes_025  = [compute_log_kill_slope(time, pop) for pop, _ in kill_sets_025]

# MIC = 8
kill_sets_8 = [
    (moxi_0125_pop_8, "0.125 µg/mL"),
    (moxi_05_pop_8,   "0.5 µg/mL"),
    (moxi_2_pop_8,    "2 µg/mL"),
    (moxi_8_pop_8,    "8 µg/mL"),
    (moxi_32_pop_8,   "32 µg/mL")
]
slopes_8   = [compute_log_kill_slope(time, pop) for pop, _ in kill_sets_8]

plt.figure(figsize=(5,3))
plt.plot(concentration_values, slopes_025, '-o', color='C2', label="MIC=0.25")
plt.plot(concentration_values, slopes_8,   '-o', color='C3', label="MIC=8")
plt.xscale('log')
plt.xlabel("Concentration (µg/mL)")
plt.ylabel("Log‐Kill slope (log CFU/h)")
plt.title("Kill Slope vs Concentration")
plt.legend(fontsize=9)
plt.grid(which='both', ls='--', alpha=0.6)
plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------------------------------------------------------
# C) Dose–Response Credible Bands (Seaborn-style)
# -----------------------------------------------------------------------------
def plot_credible_dose_response(flat_kill, MIC, r_est, color, label, obs_conc, obs_mu, marker):
    # Generate a log-spaced concentration grid
    C_grid = np.logspace(
        np.log10(concentration_values.min()),
        np.log10(concentration_values.max()),
        200
    )
    
    # Compute μ curves for each posterior sample
    all_mu = np.zeros((flat_kill.shape[0], C_grid.size))
    for j in range(flat_kill.shape[0]):
        mumax_j, k_j = flat_kill[j, 0], flat_kill[j, 1]
        all_mu[j, :] = mu_model_killing_graph(C_grid, mumax_j, k_j, MIC, r_est)
    
    # Compute 2.5th, 97.5th percentiles, and median
    low_mu    = np.percentile(all_mu, 2.5, axis=0)
    high_mu   = np.percentile(all_mu, 97.5, axis=0)
    median_mu = np.median(all_mu, axis=0)
    
    # Plot 95% credible band (fill-between)
    plt.fill_between(
        C_grid,
        low_mu,
        high_mu,
        color=color,
        alpha=0.3,
        label=f"{label} 95% CI"
    )
    
    # Plot median dose–response curve
    plt.plot(
        C_grid,
        median_mu,
        color=color,
        lw=2,
        label=f"{label} median"
    )
    
    # Plot observed killing rates
    sns.scatterplot(
        x=obs_conc,
        y=obs_mu,
        color='k',
        marker=marker,
        s=50,
        label=f"Obs (MIC={MIC})"
    )


# Prepare figure
plt.figure(figsize=(6, 5))
sns.set_style("whitegrid")
sns.set_context("talk")

# Plot for MIC = 0.25
plot_credible_dose_response(
    flat_kill25, 0.25, r_025,
    color='C2', label="MIC=0.25",
    obs_conc=concentration_values, obs_mu=mu_025_values,
    marker='o'
)

# Plot for MIC = 8
plot_credible_dose_response(
    flat_kill8,  8, r_8,
    color='C3', label="MIC=8",
    obs_conc=concentration_values, obs_mu=mu_8_values,
    marker='s'
)

# Axis labels, title, and legend
plt.xscale("log")
plt.yscale("linear")
plt.xlabel("Concentration (µg/mL)")
plt.ylabel("Killing rate μ (log CFU/mL/hr)")
plt.title("Dose–Response with 95% Credible Bands")
plt.legend(fontsize=10, frameon=True, edgecolor="gray")
plt.tight_layout()
plt.show()

