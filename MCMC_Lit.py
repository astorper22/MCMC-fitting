import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
import emcee

# Time points - in hours

# MRSA strain data - in log CFU / mL
std_pop1 = np.array([5.5, 6.3457, 7.3958, 7.9092, 8.0178, 8.6377, 8.6441])
std_pop2 = np.array([5.7, 6.1735, 7.63, 7.8476, 8.1524, 8.0218, 8])
std_pop3 = np.array([5.6, 6.4647, 8.1593, 8.0582, 8.1721, 8.0842, 8.0693])
std_pop4 = np.array([5.5, 6.4948, 8.0673, 8.1315, 7.971, 9.03, 9.03])
std_pop5 = np.array([5.5, 6.7308, 7.4098, 7.8291, 8.5351, 8.4945, 8.5189])


# Time points in hours
time = np.array([0, 4, 8, 12, 24, 36, 48])


# Estimated carrying capacity (use max of observed data)
def popK(pop):
    return np.max(pop)

# Logistic growth differential equation
def dBdt(t, B, r, K):
    return r * B * (1 - B / K)

# Log-likelihood for MCMC
def log_likelihood(params, t, pop, pop_err):
    r = params[0]
    K = popK(pop)
    B0 = 5.0  # Initial inoculum (log CFU/mL)

    sol = integrate.solve_ivp(lambda t, B: dBdt(t, B, r, K),
                              [t[0], t[-1]], [B0], t_eval=t)
    B_pred = sol.y[0]
    residuals = (pop - B_pred) / pop_err
    return -0.5 * np.sum(residuals**2)

# Uniform prior for r
def log_prior(params):
    r = params[0]
    if 0.0 < r < 2.0:
        return 0.0
    return -np.inf

# Posterior = prior + likelihood
def log_posterior(params, t, pop, pop_err):
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(params, t, pop, pop_err)

# MCMC sampling function
def run_mcmc(pop, pop_err=0.1):
    nwalkers = 32
    ndim = 1
    initial_guess = np.array([0.1])
    pos = initial_guess + 1e-3 * np.random.randn(nwalkers, ndim)

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=(time, pop, pop_err))
    sampler.run_mcmc(pos, 1000, progress=True)

    samples = sampler.get_chain(discard=100, thin=15, flat=True)
    return samples

# Plot results
def analyze_mcmc(samples, pop):
    r_mean = np.mean(samples)
    r_std = np.std(samples)

    print(f"Estimated growth rate (r): {r_mean:.3f} ± {r_std:.3f} log CFU/mL/hr")

    K = popK(pop)
    B0 = 5.0
    t_fit = np.linspace(0, 48, 300)
    sol = integrate.solve_ivp(lambda t, B: dBdt(t, B, r_mean, K), [0, 48], [B0], t_eval=t_fit)

    plt.figure(figsize=(8, 5))
    plt.scatter(time, pop, color='red', label='Observed Data')
    plt.plot(t_fit, sol.y[0], color='blue', label='Fitted Logistic Model')
    plt.title('MRSA Logistic Growth Fit (MCMC)')
    plt.xlabel('Time (hours)')
    plt.ylabel('Bacterial Density (log CFU/mL)')
    plt.legend()
    plt.tight_layout()
    plt.show()

# Run MCMC and analyze for std_pop3
samples = run_mcmc(std_pop3)
analyze_mcmc(samples, std_pop3)

