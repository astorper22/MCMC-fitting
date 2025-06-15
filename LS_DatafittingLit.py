import numpy as np
from scipy import optimize, integrate
import matplotlib.pyplot as plt

# Time points - in hours
time = np.array([0, 4, 8, 12, 24, 36, 48])

# Various MRSA strains - in log CFU / mL
std_pop1 = np.array([5.5, 6.3457, 7.3958, 7.9092, 8.0178, 8.6377, 8.6441])
std_pop2 = np.array([5.7, 6.1735, 7.63, 7.8476, 8.1524, 8.0218, 8])
std_pop3 = np.array([5.6, 6.4647, 8.1593, 8.0582, 8.1721, 8.0842, 8.0693])
std_pop4 = np.array([5.5, 6.4948, 8.0673, 8.1315, 7.971, 9.03, 9.03])
std_pop5 = np.array([5.5, 6.7308, 7.4098, 7.8291, 8.5351, 8.4945, 8.5189])

# Population data list
pops = [std_pop1, std_pop2, std_pop3, std_pop4, std_pop5]

def popK(pop):
    return pop[-1]

def dBdt(t, B, r, K):
    return r * B * (1 - B / K)

# Residual function for least squares fitting
def residuals(params, t, pop):
    r = params[0]  
    K = 9  
    B0 = 5  

    # Integrate ODE 
    sol = integrate.solve_ivp(dBdt, [t[0], t[-1]], [B0], t_eval=t, args=(r, K))
    B_pred = sol.y[0]  # Predicted values

    return (B_pred - pop)

def fit_growth_rate(pop):
    initial_guess = [0.1]  # Initial guess for r
    result = optimize.least_squares(residuals, initial_guess, args=(time, pop))
    return result.x[0]  # Return the estimated growth rate

def analyze(pop):
    r_estimated = fit_growth_rate(pop)
    print('Estimated growth rate (r):', r_estimated, 'log CFU/mL/hr')
    
    # Solve ODE for plotting the fitted curve
    K = popK(pop)
    B0 = 5
    t_fit = np.linspace(0, 48, 300)
    sol = integrate.solve_ivp(dBdt, [0, 48], [B0], t_eval=t_fit, args=(r_estimated, K))
    
    # Plot data and fitted curve
    plt.scatter(time, pop, color='red')
    plt.plot(t_fit, sol.y[0], color='blue')
    plt.title('Fitted Growth Curve')
    plt.xlabel('Time (hours)')
    plt.ylabel('Log CFU/mL')
    plt.show()

analyze(std_pop3)
