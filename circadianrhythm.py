import numpy as np
import matplotlib.pyplot as plt

# --- 2a) define piecewise profile ---
V_AC = 200.0       # µL
# step flows (µL/min) → rates (s⁻¹):
k_night     = 1.5 / V_AC / 60
k_morning   = 3.0 / V_AC / 60
k_afternoon = 2.4 / V_AC / 60

# time grid
t_sec = np.arange(0,24*3600,60)    # every minute
t_h   = t_sec/3600

# build the piecewise k(t)
k_piece = np.empty_like(t_sec, float)
for i, th in enumerate(t_h):
    if th<6 or th>=22:
        k_piece[i] = k_night
    elif th<14:
        k_piece[i] = k_morning
    else:
        k_piece[i] = k_afternoon

# --- 2b) fit linear model k0 + C cos + S sin ---
omega = 2*np.pi/(24*3600)
X = np.vstack([np.ones_like(t_sec),
               np.cos(omega*t_sec),
               np.sin(omega*t_sec)]).T
k0, C, S = np.linalg.lstsq(X, k_piece, rcond=None)[0]

# recover amplitude & phase
A   = np.hypot(C, S)
phi = np.arctan2(S, C) / omega

# reconstruct smooth cosine
k_fit = k0 + A * np.cos(omega*(t_sec - phi))

# --- 2c) plot overlay ---
plt.figure(figsize=(8,4))
plt.plot(t_h, k_piece*1e4, linewidth=2)
plt.plot(t_h, k_fit*1e4, '--', linewidth=2)
plt.xlim(0,24)
plt.xlabel('Clock time (h)')
plt.ylabel('k_clear (×10⁻⁴ s⁻¹)')
plt.grid(); plt.tight_layout()
plt.show()

# --- 2d) print fitted params ---
print(f"Baseline k₀ = {k0:.3e} s⁻¹")
print(f"Amplitude  A = {A:.3e} s⁻¹")
print(f"Phase φ = {phi/3600:.2f} hours after midnight")
