import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# =============================================================================
# 1) THERAPY MODE: 'mono' or 'rotate'
# =============================================================================
mode = 'rotate'   # 'mono' for A only, 'rotate' to alternate A/B hourly

# =============================================================================
# 2) PARAMETERS (PD, growth, geometry, diffusion, clearance)
# =============================================================================
# Moxi
r_025_hr       = 0.333
muA_max_hr     = 1.789
h_A            = 0.93
EC50_A         = 6.76*8 #6.76*MIC
# Cipro
muB_max_hr = 0.294
h_B        = 1.1
EC50_B     = 0.18  # µg/mL


# convert to per-second
r_025   = r_025_hr   / 3600.0
muA_max = muA_max_hr/ 3600.0
muB_max = muB_max_hr/ 3600.0

# carrying cap & init bacteria
K    = 1.0
B0   = 1.0

# geometry (m)
R            = 12e-3
cornea_thick = 0.55e-3
aqueous_thick= 2.53e-3
r_vit        = R - cornea_thick - aqueous_thick

# radial grid
Nr = 20
dr = R/Nr
r  = np.linspace(0, R, Nr+1)

# diffusion D(r) (m²/s)
D = np.zeros_like(r)
D[r > (R-cornea_thick)]               = 5.84e-10
mask_aq = (r <= (R-cornea_thick)) & (r>r_vit)
D[mask_aq]                            = 8.23e-10
D[r <= r_vit]                         = 3.4e-10

# aqueous clearance (s⁻¹)
k_clear       = np.zeros_like(r)
k_clear[mask_aq] = 2.35e-4

# ── Circadian Aqueous‐Clearance Parameters (from Fitted ψ(t)) ───────────────
V_AC     = 200.0        # μL anterior chamber volume (fixed)
k_base   = 1.917e-4     # s⁻¹ (baseline clearance from fit)
k_amp    = 6.008e-5     # s⁻¹ (amplitude from fit)
omega    = 2 * np.pi / (24 * 3600)  # rad/s (circadian frequency)
phi      = -11.57 * 3600            # phase shift in seconds (from fit: −11.57 h after midnight)



# =============================================================================
# 3) TIME & DOSING
# =============================================================================
dt         = 5.0                      # 1 s time step
t_end_hr   = 72
Nt         = int(t_end_hr*3600/dt)+1
times      = np.linspace(0, t_end_hr*3600, Nt)

# cmax amounts (µg/mL)
Cmax_A     = 1.95
Cmax_B     = 1.12

# schedule: for monotherapy, only A is ever dosed.
# for rotate: period in seconds to switch drugs (e.g. 1 hr)
rot_period = 240*60  

# =============================================================================
# 4) INITIALIZE FIELDS
# =============================================================================
C_A = np.zeros((Nt, Nr+1))
C_B = np.zeros((Nt, Nr+1))
B   = np.zeros((Nt, Nr+1))

# initial pulse
C_A[0,-1] = Cmax_A
C_B[0,-1] = 0.0
B[0, :]   = B0



# =============================================================================
# 5) SIMULATION LOOP
# =============================================================

for ti in range(1, Nt):
    
    if ti == 1:
        print(f"Entering time‐step loop: ti={ti}, t={Nt:.1f}s")

    
    t = times[ti]


    # 5.0) circadian clearance in aqueous chamber
    k_t = k_base + k_amp * np.cos(omega * (t - phi))
    k_clear[mask_aq] = k_t       # instant. clearance [s⁻¹] in just the AC

    # grab previous state
    CAold = C_A[ti-1].copy()
    CBold = C_B[ti-1].copy()
    Bold  = B[ti-1].copy()
    CAnew, CBnew, Bnew = CAold.copy(), CBold.copy(), Bold.copy()

    # --- 5a) diffusion + clearance for A & B ---
    for i in range(1, Nr):
        
        lapA = ((r[i+1]**2*(CAold[i+1] - CAold[i]))
              - (r[i]  **2*(CAold[i]   - CAold[i-1]))) \
              / (r[i]**2 * dr**2)
        lapB = ((r[i+1]**2*(CBold[i+1] - CBold[i]))
              - (r[i]  **2*(CBold[i]   - CBold[i-1]))) \
              / (r[i]**2 * dr**2)
        if ti == 1 and i == 1:
            print(f"lapA at ti=1, i=1 → {lapA:.3e}")


        CAnew[i] = CAold[i] + dt*(D[i]*lapA - k_clear[i]*CAold[i])
        CBnew[i] = CBold[i] + dt*(D[i]*lapB - k_clear[i]*CBold[i])

    # symmetry & zero‐flux
    CAnew[0], CBnew[0] = CAnew[1], CBnew[1]
    CAnew[-1],CBnew[-1] = CAnew[-2], CBnew[-2]

    # --- 5b) dosing pulse ---
    if mode=='rotate':
        if int(t//rot_period) % 2 == 0:
            CAnew[-1] = Cmax_A
        else:
            CBnew[-1] = Cmax_B
    else:
        CAnew[-1] = Cmax_A

    # --- 5c) bacterial growth + kill ---
    muA = muA_max * CAold**h_A / (CAold**h_A + EC50_A**h_A + 1e-12)
    muB = muB_max * CBold**h_B / (CBold**h_B + EC50_B**h_B + 1e-12)
    muT = muA + muB

    growth = r_025 * Bold * (1 - Bold/K)
    kill   = muT * Bold
    Bnew   = (Bold + dt*(growth - kill)).clip(min=0)

    # store
    C_A[ti], C_B[ti], B[ti] = CAnew, CBnew, Bnew



# =============================================================================
# 6) SNAPSHOTS
# =============================================================================
snap_hrs = [2, 8, 16, 24, 48, 72]
snap_idx = [int(hr*3600/dt) for hr in snap_hrs]

# =============================================================================
# 7) PLOTTING
# =============================================================================
def plot_1d(field, name, vmax=None):
    plt.figure(figsize=(6,4))
    for hr, idx in zip(snap_hrs, snap_idx):
        plt.plot(r*1e3, field[idx], label=f"{hr} h")
    plt.xlabel("r (mm)")
    plt.ylabel(name)
    plt.title(f"1D {name}")
    if vmax is not None: plt.ylim(0, vmax)
    plt.legend(); plt.grid(); plt.tight_layout(); plt.show()

plot_1d(C_A, "Drug A conc (µg/mL)", vmax=Cmax_A)
if mode!='mono':
    plot_1d(C_B, "Drug B conc (µg/mL)", vmax=Cmax_B)
plot_1d(B,   "Bacterial density",       vmax=K)

# 2D slices (imshow)
Nx = 300
x  = np.linspace(-R, R, Nx)
y  = np.linspace(-R, R, Nx)
X, Y = np.meshgrid(x, y)
Rxy  = np.sqrt(X**2+Y**2)

def plot_2d(arr, name, vmax):
    fig, axs = plt.subplots(1,6,figsize=(15,3), constrained_layout=True)
    for ax, hr, idx in zip(axs, snap_hrs, snap_idx):
        Z = np.zeros_like(Rxy)
        mask = Rxy<=R
        Z[mask] = np.interp(Rxy[mask], r, arr[idx])
        im = ax.imshow(Z, origin='lower',
                       extent=[-R*1e3,R*1e3,-R*1e3,R*1e3],
                       vmin=0,vmax=vmax,cmap='viridis', interpolation='bilinear')
        ax.set_title(f"{hr} h"); ax.axis('off')
    cbar = fig.colorbar(im, ax=axs, orientation='horizontal', pad=0.05)
    cbar.set_label(name)
    plt.show()

plot_2d(C_A, "Drug A conc (µg/mL)", Cmax_A)
if mode!='mono':
    plot_2d(C_B, "Drug B conc (µg/mL)", Cmax_B)
plot_2d(B,   "Bacterial density",       K)

# 3D scatter
Ng = 2000
u, v = np.random.rand(Ng), np.random.rand(Ng)
theta = 2*np.pi*u
phi   = np.arccos(2*v-1)
r_samp= np.random.rand(Ng)**(1/3)*R
X3 = r_samp*np.sin(phi)*np.cos(theta)
Y3 = r_samp*np.sin(phi)*np.sin(theta)
Z3 = r_samp*np.cos(phi)

def plot_3d(arr, name, vmax):
    fig = plt.figure(figsize=(15,3))
    for i,(hr,idx) in enumerate(zip(snap_hrs, snap_idx)):
        ax = fig.add_subplot(1,6,i+1,projection='3d')
        col = np.interp(r_samp, r, arr[idx])
        sc = ax.scatter(X3*1e3, Y3*1e3, Z3*1e3,
                        c=col, s=4, vmin=0, vmax=vmax,
                        cmap='viridis', alpha=0.6)
        ax.set_title(f"{hr} h")
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    cbar = fig.colorbar(sc, ax=fig.axes, orientation='vertical', shrink=0.6, pad=0.02)
    cbar.set_label(name)
    plt.show()

plot_3d(C_A, "Drug A conc (µg/mL)", Cmax_A)
if mode!='mono':
    plot_3d(C_B, "Drug B conc (µg/mL)", Cmax_B)
plot_3d(B,   "Bacterial density",       K)


# Total bacterial load B(t) over time (integrated radially)
B_total = 4 * np.pi * np.trapz(B * r[None, :]**2, r, axis=1)

# Plot total bacterial load over simulation time
plt.figure(figsize=(8, 4))
plt.plot(times / 3600, B_total, color='blue', lw=2)
plt.xlabel("Time (hours)")
plt.ylabel("Total Bacterial Load")
plt.title("Total Bacterial Load over Simulation Time")
plt.grid(True)
plt.tight_layout()
plt.show()


# =============================================================================
# 8) SPATIAL HETEROGENEITY: Standard Deviation of B(r,t)
# =============================================================================
# Radial volume-weighted average
B_mean = np.trapz(B * r[None, :]**2, r, axis=1) / np.trapz(r**2, r)

# Variance and standard deviation
B_var = np.trapz((B - B_mean[:, None])**2 * r[None, :]**2, r, axis=1) / np.trapz(r**2, r)
B_std = np.sqrt(B_var)

# Plot: Spatial standard deviation over time
plt.figure(figsize=(8, 4))
plt.plot(times / 3600, B_std, color='darkorange', lw=2)
plt.xlabel("Time (hours)")
plt.ylabel("Spatial Std. Dev. of B(r,t)")
plt.title("Spatial Heterogeneity of Bacterial Load Over Time")
plt.grid(True)
plt.tight_layout()
plt.show()



import numpy as np
import matplotlib.pyplot as plt

# Assumes these variables are defined:
# C_A: (Nt, Nr+1) array of Drug A concentration over time and radius
# C_B: (Nt, Nr+1) array of Drug B concentration over time and radius
# times: (Nt,) time vector in seconds
# r: (Nr+1,) radial positions in meters
# EC50_A = 8, EC50_B = 0.05 (µg/mL)

# Thresholds (can also use MIC_A or MIC_B instead)
threshold_A = EC50_A  # µg/mL
threshold_B = EC50_B  # µg/mL

# Compute time above threshold at each radial depth
dt = times[1] - times[0]  # time step in seconds

# Boolean masks: True where concentration exceeds threshold
mask_A = C_A > threshold_A
mask_B = C_B > threshold_B

# Time above threshold in hours at each radius
time_above_A = np.sum(mask_A, axis=0) * dt / 3600
time_above_B = np.sum(mask_B, axis=0) * dt / 3600

# Plot
plt.figure(figsize=(7, 4))
plt.plot(r * 1e3, time_above_A, label="Moxifloxacin (A)", lw=2)
plt.plot(r * 1e3, time_above_B, label="Ciprofloxacin (B)", lw=2)
plt.xlabel("Radial Depth (mm)")
plt.ylabel("Time > EC₅₀ (hours)")
plt.title("Time Above EC₅₀ at Each Radial Position")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



# ── Helper functions ──────────────────────────────────────────────────────────

def compute_mpc_window(C, MIC, MPC):
    return (C >= MIC) & (C <= MPC)

def time_in_mpc_window(mask, dt_hours):
    return np.sum(mask, axis=0) * dt_hours

def spatial_volume_in_window(mask, r_mm):
    dr = np.diff(r_mm)[0]
    shell_vol = 4 * np.pi * r_mm**2 * dr
    return np.sum(mask * shell_vol[None, :], axis=1)

def gini_coefficient_over_time(B):
    def _gini(x):
        n = len(x)
        m = np.mean(x)
        diff_sum = np.sum(np.abs(x[:, None] - x[None, :]))
        return diff_sum / (2 * n**2 * m) if m>0 else 0
    return np.array([_gini(bt) for bt in B])

def bliss_index(muA, muB, muAB):
    expected = muA + muB - muA*muB
    return muAB - expected

def loewe_index(C_A, C_B, EC50_A, EC50_B):
    return C_A/EC50_A + C_B/EC50_B

# ── Compute metrics ──────────────────────────────────────────────────────────

dt_hours = (times[1] - times[0]) / 3600.0
r_mm     = r * 1e3  # convert to mm

# 1) MPC window (example MPC values; adjust as needed)
MPC_A = 16.0  # µg/mL
MPC_B = 1.0   # µg/mL

mask_A_mpc = compute_mpc_window(C_A, EC50_A, MPC_A)
mask_B_mpc = compute_mpc_window(C_B, EC50_B, MPC_B)

time_A_mpc = time_in_mpc_window(mask_A_mpc, dt_hours)
time_B_mpc = time_in_mpc_window(mask_B_mpc, dt_hours)

vol_A_mpc  = spatial_volume_in_window(mask_A_mpc, r_mm)
vol_B_mpc  = spatial_volume_in_window(mask_B_mpc, r_mm)

# 2) Spatial heterogeneity (Gini)
gini_ts = gini_coefficient_over_time(B)

# 3) Synergy/Antagonism
muA  = muA_max * (C_A**h_A) / (C_A**h_A + EC50_A**h_A)
muB  = muB_max * (C_B**h_B) / (C_B**h_B + EC50_B**h_B)
muAB = muA + muB

bliss_ts = bliss_index(muA, muB, muAB)
loewe_ts = loewe_index(C_A, C_B, EC50_A, EC50_B)

# ── Plotting ─────────────────────────────────────────────────────────────────

# Time in MPC window vs radial depth
plt.figure()
plt.plot(r_mm, time_A_mpc, label="Moxi")
plt.plot(r_mm, time_B_mpc, label="Cipro")
plt.xlabel("Radial depth (mm)")
plt.ylabel("Hours in MPC window")
plt.title("Time in Mutant Prevention Window")
plt.legend()
plt.grid()
plt.tight_layout()

# Spatial volume in MPC window over time
plt.figure()
plt.plot(times/3600, vol_A_mpc, label="Moxi")
plt.plot(times/3600, vol_B_mpc, label="Cipro")
plt.xlabel("Time (h)")
plt.ylabel("Volume in MPC window (mm³)")
plt.title("Spatial Volume Within MPC Window")
plt.legend()
plt.grid()
plt.tight_layout()

# Gini coefficient over time
plt.figure()
plt.plot(times/3600, gini_ts)
plt.xlabel("Time (h)")
plt.ylabel("Gini coefficient")
plt.title("Spatial Heterogeneity of Bacterial Load")
plt.grid()
plt.tight_layout()

# Bliss synergy heatmap
plt.figure()
plt.imshow(bliss_ts.T, origin='lower', aspect='auto',
           extent=[0, times.max()/3600, r_mm[0], r_mm[-1]])
plt.xlabel("Time (h)")
plt.ylabel("Radial depth (mm)")
plt.title("")
plt.colorbar(label="Bliss index")
plt.tight_layout()



# Loewe additivity heatmap
plt.figure()
plt.imshow(loewe_ts.T, origin='lower', aspect='auto',
           extent=[0, times.max()/3600, r_mm[0], r_mm[-1]])
plt.xlabel("Time (h)")
plt.ylabel("Radial depth (mm)")
plt.title("Loewe Additivity Index")
plt.colorbar(label="Loewe index")
plt.tight_layout()

plt.show()












