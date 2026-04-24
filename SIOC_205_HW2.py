
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat


# settings
DATA_DIR = Path(".")
ADCP_FILE = DATA_DIR / "/Users/auroraczajkowski/Desktop/SIOC 205 /HW 2/exchangeData/CC_adcp.mat"
CTD_FILE = DATA_DIR / "/Users/auroraczajkowski/Desktop/SIOC 205 /HW 2/exchangeData/CW_ctds.mat"

SAVE_FIGURES = True
FIG_DIR = DATA_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)


# helpers
def matlab_datenum_to_datetime(dn):
    """
    Convert MATLAB datenum to pandas datetime.
    """
    dn = np.asarray(dn, dtype=float).ravel()
    days = np.floor(dn).astype(int)
    frac = dn - days
    out = [
        pd.Timestamp.fromordinal(int(d)) + pd.to_timedelta(f, unit="D") - pd.Timedelta(days=366)
        for d, f in zip(days, frac)
    ]
    return pd.to_datetime(out)


def year_day_to_datetime(jday, year=2008):
    """
    Convert year day to pandas datetime.
    jday=1 means Jan 1 of that year.
    """
    jday = np.asarray(jday, dtype=float).ravel()
    t0 = pd.Timestamp(f"{year}-01-01")
    return t0 + pd.to_timedelta(jday - 1, unit="D")


def godin_filter_1d(x, dt_hours=1 / 6):
    """
    Godin filter:
    three successive running means of 24h, 24h, 25h.
    For 10-min data dt_hours = 1/6, so 24h=144 samples, 25h=150.
    """
    x = np.asarray(x, dtype=float)

    n24 = int(round(24 / dt_hours))
    n25 = int(round(25 / dt_hours))

    s = pd.Series(x)
    y = s.rolling(n24, center=True, min_periods=n24 // 2).mean()
    y = y.rolling(n24, center=True, min_periods=n24 // 2).mean()
    y = y.rolling(n25, center=True, min_periods=n25 // 2).mean()

    return y.to_numpy()


def godin_filter_2d(arr, dt_hours=1 / 6):
    """
    Apply Godin filter along time axis for array shaped (nz, nt)
    """
    arr = np.asarray(arr, dtype=float)
    out = np.full_like(arr, np.nan)
    for k in range(arr.shape[0]):
        out[k, :] = godin_filter_1d(arr[k, :], dt_hours=dt_hours)
    return out


def running_nanrange(x, window):
    s = pd.Series(np.asarray(x, dtype=float))
    rmax = s.rolling(window, center=True, min_periods=max(3, window // 2)).max()
    rmin = s.rolling(window, center=True, min_periods=max(3, window // 2)).min()
    return (rmax - rmin).to_numpy()


# spring/neap window selection 
def choose_spring_neap_windows(time, water_level, window_hours=25, search_hours=24, edge_buffer_hours=36):
    """
    Choose one spring and one neap window from tidal range, while avoiding:
      - edge artifacts from rolling windows
      - incomplete windows at the start/end of the record
      - spring and neap windows overlapping each other

    Returns
    -------
    spring_slice, neap_slice, tide_range
    """
    time = pd.to_datetime(time)
    water_level = np.asarray(water_level, dtype=float)

    dt_hours = np.median(np.diff(time).astype("timedelta64[s]").astype(float)) / 3600.0
    range_window = int(round(search_hours / dt_hours))
    plot_window = int(round(window_hours / dt_hours))
    edge_buffer = int(round(edge_buffer_hours / dt_hours))

    tide_range = running_nanrange(water_level, range_window)

    n = len(time)
    valid = np.isfinite(tide_range)

    # Exclude beginning/end where rolling windows and plotting windows are incomplete
    valid[:edge_buffer] = False
    valid[-edge_buffer:] = False

    if not np.any(valid):
        raise RuntimeError("No valid points left after excluding edge effects.")

    valid_idx = np.where(valid)[0]

    # Pick spring = max tidal range in valid interior
    i_spring = valid_idx[np.nanargmax(tide_range[valid_idx])]

    # For neap, exclude region near spring so they do not overlap
    exclusion_halfwidth = plot_window
    valid_neap = valid.copy()
    i1 = max(0, i_spring - exclusion_halfwidth)
    i2 = min(n, i_spring + exclusion_halfwidth + 1)
    valid_neap[i1:i2] = False

    valid_neap_idx = np.where(valid_neap & np.isfinite(tide_range))[0]
    if len(valid_neap_idx) == 0:
        raise RuntimeError("Could not find a non-overlapping neap window.")

    i_neap = valid_neap_idx[np.nanargmin(tide_range[valid_neap_idx])]

    def make_slice(i0):
        start = i0 - plot_window // 2
        stop = start + plot_window
        start = max(0, start)
        stop = min(n, stop)

        # force exact window length when possible
        if stop - start < plot_window:
            if start == 0:
                stop = min(n, plot_window)
            elif stop == n:
                start = max(0, n - plot_window)

        return slice(start, stop)

    spring_slice = make_slice(i_spring)
    neap_slice = make_slice(i_neap)

    return spring_slice, neap_slice, tide_range


# Load MATLAB files 
adcp_mat = loadmat(ADCP_FILE, squeeze_me=True, struct_as_record=False)
ctd_mat = loadmat(CTD_FILE, squeeze_me=True, struct_as_record=False)

adcpmeanALL = adcp_mat["adcpmeanALL"]
CW = ctd_mat["CW"]

# ADCP variables
timePST = np.asarray(adcpmeanALL.timePST).ravel()
time_adcp = matlab_datenum_to_datetime(timePST)

jdayADCP = np.asarray(adcp_mat["jdayADCP"]).ravel()
wlADCP = np.asarray(adcp_mat["wlADCP"]).ravel()

depth = np.asarray(adcpmeanALL.depth).ravel()
upperrange = np.asarray(adcpmeanALL.upperrange).ravel()
sAlong = np.asarray(adcpmeanALL.sAlong)          # (70, 4032)

hsigma = np.asarray(adcp_mat["hsigma"]).ravel()
zcoords = np.asarray(adcp_mat["zcoords"])        # (51, 4032)
s_sigma = np.asarray(adcp_mat["s_sigma"])        # (51, 4032)

# CTD variables
jday_ctd = np.asarray(ctd_mat["jday"]).ravel()
time_ctd = year_day_to_datetime(jday_ctd, year=2008)

cw_depth = np.asarray(CW.depth)                  # (3, 4033)
cw_density = np.asarray(CW.density)              # (3, 4033)

# Handout formula for depth above bed:
# CW.depthAboveBed = -(CW.depth' - CW.depth(end,:)' - 0.5)
# In Python, keeping shape as (time, 3):
depthAboveBed = -(cw_depth.T - cw_depth.T[:, [-1]] - 0.5)  # (4033, 3)

# Also available directly in file, but we recompute to match instructions
# depthAboveBed_file = np.asarray(CW.depthAboveBed)         # (4033, 3)

print("Loaded ADCP and CTD MATLAB files.")
print(f"sAlong shape   = {sAlong.shape}")
print(f"s_sigma shape  = {s_sigma.shape}")
print(f"density shape  = {cw_density.shape}")


# Figure 1: Plot tidal circulation over depth and time
fig1, ax = plt.subplots(figsize=(12, 5))

pcm = ax.pcolormesh(
    time_adcp,
    upperrange,
    sAlong,
    shading="auto"
)
plt.colorbar(pcm, ax=ax, label="Along-channel velocity (m s$^{-1}$)")
ax.plot(time_adcp, depth, color="k", lw=1.2, label="Water depth")
ax.set_xlabel("Time")
ax.set_ylabel("Height above bed (m)")
ax.set_title("Tidal circulation over depth and time (CC ADCP)")
ax.legend(loc="upper right")
fig1.tight_layout()

if SAVE_FIGURES:
    fig1.savefig(FIG_DIR / "01_tidal_circulation_depth_time_mat.png", dpi=200)


# Figure 2: Residual circulation profile + mean density profile
# Use sigma-coordinate velocity for time-mean residual.
u_residual_sigma = np.nanmean(s_sigma, axis=1)   # mean over time, keep vertical structure

# Mean density profile from 3 CTDs
rho_mean = np.nanmean(cw_density, axis=1)        # 3 points
z_ctd_mean = np.nanmean(depthAboveBed, axis=0)   # 3 points

fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))

ax1.plot(u_residual_sigma, hsigma, marker="o")
ax1.axvline(0, color="k", lw=1)
ax1.set_xlabel("Residual along-channel velocity (m s$^{-1}$)")
ax1.set_ylabel("Sigma coordinate")
ax1.set_title("Mean residual circulation profile")

order = np.argsort(z_ctd_mean)
ax2.plot(rho_mean[order], z_ctd_mean[order], marker="o")
ax2.set_xlabel("Mean density (kg m$^{-3}$)")
ax2.set_ylabel("Height above bed (m)")
ax2.set_title("Mean density profile (CW CTDs)")

fig2.tight_layout()

if SAVE_FIGURES:
    fig2.savefig(FIG_DIR / "02_residual_velocity_density_profiles_mat.png", dpi=200)


# Figure 3: Spring vs neap residual circulation

spring_slice, neap_slice, tide_range = choose_spring_neap_windows(
    time_adcp,
    wlADCP,
    window_hours=25,       # about 2 tidal cycles
    search_hours=24,
    edge_buffer_hours=36   # keeps windows away from record edges
)

# Average over time within each window
u_spring_sigma = np.nanmean(s_sigma[:, spring_slice], axis=1)
u_neap_sigma = np.nanmean(s_sigma[:, neap_slice], axis=1)

t_spring = time_adcp[spring_slice]
t_neap = time_adcp[neap_slice]

# Sanity check
print("Spring window:", t_spring[0], "to", t_spring[-1], "npts =", len(t_spring))
print("Neap window  :", t_neap[0], "to", t_neap[-1], "npts =", len(t_neap))
print("Spring/Neap windows overlap?:", not (t_spring[-1] < t_neap[0] or t_neap[-1] < t_spring[0]))

fig3, ax = plt.subplots(figsize=(6, 6))
ax.plot(
    u_spring_sigma,
    hsigma,
    lw=2,
    label=f"Spring (~{t_spring[0].strftime('%Y-%m-%d')})"
)
ax.plot(
    u_neap_sigma,
    hsigma,
    lw=2,
    label=f"Neap (~{t_neap[0].strftime('%Y-%m-%d')})"
)
ax.axvline(0, color="k", lw=1)
ax.set_xlabel("Residual along-channel velocity (m s$^{-1}$)")
ax.set_ylabel("Sigma coordinate")
ax.set_title("Residual circulation: spring vs neap")
ax.legend()
fig3.tight_layout()

if SAVE_FIGURES:
    fig3.savefig(FIG_DIR / "03_spring_vs_neap_residual_profiles_mat.png", dpi=200)


# Optional: show how those windows were selected
fig3b, ax = plt.subplots(figsize=(12, 4))
ax.plot(time_adcp, wlADCP, label="Water level")
ax.plot(time_adcp, tide_range, label="24 h tidal range proxy")
ax.axvspan(t_spring[0], t_spring[-1], alpha=0.25, label="Spring window")
ax.axvspan(t_neap[0], t_neap[-1], alpha=0.25, label="Neap window")
ax.set_xlabel("Time")
ax.set_title("Selected spring and neap windows")
ax.legend()
fig3b.tight_layout()

if SAVE_FIGURES:
    fig3b.savefig(FIG_DIR / "03b_spring_neap_windows_mat.png", dpi=200)

# Extra: Godin-filtered residual over time
dt_hours = np.median(np.diff(time_adcp).astype("timedelta64[s]").astype(float)) / 3600.0
u_godin_sigma = godin_filter_2d(s_sigma, dt_hours=dt_hours)

fig4, ax = plt.subplots(figsize=(12, 5))
pcm = ax.pcolormesh(
    time_adcp,
    hsigma,
    u_godin_sigma,
    shading="auto"
)
plt.colorbar(pcm, ax=ax, label="Godin-filtered along-channel velocity (m s$^{-1}$)")
ax.set_xlabel("Time")
ax.set_ylabel("Sigma coordinate")
ax.set_title("Subtidal residual circulation over time (Godin filtered)")
fig4.tight_layout()

if SAVE_FIGURES:
    fig4.savefig(FIG_DIR / "04_godin_filtered_residual_mat.png", dpi=200)


# Useful printouts 
print("\nSpring window:")
print(f"  start = {t_spring[0]}")
print(f"  end   = {t_spring[-1]}")

print("\nNeap window:")
print(f"  start = {t_neap[0]}")
print(f"  end   = {t_neap[-1]}")

print("\nMean density profile:")
for z_i, rho_i in zip(z_ctd_mean[order], rho_mean[order]):
    print(f"  z_above_bed = {z_i:6.2f} m, rho = {rho_i:8.3f} kg/m^3")

plt.show()