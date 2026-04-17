from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Define File Paths
ocean_file = Path("/Users/auroraczajkowski/Desktop/SIOC 205 /HW 1/M2_ocean.txt")
estuary_file = Path("/Users/auroraczajkowski/Desktop/SIOC 205 /HW 1/M2_mean.txt")

# Load Data
ocean = np.genfromtxt(ocean_file, delimiter=",", names=True)
estuary = np.genfromtxt(estuary_file, delimiter=",", names=True)

# Convert jday to datetime
base = pd.Timestamp("2008-01-01")

date_ocean = base + pd.to_timedelta(ocean["jday"] - 1, unit="D")
date_estuary = base + pd.to_timedelta(estuary["jday"] - 1, unit="D")

# Helper functions 
def clean_series(time_plot, time_num, values, min_value=None):
    """
    Clean a time series while keeping:
    - time_plot: datetime array for plotting
    - time_num: numeric time array for calculations
    - values: data values
    """
    time_plot = np.asarray(time_plot)
    time_num = np.asarray(time_num, dtype=float)
    values = np.asarray(values, dtype=float)

    mask = np.isfinite(time_num) & np.isfinite(values)
    if min_value is not None:
        mask &= values > min_value

    time_plot = time_plot[mask]
    time_num = time_num[mask]
    values = values[mask]

    order = np.argsort(time_num)
    time_plot = time_plot[order]
    time_num = time_num[order]
    values = values[order]

    keep = np.concatenate(([True], np.diff(time_num) > 0))
    time_plot = time_plot[keep]
    time_num = time_num[keep]
    values = values[keep]

    return time_plot, time_num, values


def harmonic_fit(time_days, signal, constituents):
    """
    Fit a harmonic model:
        signal(t) = a0 + sum[a*cos(wt) + b*sin(wt)]
    using least squares.
    """
    t_hours = (time_days - time_days[0]) * 24.0

    names = list(constituents.keys())
    periods = np.array([constituents[name] for name in names], dtype=float)
    omega = 2.0 * np.pi / periods

    X_cols = [np.ones_like(t_hours)]
    for w in omega:
        X_cols.append(np.cos(w * t_hours))
        X_cols.append(np.sin(w * t_hours))

    X = np.column_stack(X_cols)
    beta, _, _, _ = np.linalg.lstsq(X, signal, rcond=None)
    fit = X @ beta

    results = []
    for i, name in enumerate(names):
        a = beta[1 + 2 * i]
        b = beta[2 + 2 * i]
        amp = np.sqrt(a**2 + b**2)

        phase = np.degrees(np.arctan2(b, a))
        if phase < 0:
            phase += 360.0

        results.append({"name": name, "amp": amp, "phase": phase})

    return results, fit


def print_constituent_table(ocean_results, estuary_results, title="Constituent comparison"):
    print(f"\n{title}")
    print("-" * 84)
    print(
        f"{'Constituent':<12}"
        f"{'Ocean Amp (m)':>15}"
        f"{'Ocean Phase (deg)':>20}"
        f"{'Estuary Amp (m)':>18}"
        f"{'Estuary Phase (deg)':>22}"
    )
    print("-" * 84)

    for o, e in zip(ocean_results, estuary_results):
        print(
            f"{o['name']:<12}"
            f"{o['amp']:15.3f}"
            f"{o['phase']:20.3f}"
            f"{e['amp']:18.3f}"
            f"{e['phase']:22.3f}"
        )


# Clean data
plot_ocean, jday_ocean, depth_ocean = clean_series(
    date_ocean, ocean["jday"], ocean["depth"]
)
plot_estuary, jday_estuary, depth_estuary = clean_series(
    date_estuary, estuary["jday"], estuary["depth"]
)

plot_vel, jday_vel, sAlong = clean_series(
    date_estuary, estuary["jday"], estuary["sAlong"]
)
plot_sal, jday_sal, Smean = clean_series(
    date_estuary, estuary["jday"], estuary["Smean"]
)

# Stage-curve data
mask_stage = (
    np.isfinite(estuary["jday"])
    & np.isfinite(estuary["depth"])
    & np.isfinite(estuary["sAlong"])
)
depth_stage = np.asarray(estuary["depth"][mask_stage], dtype=float)
flow_stage = np.asarray(estuary["sAlong"][mask_stage], dtype=float)

# Figure 1: Water depth versus time (shared x-axis)
fig1, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

axes[0].plot(plot_ocean, depth_ocean, color="steelblue", linewidth=1.5)
axes[0].axvspan(
    plot_estuary.min(),
    plot_estuary.max(),
    color="gray",
    alpha=0.2,
    label="Estuary time window",
)
axes[0].set_title("Ocean Water Depth")
axes[0].set_ylabel("Depth (m)")
axes[0].grid(True, alpha=0.3)
axes[0].legend()

axes[1].plot(plot_estuary, depth_estuary, color="hotpink", linewidth=1.5)
axes[1].set_title("Estuary Water Depth")
axes[1].set_xlabel("Date")
axes[1].set_ylabel("Depth (m)")
axes[1].grid(True, alpha=0.3)

# Base harmonic analysis
depth_ocean_dm = depth_ocean - np.nanmean(depth_ocean)

mask_est_tide = depth_estuary > 1.0
plot_estuary_tide = plot_estuary[mask_est_tide]
jday_estuary_tide = jday_estuary[mask_est_tide]
depth_estuary_tide = depth_estuary[mask_est_tide]
depth_estuary_dm = depth_estuary_tide - np.nanmean(depth_estuary_tide)

constituents = {
    "M2": 12.4206,
    "S2": 12.0000,
    "N2": 12.6583,
    "K1": 23.9345,
    "O1": 25.8193,
}

ocean_results, ocean_fit = harmonic_fit(jday_ocean, depth_ocean_dm, constituents)
estuary_results, estuary_fit = harmonic_fit(jday_estuary_tide, depth_estuary_dm, constituents)

print_constituent_table(ocean_results, estuary_results)

# Figure 2: Tidal constituent amplitudes
names = [r["name"] for r in ocean_results]
ocean_amp = np.array([r["amp"] for r in ocean_results], dtype=float)
estuary_amp = np.array([r["amp"] for r in estuary_results], dtype=float)

x = np.arange(len(names))
width = 0.38

fig2, ax = plt.subplots(figsize=(10, 6))
ax.bar(x - width / 2, ocean_amp, width, color="steelblue", label="Ocean")
ax.bar(x + width / 2, estuary_amp, width, color="hotpink", label="Estuary")

ax.set_title("Tidal Constituent Amplitudes")
ax.set_xlabel("Constituent")
ax.set_ylabel("Amplitude (m)")
ax.set_xticks(x)
ax.set_xticklabels(names)
ax.grid(True, axis="y", alpha=0.3)
ax.legend()
# ax.set_yscale("log")

# Figure 3: Estuary velocity and salinity versus time
fig3, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

axes[0].plot(plot_vel, sAlong, color="purple", linewidth=1.5)
axes[0].set_title("Estuary Depth-Averaged Along-Stream Velocity")
axes[0].set_ylabel("Velocity (m/s)")
axes[0].grid(True, alpha=0.3)

axes[1].plot(plot_sal, Smean, color="teal", linewidth=1.5)
axes[1].set_title("Estuary Depth-Averaged Salinity")
axes[1].set_xlabel("Date")
axes[1].set_ylabel("Salinity")
axes[1].grid(True, alpha=0.3)

# Figure 4: Stage curve (flow vs tidal elevation)
fig4, ax = plt.subplots(figsize=(8, 6))
ax.scatter(flow_stage, depth_stage, s=10, alpha=0.5, color="mediumorchid")
ax.set_title("Estuary Flow vs Tidal Elevation")
ax.set_xlabel("Flow (m/s)")
ax.set_ylabel("Depth (m)")
ax.grid(True, alpha=0.3)

# Figure 5: Estuary - Ocean residual and overlay
ocean_series = pd.Series(ocean["depth"], index=pd.DatetimeIndex(date_ocean)).sort_index()
estuary_series = pd.Series(estuary["depth"], index=pd.DatetimeIndex(date_estuary)).sort_index()

ocean_interp = ocean_series.reindex(estuary_series.index).interpolate(method="time")
residual = estuary_series - ocean_interp

fig5, ax = plt.subplots(figsize=(10, 6))
ax.plot(estuary_series.index, estuary_series.values, color="hotpink", linewidth=1.5, label="Estuary")
ax.plot(ocean_interp.index, ocean_interp.values, color="steelblue", linewidth=1.5, label="Ocean")
ax.plot(residual.index, residual.values, color="black", linestyle="--", linewidth=1.5, label="Estuary - Ocean")

ax.set_title("Estuary vs Ocean Water Level and Residual")
ax.set_xlabel("Date")
ax.set_ylabel("Depth (m)")
ax.grid(True, alpha=0.3)
ax.legend()

# Figure 6: Observed vs harmonic fit
fig6, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

axes[0].plot(plot_ocean, depth_ocean_dm, color="steelblue", linewidth=1.2, label="Observed")
axes[0].plot(plot_ocean, ocean_fit, "k--", linewidth=1.5, label="Harmonic fit")
axes[0].axvspan(
    plot_estuary_tide.min(),
    plot_estuary_tide.max(),
    color="gray",
    alpha=0.2,
    label="Estuary time window",
)
axes[0].set_title("Ocean: Demeaned Water Depth and Harmonic Fit")
axes[0].set_ylabel("Depth anomaly (m)")
axes[0].grid(True, alpha=0.3)
axes[0].legend()

axes[1].plot(plot_estuary_tide, depth_estuary_dm, color="hotpink", linewidth=1.2, label="Observed")
axes[1].plot(plot_estuary_tide, estuary_fit, "k--", linewidth=1.5, label="Harmonic fit")
axes[1].set_title("Estuary: Demeaned Water Depth and Harmonic Fit")
axes[1].set_xlabel("Date")
axes[1].set_ylabel("Depth anomaly (m)")
axes[1].grid(True, alpha=0.3)
axes[1].legend()

# Figure 7: Residual only (Estuary - Ocean)
fig7, ax = plt.subplots(figsize=(10, 6))
ax.plot(residual.index, residual.values, color="black", linewidth=1.5)
ax.axhline(0, color="gray", linestyle="--", linewidth=1)
ax.set_title("Residual Water Level (Estuary - Ocean)")
ax.set_xlabel("Date")
ax.set_ylabel("Depth Difference (m)")
ax.grid(True, alpha=0.3)

# Figure 8: Predicted water levels from harmonic analysis
ocean_pred = ocean_fit + np.nanmean(depth_ocean)
estuary_pred = estuary_fit + np.nanmean(depth_estuary_tide)

fig8, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

axes[0].plot(plot_ocean, depth_ocean, color="steelblue", linewidth=1.2, label="Observed")
axes[0].plot(plot_ocean, ocean_pred, "k--", linewidth=1.5, label="Predicted")
axes[0].set_title("Ocean Water Level: Observed vs Predicted")
axes[0].set_ylabel("Depth (m)")
axes[0].grid(True, alpha=0.3)
axes[0].legend()

axes[1].plot(plot_estuary_tide, depth_estuary_tide, color="hotpink", linewidth=1.2, label="Observed")
axes[1].plot(plot_estuary_tide, estuary_pred, "k--", linewidth=1.5, label="Predicted")
axes[1].set_title("Estuary Water Level: Observed vs Predicted")
axes[1].set_xlabel("Date")
axes[1].set_ylabel("Depth (m)")
axes[1].grid(True, alpha=0.3)
axes[1].legend()

# Figure 9: Prediction residuals and RMSE
ocean_residual_pred = depth_ocean - ocean_pred
estuary_residual_pred = depth_estuary_tide - estuary_pred

ocean_rmse = np.sqrt(np.nanmean(ocean_residual_pred**2))
estuary_rmse = np.sqrt(np.nanmean(estuary_residual_pred**2))

print("\nPrediction error summary")
print("-" * 40)
print(f"Ocean RMSE:   {ocean_rmse:.3f} m")
print(f"Estuary RMSE: {estuary_rmse:.3f} m")

fig9, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

axes[0].plot(plot_ocean, ocean_residual_pred, color="steelblue", linewidth=1.2)
axes[0].axhline(0, color="gray", linestyle="--", linewidth=1)
axes[0].set_title("Ocean Prediction Residual (Observed - Predicted)")
axes[0].set_ylabel("Residual (m)")
axes[0].grid(True, alpha=0.3)

axes[1].plot(plot_estuary_tide, estuary_residual_pred, color="hotpink", linewidth=1.2)
axes[1].axhline(0, color="gray", linestyle="--", linewidth=1)
axes[1].set_title("Estuary Prediction Residual (Observed - Predicted)")
axes[1].set_xlabel("Date")
axes[1].set_ylabel("Residual (m)")
axes[1].grid(True, alpha=0.3)

# Figure 10: Add M4 to the water-level constituent analysis
constituents_m4 = {
    "M2": 12.4206,
    "S2": 12.0000,
    "N2": 12.6583,
    "K1": 23.9345,
    "O1": 25.8193,
    "M4": 6.2103,
}

ocean_results_m4, ocean_fit_m4 = harmonic_fit(jday_ocean, depth_ocean_dm, constituents_m4)
estuary_results_m4, estuary_fit_m4 = harmonic_fit(jday_estuary_tide, depth_estuary_dm, constituents_m4)

print_constituent_table(ocean_results_m4, estuary_results_m4, title="Constituent comparison including M4")

names_m4 = [r["name"] for r in ocean_results_m4]
ocean_amp_m4 = np.array([r["amp"] for r in ocean_results_m4], dtype=float)
estuary_amp_m4 = np.array([r["amp"] for r in estuary_results_m4], dtype=float)

x_m4 = np.arange(len(names_m4))

fig10, ax = plt.subplots(figsize=(10, 6))
ax.bar(x_m4 - width / 2, ocean_amp_m4, width, color="steelblue", label="Ocean")
ax.bar(x_m4 + width / 2, estuary_amp_m4, width, color="hotpink", label="Estuary")
ax.set_title("Tidal Constituent Amplitudes Including M4")
ax.set_xlabel("Constituent")
ax.set_ylabel("Amplitude (m)")
ax.set_xticks(x_m4)
ax.set_xticklabels(names_m4)
ax.grid(True, axis="y", alpha=0.3)
ax.legend()

# Figure 11: Compare estuary and ocean depth-averaged flow
if "sAlong" in ocean.dtype.names:
    plot_ocean_vel, jday_ocean_vel, ocean_sAlong = clean_series(
        date_ocean, ocean["jday"], ocean["sAlong"]
    )

    fig11, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    axes[0].plot(plot_ocean_vel, ocean_sAlong, color="steelblue", linewidth=1.2)
    axes[0].set_title("Ocean Depth-Averaged Along-Stream Velocity")
    axes[0].set_ylabel("Velocity (m/s)")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(plot_vel, sAlong, color="hotpink", linewidth=1.2)
    axes[1].set_title("Estuary Depth-Averaged Along-Stream Velocity")
    axes[1].set_xlabel("Date")
    axes[1].set_ylabel("Velocity (m/s)")
    axes[1].grid(True, alpha=0.3)
else:
    print("\nOcean file does not contain 'sAlong', so the ocean-flow comparison extra cannot be run with this file.")

# Figure 12: Compare stage curves for ocean and estuary
if "sAlong" in ocean.dtype.names:
    mask_stage_ocean = (
        np.isfinite(ocean["jday"])
        & np.isfinite(ocean["depth"])
        & np.isfinite(ocean["sAlong"])
    )
    ocean_depth_stage = np.asarray(ocean["depth"][mask_stage_ocean], dtype=float)
    ocean_flow_stage = np.asarray(ocean["sAlong"][mask_stage_ocean], dtype=float)

    fig12, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    axes[0].scatter(ocean_depth_stage, ocean_flow_stage, s=10, alpha=0.5, color="steelblue")
    axes[0].set_title("Ocean Flow vs Tidal Elevation")
    axes[0].set_xlabel("Depth (m)")
    axes[0].set_ylabel("sAlong (m/s)")
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(depth_stage, flow_stage, s=10, alpha=0.5, color="mediumorchid")
    axes[1].set_title("Estuary Flow vs Tidal Elevation")
    axes[1].set_xlabel("Depth (m)")
    axes[1].grid(True, alpha=0.3)
else:
    print("\nOcean file does not contain 'sAlong', so the ocean stage-curve comparison extra cannot be run with this file.")

# Figure 13: Harmonic analysis on velocity
velocity_constituents = {
    "M2": 12.4206,
    "S2": 12.0000,
    "N2": 12.6583,
    "K1": 23.9345,
    "O1": 25.8193,
    "M4": 6.2103,
}

sAlong_dm = sAlong - np.nanmean(sAlong)
estuary_vel_results, estuary_vel_fit = harmonic_fit(jday_vel, sAlong_dm, velocity_constituents)

print("\nEstuary velocity constituent analysis")
print("-" * 60)
for row in estuary_vel_results:
    print(f"{row['name']:<8} Amp = {row['amp']:.3f} m/s   Phase = {row['phase']:.3f} deg")

fig13, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

axes[0].plot(plot_vel, sAlong_dm, color="purple", linewidth=1.2, label="Observed velocity anomaly")
axes[0].plot(plot_vel, estuary_vel_fit, "k--", linewidth=1.5, label="Harmonic fit")
axes[0].set_title("Estuary Velocity: Observed vs Harmonic Fit")
axes[0].set_ylabel("Velocity anomaly (m/s)")
axes[0].grid(True, alpha=0.3)
axes[0].legend()

if "sAlong" in ocean.dtype.names:
    plot_ocean_vel, jday_ocean_vel, ocean_sAlong = clean_series(
        date_ocean, ocean["jday"], ocean["sAlong"]
    )
    ocean_sAlong_dm = ocean_sAlong - np.nanmean(ocean_sAlong)
    ocean_vel_results, ocean_vel_fit = harmonic_fit(jday_ocean_vel, ocean_sAlong_dm, velocity_constituents)

    print("\nOcean velocity constituent analysis")
    print("-" * 60)
    for row in ocean_vel_results:
        print(f"{row['name']:<8} Amp = {row['amp']:.3f} m/s   Phase = {row['phase']:.3f} deg")

    names_vel = [r["name"] for r in ocean_vel_results]
    ocean_vel_amp = np.array([r["amp"] for r in ocean_vel_results], dtype=float)
    estuary_vel_amp = np.array([r["amp"] for r in estuary_vel_results], dtype=float)

    x_vel = np.arange(len(names_vel))

    fig14, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x_vel - width / 2, ocean_vel_amp, width, color="steelblue", label="Ocean velocity")
    ax.bar(x_vel + width / 2, estuary_vel_amp, width, color="hotpink", label="Estuary velocity")
    ax.set_title("Velocity Constituent Amplitudes")
    ax.set_xlabel("Constituent")
    ax.set_ylabel("Amplitude (m/s)")
    ax.set_xticks(x_vel)
    ax.set_xticklabels(names_vel)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    axes[1].plot(plot_ocean_vel, ocean_sAlong_dm, color="steelblue", linewidth=1.2, label="Observed velocity anomaly")
    axes[1].plot(plot_ocean_vel, ocean_vel_fit, "k--", linewidth=1.5, label="Harmonic fit")
    axes[1].set_title("Ocean Velocity: Observed vs Harmonic Fit")
    axes[1].set_xlabel("Date")
    axes[1].set_ylabel("Velocity anomaly (m/s)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
else:
    axes[1].axis("off")
    axes[1].text(
        0.5, 0.5,
        "Ocean file does not contain sAlong,\nso ocean velocity analysis is unavailable.",
        ha="center", va="center", transform=axes[1].transAxes
    )

# Show plots
plt.show()