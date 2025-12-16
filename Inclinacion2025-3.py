#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inclination vs dynamical–baryonic discrepancy
No error propagation; linear fit with and without bootstrap.
Plots enhanced to make the slope clearly visible.

Author: Carlos A. Ortiz (base) + modifications.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# ==========================================================
# 1. READ CATALOGUE AND FILTER FLAGS
# ==========================================================
df = pd.read_csv("GS21b_catalog.csv")

# Take the last 4 columns as flag columns (adjust if needed)
flag_cols = list(df.columns[-4:])
print("Flag columns detected:", flag_cols)

# Normalise flags: strip spaces and uppercase
flags_norm = df[flag_cols].apply(
    lambda col: col.astype(str).str.strip().str.upper()
)

# Keep rows that do NOT have 'F' in any of the flag columns
mask_keep = ~flags_norm.eq("F").any(axis=1)

before = len(df)
df = df[mask_keep].copy()
after = len(df)
print(f"Filtered flags: kept {after} / {before} rows (removed {before - after}).")

# ==========================================================
# 2. CONSTANTS AND PARAMETERS
# ==========================================================
G  = 4.302e-6   # kpc (km/s)^2 / Msun
He = 1.33       # helium correction factor on HI

k_Re   = 1.20
k_Ropt = 1.05
k_Rout = 1.00

# Characteristic radii
Re   = df["Re"].astype(float).values              # kpc
Ropt = 1.89 * Re
Rout = 2.95 * Re

# Rotational velocities (km/s)
Ve   = df["Ve"].astype(float).values
Vopt = df["Vopt"].astype(float).values
Vout = df["Vout"].astype(float).values

# Total masses (Msun)
Mstar_tot = df["Mstar"].astype(float).values
MH2_tot   = df["MH2"].astype(float).values
MHI_tot   = df["MHI"].astype(float).values

# Scale radii
RD   = 0.59 * Re                                      # stellar disc scale radius
Rgas = 10.0**df["log(Rgas)"].astype(float).values     # kpc
RH2  = Rgas                                           # molecular gas
RHI  = 2.0 * Rgas                                     # atomic gas extended

# Inclination (degrees)
INC = df["INC"].astype(float).values

# ==========================================================
# 3. MASS FUNCTIONS (NO ERRORS)
# ==========================================================
def frac_enclosed(R, Rs):
    """
    Mass fraction enclosed for an exponential disc:
    f(x) = 1 - (1 + x) exp(-x),  with x = R/Rs.
    """
    x = np.zeros_like(R)
    mask = (Rs > 0)
    x[mask] = R[mask] / Rs[mask]
    return 1.0 - (1.0 + x)*np.exp(-x)

def enclosed_mass(Mtot, R, Rs):
    """M(<R) = f(R/Rs) * Mtot."""
    f_x = frac_enclosed(R, Rs)
    return f_x * Mtot

def dyn_mass(V, R, kappa):
    """M_dyn = kappa * V^2 R / G."""
    return kappa * (V**2) * R / G

def baryonic_mass(Ms, MH2, MHI, He):
    """M_bar = Ms + MH2 + He * MHI."""
    return Ms + MH2 + He*MHI

def fDM_from_masses(Mbar, Mdyn):
    """f_DM = 1 - M_bar / M_dyn."""
    return 1.0 - (Mbar / Mdyn)

# ==========================================================
# 4. ENCLOSED MASSES AT Re, Ropt, Rout
# ==========================================================

# --- Re ---
Mstar_Re = enclosed_mass(Mstar_tot, Re,   RD)
MH2_Re   = enclosed_mass(MH2_tot,   Re,   RH2)
MHI_Re   = enclosed_mass(MHI_tot,   Re,   RHI)

# --- Ropt ---
Mstar_Ropt = enclosed_mass(Mstar_tot, Ropt, RD)
MH2_Ropt   = enclosed_mass(MH2_tot,   Ropt, RH2)
MHI_Ropt   = enclosed_mass(MHI_tot,   Ropt, RHI)

# --- Rout ---
Mstar_Rout = enclosed_mass(Mstar_tot, Rout, RD)
MH2_Rout   = enclosed_mass(MH2_tot,   Rout, RH2)
MHI_Rout   = enclosed_mass(MHI_tot,   Rout, RHI)

# ==========================================================
# 5. M_bar, M_dyn AND f_DM (CENTRAL VALUES)
# ==========================================================
# Baryonic mass
Mbar_Re   = baryonic_mass(Mstar_Re,   MH2_Re,   MHI_Re,   He)
Mbar_Ropt = baryonic_mass(Mstar_Ropt, MH2_Ropt, MHI_Ropt, He)
Mbar_Rout = baryonic_mass(Mstar_Rout, MH2_Rout, MHI_Rout, He)

# Dynamical mass
Mdyn_Re   = dyn_mass(Ve,   Re,   k_Re)
Mdyn_Ropt = dyn_mass(Vopt, Ropt, k_Ropt)
Mdyn_Rout = dyn_mass(Vout, Rout, k_Rout)

# Dark-matter fraction (discrepancy), in fraction and percent
fDM_Re       = fDM_from_masses(Mbar_Re,   Mdyn_Re)
fDM_Ropt     = fDM_from_masses(Mbar_Ropt, Mdyn_Ropt)
fDM_Rout     = fDM_from_masses(Mbar_Rout, Mdyn_Rout)

fDM_Re_pct   = 100.0 * fDM_Re
fDM_Ropt_pct = 100.0 * fDM_Ropt
fDM_Rout_pct = 100.0 * fDM_Rout

# ==========================================================
# 6. BOOTSTRAP (PAIRS)
# ==========================================================
def bootstrap_pairs(x, y, n_boot=10000, seed=1234):
    """
    Pairs bootstrap on (x,y) with replacement.
    Returns arrays of slopes, intercepts, and r values.
    """
    rng = np.random.default_rng(seed)
    n = x.size
    slopes = np.empty(n_boot)
    intercepts = np.empty(n_boot)
    rs = np.empty(n_boot)

    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        xb, yb = x[idx], y[idx]
        slope, intercept, r, p, _ = linregress(xb, yb)
        slopes[b] = slope
        intercepts[b] = intercept
        rs[b] = r

    return slopes, intercepts, rs

def summarize_bootstrap(slopes, intercepts, rs, alpha=0.05, label=""):
    """
    Print median, (1-alpha) confidence intervals, and p_boot
    for the hypothesis slope ≈ 0.
    """
    lo, hi = 100*alpha/2, 100*(1-alpha/2)

    s_med = np.median(slopes)
    s_lo, s_hi = np.percentile(slopes, [lo, hi])

    a_med = np.median(intercepts)
    a_lo, a_hi = np.percentile(intercepts, [lo, hi])

    r_med = np.median(rs)
    r_lo, r_hi = np.percentile(rs, [lo, hi])

    # Two-sided p_boot for slope ≈ 0
    p_boot = 2 * min((slopes <= 0).mean(), (slopes >= 0).mean())

    print(f"\n[{label}] Pairs bootstrap (CI {int((1-alpha)*100)}%):")
    print(f"  slope:     {s_med:.3f}  [{s_lo:.3f}, {s_hi:.3f}]  %/deg")
    print(f"  intercept: {a_med:.2f}  [{a_lo:.2f}, {a_hi:.2f}]  %")
    print(f"  r:         {r_med:.3f}  [{r_lo:.3f}, {r_hi:.3f}]")
    print(f"  p_boot(slope ≈ 0): {p_boot:.3f}")

# ==========================================================
# 7. ANALYSIS + PLOT (SLOPE VISUAL, 0–90° X-AXIS)
# ==========================================================
def analyse_radius_simple(INC, fDM_pct, label, line_color="red"):
    """
    For a given radius:
      - Select physically meaningful points (fDM > 0).
      - Perform standard linear regression (fDM[%] vs INC[deg]).
      - Compute SNR = slope / stderr (only here, not in bootstrap).
      - Run pairs bootstrap to obtain CI and p_boot.
      - Plot:
          * scatter of points,
          * best-fit line,
          * line solid in 25°–75° and dashed outside,
          * x-axis from 0° to 90°,
          * English annotations (slope, SNR, r, p, N).
    """
    mask = (
        np.isfinite(INC) &
        np.isfinite(fDM_pct) &
        (fDM_pct > 5.0)
    )

    x = INC[mask]
    y = fDM_pct[mask]

    if len(x) < 5:
        print(f"[{label}] Too few valid data points after selection (N={len(x)}).")
        return

    # ---------- Standard linear regression ----------
    slope, intercept, r, p, stderr = linregress(x, y)

    # Signal-to-noise ratio for the slope
    SNR = slope / stderr if stderr > 0 else np.nan

    print(f"\n=== {label} (unweighted OLS) ===")
    print(f"  fDM(%) = {intercept:.2f} + {slope:.3f} · INC")
    print(f"  r = {r:.3f}, p = {p:.3e}, N = {len(x)}")
    print(f"  stderr(slope) = {stderr:.3e}")
    print(f"  SNR = slope/stderr = {SNR:.3f} sigma")

    # ---------- Bootstrap by pairs (sin SNR de bootstrap) ----------
    slopes_b, intercepts_b, rs_b = bootstrap_pairs(x, y, n_boot=10000, seed=1234)
    summarize_bootstrap(slopes_b, intercepts_b, rs_b, alpha=0.05, label=label)

    # ---------- Prepare line for full 0–90 deg range ----------
    x_fit = np.linspace(0.0, 90.0, 300)
    y_fit = intercept + slope * x_fit

    # Core range [25°, 75°] in solid line; outside as dashed
    core_mask   = (x_fit >= 25.0) & (x_fit <= 75.0)
    outer_mask  = ~core_mask

    # For y-axis range, use data percentiles to compress a bit
    y_min = np.percentile(y, 5)
    y_max = np.percentile(y, 95)
    padding = 0.15 * (y_max - y_min)

    # ---------- Plot ----------
    plt.figure(figsize=(8,6))

    # Scatter points
    plt.scatter(x, y, s=25, alpha=0.70, color="steelblue",
                label="Galaxies (fDM > 5%)")

    # Best-fit line, dashed outside [25°,75°] and solid inside
    plt.plot(
        x_fit[outer_mask], y_fit[outer_mask],
        linestyle="--",
        color=line_color,
        linewidth=2,
        alpha=0.8,
        label="Linear fit (extrapolated)"
    )
    plt.plot(
        x_fit[core_mask], y_fit[core_mask],
        linestyle="-",
        color=line_color,
        linewidth=3,
        label="Linear fit (25°–75°)"
    )

    # Vertical lines marking the adopted inclination range
    plt.axvline(25.0, color="red", linestyle=":", linewidth=1.5, alpha=0.8)
    plt.axvline(75.0, color="red", linestyle=":", linewidth=1.5, alpha=0.8)

    # Axis labels and title in English
    plt.xlabel("Inclination (degrees)")
    plt.ylabel("Dynamical-to-baryonic fraction (%)")
    plt.title(f"Inclination vs fraction  — {label}")

    # X-axis from 0° to 90°, with grid ticks
    plt.xlim(0.0, 90.0)
    plt.xticks(np.arange(0, 91, 10))

    # Compressed y-range to make the slope visible
    plt.ylim(y_min - padding, y_max + padding)

    plt.grid(True, alpha=0.3)

    # Annotation box with slope, SNR, r, p, N
    text = (
        f"slope = {slope:.3f} %/deg\n"
        #f"SNR = {SNR:.2f}σ\n"
        f"r = {r:.2f}\n"
        f"p = {p:.2e}\n"
        f"N = {len(x)}"
    )
    ax = plt.gca()
    ax.text(
        0.03, 0.97, text,
        transform=ax.transAxes,
        va="top", ha="left",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85)
    )

    plt.legend()
    plt.tight_layout()
    plt.savefig(f"Inclination_vs_fDM_{label}_0to90_bootstrap.png",
                dpi=300, bbox_inches="tight")
    plt.show()

# ==========================================================
# 8. RUN FOR Re, Ropt, Rout
# ==========================================================
analyse_radius_simple(INC, fDM_Re_pct,   label="Re",   line_color="red")
analyse_radius_simple(INC, fDM_Ropt_pct, label="Ropt", line_color="red")
analyse_radius_simple(INC, fDM_Rout_pct, label="Rout", line_color="red")