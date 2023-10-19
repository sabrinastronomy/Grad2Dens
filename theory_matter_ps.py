"""
Generate theoretical matter power spectrum
Written by Adélie Gorce
"""
import numpy as np
import camb
from camb import model

# SETTINGS
nonlinear = False
kmax = 1e3
z_max = 15.
little_h = False

# COSMOLOGY
h = 0.6774000
H0 = h * 100.
Om_0 = 0.309
Ol_0 = 0.691
Ob_0 = 0.049
obh2 = Ob_0 * h**2
och2 = (Om_0 - Ob_0) * h**2
A_s = 2.139e-9
n_s = 0.9677
T_CMB = 2.7260  # K

def generate_matter_pspec(z_max, kmax=kmax, little_h=little_h, nonlinear=nonlinear):
    pars = camb.CAMBparams()
    pars.set_cosmology(
        H0=H0, ombh2=obh2, omch2=och2, TCMB=T_CMB
    )
    pars.InitPower.set_params(ns=n_s, r=0, As=A_s)
    pars.WantTransfer = True
    pars.set_dark_energy()

    data = camb.get_background(pars)
    results = camb.get_results(pars)

    interp_l = camb.get_matter_power_interpolator(
        pars,
        nonlinear=nonlinear,
        kmax=kmax,
        hubble_units=little_h,
        k_hunit=little_h,
        zmax=z_max,
        var1=model.Transfer_nonu,
        var2=model.Transfer_nonu,
    )