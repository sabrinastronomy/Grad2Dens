import numpy as np
import camb
from camb import model
import matplotlib.pyplot as plt
import powerbox as pbox

# SETTINGS (see below for usage)
nonlinear = False
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
tau = 0.054 # thomson/reionization optical depth
z_max = 15 # highest z value to calculate pspec at

def p_spec_normal(field, nbins, resolution, area):
    """
    square before averaging (histogramming)
    """
    curr_side_length = np.shape(field)[0]
    fft_data = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(field)))
    fft_data_squared = np.abs(fft_data) ** 2
    k_arr = np.fft.fftshift(np.fft.fftfreq(curr_side_length)) * 2 * np.pi
    k1, k2 = np.meshgrid(k_arr, k_arr)
    k_mag_full = np.sqrt(k1 ** 2 + k2 ** 2)

    counts, bin_edges = np.histogram(k_mag_full, nbins)
    binned_power, _ = np.histogram(k_mag_full, nbins, weights=fft_data_squared)
    kvals = 0.5 * (bin_edges[:-1] + bin_edges[1:]) # center of bins
    pspec = binned_power / counts # average power in each bin
    print(f"resolution, {resolution}")
    pspec *= resolution**2 # converting form pixels^2 to Mpc^2
    pspec /= area
    return counts, pspec, kvals

def get_truth_matter_pspec(kmax, side_length, z):
    # Initialise CAMB object
    pars = camb.CAMBparams()
    pars.set_cosmology(
        H0=H0, ombh2=obh2, omch2=och2, TCMB=T_CMB, tau=tau
    )
    pars.InitPower.set_params(ns=n_s, r=0, As=A_s)
    pars.WantTransfer = True
    pars.set_dark_energy()
    # Compute everything you need for the power spectrum
    data = camb.get_background(pars)
    results = camb.get_results(pars)
    # compute ps and interpolate
    interp_l = camb.get_matter_power_interpolator(
        pars,  # cosmology
        nonlinear=nonlinear,  # linear or non-linear matter PS
        kmax=kmax,  # max k: computations much slower for larger values
        hubble_units=little_h,  # PS in Mpc or h-1 Mpc?
        k_hunit=little_h,  # k modes in Mpc-1 or hMpc-1?
        zmax=z_max,  # max z you're interpolating to
        var1=model.Transfer_nonu,  # which matter components to include. Here, everything but neutrinos
        var2=model.Transfer_nonu,
    )
    pspec_k_func = lambda k: interp_l.P(z, k) / side_length
    return pspec_k_func

def convert_pspec_2_2D(power_spectrum, num_k_modes, z):
    p_spec_2D = np.empty((num_k_modes, num_k_modes))
    kfreq = np.fft.fftshift(np.fft.fftfreq(num_k_modes)) * (2 * np.pi)# 2D frequencies, not yet right scale
    print(np.max(kfreq))
    k1, k2 = np.meshgrid(kfreq, kfreq)
    k_mag_full = np.sqrt(k1 ** 2 + k2 ** 2)
    print("max, min")
    print(np.max(k_mag_full))
    print(np.min(k_mag_full))

    # unbin stuff, I think for loop is fine, since you just do it once
    for i in range(num_k_modes):
        for j in range(num_k_modes):
            val = power_spectrum(z, k_mag_full[i, j])
            p_spec_2D[i, j] = val
    print(np.count_nonzero(p_spec_2D))
    return k_mag_full, p_spec_2D


if __name__ == "__main__":
    larger_side_length = larger_num_k_modes = 256
    # num_k_modes = 128
    # larger_num_k_modes = larger_side_length // 2
    dim = 2
    z = 7

    k_arr = np.fft.fftshift(np.fft.fftfreq(larger_side_length)) * 2 * np.pi
    k1, k2 = np.meshgrid(k_arr, k_arr)
    k_mag_full = np.sqrt(k1 ** 2 + k2 ** 2)

    kmax = 2*np.pi / larger_side_length * larger_num_k_modes
    pspec_k_func = get_truth_matter_pspec(kmax, larger_side_length, z)


    pb = pbox.PowerBox(
        N=larger_side_length // 2,  # number of wavenumbers, and pixels
        dim=dim,  # dimension of box
        pk=pspec_k_func,  # The power-spectrum
        boxlength=larger_side_length,  # Size of the box (sets the units of k in pk)
        seed=1010,  # Use the same seed as our powerbox
        # ensure_physical=True
    )

    resolution = larger_side_length / larger_num_k_modes
    area = larger_side_length**2
    # cut_out = pb.delta_x()[0:side_length, 0:side_length]
    p_k_field, bins_field = pbox.get_power(pb.delta_x(), boxlength=larger_side_length)
    p_spec_cmb = pspec_k_func(bins_field)
    _, pspec_own, kvals_own = p_spec_normal(pb.delta_x(), 128, resolution, area)
    # _, pspec_own_small, kvals_own_small = p_spec_normal(cut_out, 128, resolution, area)

    plt.title("these should all be the same")
    plt.loglog(bins_field, p_k_field, label=rf'powerbox (large box)', c="m")
    plt.loglog(bins_field, p_spec_cmb.squeeze(), label=rf'truth (large box)', ls="--", c="m")
    plt.loglog(kvals_own, pspec_own, label=rf'my pspec (large box)')
    # plt.loglog(kvals_own_small, pspec_own_small, label=rf'my pspec (small box)')

    plt.legend()
    plt.savefig("all_pspec.png")
    # plt.loglog(bins_field, np.abs(p_k_field - pspec_true(z, bins_field).squeeze()), label=rf'difference z={z} CMB')
    plt.close()

    plt.close()
    plt.imshow(pb.delta_x())
    plt.title("truth field")
    plt.colorbar()
    plt.show()
    #
    # cut_out = pb.delta_x()[0:side_length, 0:side_length]
    # p_spec_2d_pbox_full = np.abs(np.fft.fftshift(np.fft.fftn(cut_out)))**2 / side_length**2
    # k_mag_full, p_spec_2D_cmb = convert_pspec_2_2D(pspec_true, side_length, z)
    # kfreq = np.fft.fftshift(np.fft.fftfreq(side_length)) * (2 * np.pi)# 2D frequencies, not yet right scale
    # k1, k2 = np.meshgrid(kfreq, kfreq)
    # p_spec_2d_cmb = np.reshape(p_spec_2D_cmb.flatten(), (side_length, side_length))

    # counts, pspec, kvals = p_spec_normal(p_spec_2d_cmb, 250)
    # counts_func = interpolate.interp1d(kvals, counts, fill_value="extrapolate")
    # weights = counts_func(k_mag_full)
    # p_spec_2d_cmb *= (1 / weights ** 2)
    # weights = k_mag_full

    # plt.imshow(p_spec_2d_pbox_full, norm=matplotlib.colors.LogNorm())
    # plt.title("2d pbox power spec")
    # plt.colorbar()
    # plt.savefig(f"pspec_debug_plots/2d_pspec_pbox_{z}.png")
    # plt.close()
    #
    # plt.imshow(p_spec_2d_cmb, norm=matplotlib.colors.LogNorm())
    # plt.title("2d truth power spec")
    # plt.colorbar()
    # plt.savefig(f"pspec_debug_plots/2d_pspec_cmb_{z}.png")
    # plt.close()

    # x = (p_spec_2d_pbox_full-p_spec_2d_cmb) / p_spec_2d_cmb
    # x = np.reshape(x, (side_length, side_length))
    # plt.imshow(x, norm=matplotlib.colors.SymLogNorm(linthresh=0.001))
    # plt.title("diff pspec")
    # plt.colorbar()
    # plt.savefig(f"pspec_debug_plots/diff_{z}.png")
    # plt.close()

    # plt.imshow(weights, norm=matplotlib.colors.SymLogNorm(linthresh=0.001))
    # plt.title("weights")
    # plt.colorbar()
    # plt.savefig(f"pspec_debug_plots/weights_{z}.png")
    # plt.close()