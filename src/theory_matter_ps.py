import jax.numpy as np
import camb
from camb import model
import matplotlib.pyplot as plt
import powerbox as pbox
import matplotlib
from astropy.cosmology import Cosmology

## defaults for paper plots
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams.update({'font.size': 12})

# Adélie's old parameters, unsure which CMB model they're from
# h = 0.6774000
# H0 = h * 100.
# Om_0 = 0.309
# Ol_0 = 0.691
# Ob_0 = 0.049
# obh2 = Ob_0 * h**2
# och2 = (Om_0 - Ob_0) * h**2
# print("physical density in baryons")
# print(obh2)
# print("physical density in CDM")
# print(och2)
# A_s = 2.139e-9
# n_s = 0.9677
# T_CMB = 2.7260  # K
# tau = 0.054 # thomson/reionization optical depth
# z_max = 15 # highest z value to calculate pspec at

def CROSS_circular_spec_normal(field_1, field_2, nbins, resolution, area):
    """
    2D field
    square before averaging (histogramming)
    RESOLUTION IS [MPC/PIXEL]
    """
    curr_side_length = np.shape(field_1)[0]
    fft_data_1 = np.fft.fftn(field_1)
    fft_data_2 = np.fft.fftn(field_2)

    fft_data_squared = np.abs(fft_data_1*fft_data_2)
    k_arr = np.fft.fftfreq(curr_side_length) * 2 * np.pi
    k_arr *= 1 / resolution # convert from pixel^-1 to Mpc^-1
    k1, k2 = np.meshgrid(k_arr, k_arr) # 3D!! meshgrid :)
    k_mag_full = np.sqrt(k1 ** 2 + k2 ** 2)

    counts, bin_edges = np.histogram(k_mag_full, nbins)
    binned_power, _ = np.histogram(k_mag_full, nbins, weights=fft_data_squared)

    bin_means = (np.histogram(k_mag_full, nbins, weights=k_mag_full)[0] /
                 np.histogram(k_mag_full, nbins)[0]) # mean k value in each bin
    pspec = binned_power / counts # average power in each bin
    # print(f"resolution, {resolution} mpc/pixel") # pixels/mpc
    pspec /= area # pixels^6 to pixels^3
    pspec *= resolution**2 # converting form pixels^3 to Mpc^3
    return counts, pspec, bin_means

def circular_spec_normal(field, nbins, resolution, area):
    """
    2D field, AREA SHOULD BE IN PIXELS^2
    square before averaging (histogramming)
    RESOLUTION IS [MPC/PIXEL]
    """
    curr_side_length = np.shape(field)[0]
    fft_data = np.fft.fftn(field)
    fft_data_squared = np.abs(fft_data**2)
    k_arr = np.fft.fftfreq(curr_side_length) * 2 * np.pi
    k_arr *= 1 / resolution # convert from pixel^-1 to Mpc^-1
    k1, k2 = np.meshgrid(k_arr, k_arr) # 3D!! meshgrid :)
    k_mag_full = np.sqrt(k1 ** 2 + k2 ** 2)
    counts, bin_edges = np.histogram(k_mag_full, nbins)
    binned_power, _ = np.histogram(k_mag_full, nbins, weights=fft_data_squared)

    bin_means = (np.histogram(k_mag_full, nbins, weights=k_mag_full)[0] /
                 np.histogram(k_mag_full, nbins)[0]) # mean k value in each bin
    pspec = binned_power / counts # average power in each bin
    # print(f"resolution, {resolution} mpc/pixel") # pixels/mpc
    pspec /= area # pixels^6 to pixels^3
    pspec *= resolution**2 # converting form pixels^3 to Mpc^3
    return counts, pspec, bin_means

def after_circular_spec_normal(field, nbins, resolution, area):
    """
    2D field
    square AFTER averaging (histogramming)
    RESOLUTION IS [MPC/PIXEL]
    """
    curr_side_length = np.shape(field)[0]
    fft_data = np.abs(np.fft.fftn(field))
    k_arr = np.fft.fftfreq(curr_side_length) * 2 * np.pi
    k_arr *= 1 / resolution # convert from pixel^-1 to Mpc^-1
    k1, k2 = np.meshgrid(k_arr, k_arr) # 3D!! meshgrid :)

    k_mag_full = np.sqrt(k1 ** 2 + k2 ** 2)

    counts, bin_edges = np.histogram(k_mag_full, nbins)
    binned_power, _ = np.histogram(k_mag_full, nbins, weights=fft_data)

    bin_means = (np.histogram(k_mag_full, nbins, weights=k_mag_full)[0] /
                 np.histogram(k_mag_full, nbins)[0]) # mean k value in each bin
    pspec = binned_power**2 / counts # average power in each bin
    # print(f"resolution, {resolution} mpc/pixel") # pixels/mpc
    pspec /= area # pixels^6 to pixels^3
    pspec *= resolution**2 # converting form pixels^3 to Mpc^3
    return counts, pspec, bin_means

def spherical_p_spec_normal(field, nbins, resolution, volume):
    """
    3D field
    square before averaging (histogramming)
    RESOLUTION IS [MPC/PIXEL]
    """
    curr_side_length = np.shape(field)[0]
    fft_data = np.fft.fftn(field)
    fft_data_squared = np.abs(fft_data**2)
    k_arr = np.fft.fftfreq(curr_side_length) * 2 * np.pi
    k_arr *= 1 / resolution
    k1, k2, k3 = np.meshgrid(k_arr, k_arr, k_arr) # 3D!! meshgrid :)
    k_mag_full = np.sqrt(k1 ** 2 + k2 ** 2 + k3 ** 2)

    counts, bin_edges = np.histogram(k_mag_full, nbins)
    binned_power, _ = np.histogram(k_mag_full, nbins, weights=fft_data_squared)

    bin_means = (np.histogram(k_mag_full, nbins, weights=k_mag_full)[0] /
                 np.histogram(k_mag_full, nbins)[0]) # mean k value in each bin
    pspec = binned_power / counts # average power in each bin
    # print(f"resolution, {resolution} mpc/pixel") # mpc/pixels
    pspec /= volume # pixels^6 to pixels^3
    pspec *= resolution**3 # converting form pixels^3 to Mpc^3
    return counts, pspec, bin_means

def after_spherical_p_spec_normal(field, nbins, resolution, volume):
    """
    3D field
    square after averaging (histogramming)
    RESOLUTION IS [MPC/PIXEL]
    """
    """
    3D field
    square before averaging (histogramming)
    RESOLUTION IS [MPC/PIXEL]
    """
    curr_side_length = np.shape(field)[0]
    fft_data = np.fft.fftn(field)
    fft_data_abs= np.abs(fft_data)
    k_arr = np.fft.fftfreq(curr_side_length) * 2 * np.pi
    k_arr *= 1 / resolution
    k1, k2, k3 = np.meshgrid(k_arr, k_arr, k_arr) # 3D!! meshgrid :)
    k_mag_full = np.sqrt(k1 ** 2 + k2 ** 2 + k3 ** 2)

    counts, bin_edges = np.histogram(k_mag_full, nbins)
    binned_power, _ = np.histogram(k_mag_full, nbins, weights=fft_data_abs)

    bin_means = (np.histogram(k_mag_full, nbins, weights=k_mag_full)[0] /
                 np.histogram(k_mag_full, nbins)[0])  # mean k value in each bin
    pspec = binned_power**2 / counts  # average power in each bin
    pspec /= volume # pixels^6 to pixels^3
    pspec *= resolution**3 # converting form pixels^3 to Mpc^3
    return counts, pspec, bin_means

def get_truth_matter_pspec(kmax, side_length, z, dim):
    # Initialise CAMB object
    print(f"redshift = {z}")
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=67.6, ombh2=0.022, omch2=0.119) # Planck 2018, Table 2 rightmost column (same as 21cmFAST)
    pars.InitPower.set_params(As=2.105e-9, ns=0.9665)
    pars.WantTransfer = True
    pars.set_dark_energy()
    # Compute everything you need for the power spectrum
    data = camb.get_background(pars)
    results = camb.get_results(pars)
    # compute ps and interpolate
    interp_l = camb.get_matter_power_interpolator(
        pars,  # cosmology
        nonlinear=True,  # linear or non-linear matter PS
        kmax=kmax,  # max k: computations much slower for larger values
        hubble_units=False,  # PS in Mpc or h-1 Mpc?
        k_hunit=False,  # k modes in Mpc-1 or hMpc-1?
        zmax=15,  # max z you're interpolating to
        var1=model.Transfer_nonu,  # which matter components to include. Here, everything but neutrinos
        var2=model.Transfer_nonu,
    )
    if dim == 2:
        pspec_k_func = lambda k: interp_l.P(z, k, grid=False) / side_length
    elif dim == 3:
        pspec_k_func = lambda k: interp_l.P(z, k, grid=False)
    elif dim != 3:
        print("# of dimensions not supported.")
        exit()
    return pspec_k_func

def convert_pspec_2_2D(power_spectrum, num_k_modes, resolution):
    p_spec_2D = np.empty((num_k_modes, num_k_modes))
    kfreq = np.fft.fftshift(np.fft.fftfreq(num_k_modes)) * (2 * np.pi)# 2D frequencies, not yet right scale
    k1, k2 = np.meshgrid(kfreq, kfreq)
    k_mag_full = np.sqrt(k1 ** 2 + k2 ** 2)

    # unbin stuff, I think for loop is fine, since you just do it once
    for i in range(num_k_modes):
        for j in range(num_k_modes):
            val = power_spectrum(k_mag_full[i, j])
            p_spec_2D[i, j] = val * resolution

    return k_mag_full, p_spec_2D

def calc_2D_pspec(field, area, resolution, pixel_side_length):
    p_spec_2d_pbox_full = np.fft.fftshift(np.fft.fftn((field)))
    p_spec_2d_pbox_full = np.abs(p_spec_2d_pbox_full**2)
    p_spec_2d_pbox_full *= area
    return p_spec_2d_pbox_full

def convert_pspec_1D_to_2D(power_spectrum_func, num_k_modes, resolution, dc_value=0.0):
    """
    Convert 1D power spectrum function P(k) to 2D grid with correct amplitudes

    Args:
        power_spectrum_func: Function that takes |k| and returns P(k)
        num_k_modes: Grid size (assumes square grid)
        box_size: Physical size of the simulation box
        dc_value: Value to set for DC component (k=0)

    Returns:
        k_mag_full: 2D array of |k| values at each grid point
        p_spec_2D: 2D power spectrum on grid with correct amplitudes
    """
    # Create frequency grid with proper scaling
    # The fundamental frequency is 2π/L where L is box_size
    k_arr = np.fft.fftshift(np.fft.fftfreq(num_k_modes) * 2 * np.pi)
    k_arr *= 1 / resolution
    k1, k2 = np.meshgrid(k_arr, k_arr)

    k_mag_full = np.sqrt(k1 ** 2 + k2 ** 2)

    # Vectorized evaluation of P(k)
    p_spec_2D = power_spectrum_func(k_mag_full)

    # The issue was here - you had "* resolution" which was wrong
    for i in range(num_k_modes):
        for j in range(num_k_modes):
            val = power_spectrum_func(k_mag_full[i, j])
            p_spec_2D[i, j] = val * num_k_modes** 2 / resolution**2  # NO extra scaling factor!
    # pspec /= volume # pixels^6 to pixels^3
    # pspec *= resolution**3 # converting form pixels^3 to Mpc^3
    return k_mag_full, p_spec_2D

#def convert_pspec_1D_to_2D(power_spectrum_func, num_k_modes, resolution, dc_value=0.0):
#     """
#     Convert 1D power spectrum function P(k) to 2D grid with correct amplitudes
#
#     Args:
#         power_spectrum_func: Function that takes |k| and returns P(k)
#         num_k_modes: Grid size (assumes square grid)
#         box_size: Physical size of the simulation box
#         dc_value: Value to set for DC component (k=0)
#
#     Returns:
#         k_mag_full: 2D array of |k| values at each grid point
#         p_spec_2D: 2D power spectrum on grid with correct amplitudes
#     """
#     # Create frequency grid with proper scaling
#     # The fundamental frequency is 2π/L where L is box_size
#     k_arr = np.fft.fftshift(np.fft.fftfreq(num_k_modes) * 2 * np.pi)
#     k_arr *= 1 / resolution
#     k1, k2 = np.meshgrid(k_arr, k_arr)
#
#     k_mag_full = np.sqrt(k1 ** 2 + k2 ** 2)
#
#     # Vectorized evaluation of P(k)
#     p_spec_2D = power_spectrum_func(k_mag_full)
#
#     # The issue was here - you had "* resolution" which was wrong
#     for i in range(num_k_modes):
#         for j in range(num_k_modes):
#             val = power_spectrum_func(k_mag_full[i, j])
#             p_spec_2D[i, j] = val * pixel_side_length** 2 / resolution**2  # NO extra scaling factor!
#     # pspec /= volume # pixels^6 to pixels^3
#     # pspec *= resolution**3 # converting form pixels^3 to Mpc^3
#     return k_mag_full, p_spec_2D

#
# def circular_spec_normal(field, nbins, resolution, area):
#     """
#     2D field, AREA SHOULD BE IN PIXELS^2
#     square before averaging (histogramming)
#     RESOLUTION IS [MPC/PIXEL]
#     """
#     curr_side_length = np.shape(field)[0]
#     fft_data = np.fft.fftn(field)
#     fft_data_squared = np.abs(fft_data**2)
#     k_arr = np.fft.fftfreq(curr_side_length) * 2 * np.pi
#     k_arr *= 1 / resolution # convert from pixel^-1 to Mpc^-1
#     k1, k2 = np.meshgrid(k_arr, k_arr) # 3D!! meshgrid :)
#     k_mag_full = np.sqrt(k1 ** 2 + k2 ** 2)
#     counts, bin_edges = np.histogram(k_mag_full, nbins)
#     binned_power, _ = np.histogram(k_mag_full, nbins, weights=fft_data_squared)
#
#     bin_means = (np.histogram(k_mag_full, nbins, weights=k_mag_full)[0] /
#                  np.histogram(k_mag_full, nbins)[0]) # mean k value in each bin
#     pspec = binned_power / counts # average power in each bin
#     # print(f"resolution, {resolution} mpc/pixel") # pixels/mpc
#     pspec /= area # pixels^6 to pixels^3
#     pspec *= resolution**2 # converting form pixels^3 to Mpc^3
#     return counts, pspec, bin_means

# Example usage and validation
def test_conversion(p_spec_k_func, pixel_side_length, physical_side_length):
    """Test the conversion with a simple power law"""

    k_mag, p_spec_2D = convert_pspec_1D_to_2D(
        p_spec_k_func, pixel_side_length, physical_side_length
    )

    print(f"Grid shape: {p_spec_2D.shape}")
    print(f"k range: [{k_mag.min():.3f}, {k_mag.max():.3f}]")
    print(f"Power spectrum range: [{p_spec_2D[p_spec_2D > 0].min():.3e}, {p_spec_2D.max():.3e}]")

    return k_mag, p_spec_2D


# Uncomment to test:


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib import cm
    import numpy as np
    pixel_side_length = 32
    physical_side_length = 64
    resolution = physical_side_length / pixel_side_length
    area = pixel_side_length**2
    dim = 2
    z = 7

    kmax = 50

    pspec_box = np.load("pspec_box.npy")
    truth_field = np.load("truth_field.npy")
    pbox_2 = np.load("pbox_2.npy")
    pbox_3 = np.load("pbox_3.npy")
    pbox_4 = np.load("pbox_4.npy")
    pbox_best = np.load("pbox_best.npy")

    p_spec_k_func = get_truth_matter_pspec(kmax, physical_side_length, z, dim)
    k_mag_full, p_spec_2D = convert_pspec_1D_to_2D(p_spec_k_func, pixel_side_length, resolution) #, dc_value=0.0)
    plt.imshow(p_spec_2D)
    plt.colorbar()
    # k_mag, p_spec_2D = test_conversion(p_spec_k_func, pixel_side_length, physical_side_length)

    plt.show()

    plt.imshow(pspec_box)
    plt.colorbar()
    plt.show()
    count = np.count_nonzero(pspec_box > 4)  # strict >
    print(count)
    counts, pspec_inferred, bin_means = circular_spec_normal(truth_field, 32,resolution, area)
    plt.semilogy(bin_means, pspec_inferred)
    plt.semilogy(bin_means, p_spec_k_func(bin_means))
    plt.show()

    print(np.max(p_spec_2D)/np.max(pbox_4))

    plt.imshow(pspec_box-p_spec_2D)
    plt.colorbar()
    plt.show()

    plt.imshow(pbox_best-pspec_box)
    plt.colorbar()
    plt.show()
    # mask = k_mag_full != 0
    # k_mag_full = k_mag_full[mask] # masking 0
    # print("k_mag_full min")
    # print(np.min(k_mag_full))
    #
    # print("k_mag_full max")
    # print(np.max(k_mag_full))
    # gpu = GPUtil.getGPUs()[0]
    # print(gpu)
    # redshifts = [0, 6, 8, 10]
    # colors = cm.viridis(np.linspace(0, 1, len(redshifts)))
    # for i, z in enumerate(redshifts): # making Figure CAMB_3dpspec
    #     if dim == 3:
    #         p_spec_k_func = get_truth_matter_pspec(kmax, physical_side_length, z, dim)
    #         k_mag_full = k_mag_full.flatten()
    #         plt.semilogy(k_mag_full, p_spec_k_func(k_mag_full), label=f'z = {z}', color=colors[i])
    #         gpu = GPUtil.getGPUs()[0]
    #         print(
    #             f"GPU ID: {gpu.id}, Memory Free: {gpu.memoryFree}MB, Memory Used: {gpu.memoryUsed}MB, Memory Total: {gpu.memoryTotal}MB")
    #
    # plt.xlabel(r"$\mathbf{k}~[\rm Mpc^{-1}]$")
    # plt.ylabel(r"$\rm P_{mm, truth}~[Mpc^3]$")
    # plt.legend()
    # plt.savefig("../paper_plots/CAMB_3dpspec.png", dpi=300)

