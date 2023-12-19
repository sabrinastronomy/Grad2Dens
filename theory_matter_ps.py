import jax.numpy as np
import camb
from camb import model
import matplotlib.pyplot as plt
import powerbox as pbox
import matplotlib
from jax_battaglia_full import Dens2bBatt

# SETTINGS (see below for usage)
nonlinear = False
little_h = False

# COSMOLOGY
H0 = 67.4
h = H0 / 100
Om_0 = 0.315 # total matter
Ol_0 = 0.6847 # ?
obh2 = 0.02237
Ob_0 = obh2 / h**2
och2 = (Om_0 - Ob_0) * h**2 # cold dark matter
A_s = 2.100e-9
n_s = 0.965
T_CMB = 2.7255  # K
tau = 0.054 # thomson/reionization optical depth
z_max = 15 # highest z value to calculate pspec at

# Adélie's old parameters
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

def circular_spec_normal(field, nbins, resolution, area, verbose=False):
    """
    square before averaging (histogramming)
    """
    curr_side_length = np.shape(field)[0]
    fft_data = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(field)))
    # fft_data_squared = np.real(fft_data * np.conj(fft_data)) # units pixels^4
    fft_data_squared = np.abs(fft_data**2)
    k_arr = np.fft.fftshift(np.fft.fftfreq(curr_side_length)) * 2 * np.pi
    k_arr *= resolution # pixels/side length, changing to Mpc^-1
    k1, k2 = np.meshgrid(k_arr, k_arr)
    k_mag_full = np.sqrt(k1 ** 2 + k2 ** 2)

    counts, bin_edges = np.histogram(k_mag_full, nbins)

    binned_power, _ = np.histogram(k_mag_full, nbins, weights=fft_data_squared)

    bin_means = (np.histogram(k_mag_full, nbins, weights=k_mag_full)[0] /
                 np.histogram(k_mag_full, nbins)[0]) # mean k value in each bin

    # kvals = 0.5 * (bin_edges[:-1] + bin_edges[1:]) # center of bins
    pspec = binned_power / counts # average power in each bin
    if verbose:
        print(f"resolution, {1/resolution} mpc/pixel") # mpc/pixel
    pspec /= area # pixels^4 --> pixels^2
    pspec *= (1/resolution)**2 # converting form pixels^2 to Mpc^2
    return counts, pspec, bin_means

def spherical_p_spec_normal(field, nbins, resolution, volume):
    """
    square before averaging (histogramming)
    """
    curr_side_length = np.shape(field)[0]
    fft_data = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(field)))
    fft_data_squared = fft_data * np.conj(fft_data)
    k_arr = np.fft.fftshift(np.fft.fftfreq(curr_side_length)) * 2 * np.pi
    k_arr *= resolution
    k1, k2, k3 = np.meshgrid(k_arr, k_arr, k_arr) # 3D!! meshgrid :)
    k_mag_full = np.sqrt(k1 ** 2 + k2 ** 2 + k3 ** 2)

    counts, bin_edges = np.histogram(k_mag_full, nbins)
    binned_power, _ = np.histogram(k_mag_full, nbins, weights=fft_data_squared)

    kvals = 0.5 * (bin_edges[:-1] + bin_edges[1:]) # center of bins
    pspec = binned_power / counts # average power in each bin
    print(f"resolution, {resolution} pixels/mpc") # pixels/mpc
    pspec /= volume
    pspec *= (1/resolution)**3 # converting form pixels^3 to Mpc^3
    print("PLEASE CHECK THIS")
    return counts, pspec, kvals

def get_truth_matter_pspec(kmax, side_length, z, dim):
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
    print(f"# of dimensions {dim}")
    if dim == 2:
        pspec_k_func = lambda k: interp_l.P(z, k) / side_length
    elif dim == 3:
        pspec_k_func = lambda k: interp_l.P(z, k)
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
    p_spec_2d_pbox_full = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(field)))
    p_spec_2d_pbox_full = np.abs(p_spec_2d_pbox_full**2)
    p_spec_2d_pbox_full *= area
    return p_spec_2d_pbox_full


if __name__ == "__main__":



    pixel_side_length = 256
    physical_side_length = 128

    dim = 2
    z = 7
    if dim == 3:
        no_shift_k = np.fft.fftfreq(pixel_side_length)
        k_arr = np.fft.fftshift(np.fft.fftfreq(pixel_side_length)) * 2 * np.pi
        k1, k2, k3 = np.meshgrid(k_arr, k_arr, k_arr)
        k_mag_full = np.sqrt(k1 ** 2 + k2 ** 2 + k3**2)

        kmax = 2 * np.pi / physical_side_length * (pixel_side_length / 2)
        print("kmax for theory")
        print(kmax)
        pspec_k_func = get_truth_matter_pspec(kmax, physical_side_length, z, dim)



        pb = pbox.PowerBox(
            N=pixel_side_length,
            dim=dim,  # dimension of box
            pk=pspec_k_func,  # The power-spectrum
            boxlength=physical_side_length,  # Size of the box (sets the units of k in pk)
            seed=1010,  # Use the same seed as our powerbox
        )

        resolution = pixel_side_length / physical_side_length
        volume_pixels = pixel_side_length ** 3
        p_k_field, bins_field = pbox.get_power(pb.delta_x(), boxlength=pb.boxlength)

        print("bins field")
        print(np.max(bins_field))

        p_spec_cmb = pspec_k_func(bins_field)
        _, pspec_own, kvals_own = spherical_p_spec_normal(pb.delta_x(), 64, resolution, volume_pixels)

        plt.title(f"physics side length = {physical_side_length}, number of pixels = {pixel_side_length}")
        plt.loglog(bins_field, p_k_field, label=rf'powerbox (large box)', c="m")
        plt.loglog(bins_field, p_spec_cmb.squeeze(), label=rf'truth (large box)', ls="--", c="m")
        plt.loglog(kvals_own, pspec_own, label=rf'my pspec (large box)')
        plt.legend()
        plt.savefig("pspec_3D_comp.png")
        plt.close()

        plt.close()
        plt.imshow(pb.delta_x()[0, :, :])
        plt.title("truth field")
        plt.colorbar()
        plt.show()


    if dim == 2:
        # no_shift_k = np.fft.fftfreq(pixel_side_length)
        # k_arr = np.fft.fftshift(np.fft.fftfreq(pixel_side_length)) * 2 * np.pi
        # k1, k2 = np.meshgrid(k_arr, k_arr)
        # k_mag_full = np.sqrt(k1 ** 2 + k2 ** 2)

        kmax = 30
        pspec_k_func = get_truth_matter_pspec(kmax, physical_side_length, z, dim)

        ## benji
        pb = pbox.PowerBox(
            N=512,
            dim=dim,  # dimension of box
            pk=pspec_k_func,  # The power-spectrum
            boxlength=1024,  # Size of the box (sets the units of k in pk)
            seed=1010,  # Use the same seed as our powerbox
        )
        p_k_field, bins_field = pbox.get_power(pb.delta_x(), boxlength=pb.boxlength)
        plt.loglog(bins_field, p_k_field, label="side length = 1024 Mpc")
        pb_2 = pbox.PowerBox(
            N=512,
            dim=dim,  # dimension of box
            pk=pspec_k_func,  # The power-spectrum
            boxlength=256,  # Size of the box (sets the units of k in pk)
            seed=1010,  # Use the same seed as our powerbox
        )

        p_k_field, bins_field = pbox.get_power(pb_2.delta_x(), boxlength=pb_2.boxlength)
        plt.loglog(bins_field, p_k_field, label="side length = 256 Mpc")
        plt.legend()
        plt.show()
        plt.close()
        plt.imshow(pb.delta_x())
        plt.title("1024 Mpc")
        plt.show()
        plt.close()
        plt.imshow(pb_2.delta_x())
        plt.title("256 Mpc")
        plt.show()
        plt.close()


        pb = pbox.PowerBox(
            N=pixel_side_length,
            dim=dim,  # dimension of box
            pk=pspec_k_func,  # The power-spectrum
            boxlength=physical_side_length,  # Size of the box (sets the units of k in pk)
            seed=1010,  # Use the same seed as our powerbox
        )

        pb_2 = pbox.PowerBox(
            N=pixel_side_length,
            dim=dim,  # dimension of box
            pk=pspec_k_func,  # The power-spectrum
            boxlength=3432,  # Size of the box (sets the units of k in pk)
            seed=1010,  # Use the same seed as our powerbox
        )

        resolution = pixel_side_length / physical_side_length
        area = pixel_side_length**2
        p_k_field, bins_field = pbox.get_power(pb.delta_x(), boxlength=pb.boxlength)


        p_spec_cmb = pspec_k_func(bins_field)
        num_bins = len(bins_field)

        _, pspec_own, kvals_own = circular_spec_normal(pb.delta_x(), num_bins, resolution, area)
        batt_model_instance = Dens2bBatt(pb.delta_x(), delta_pos=1, set_z=7, flow=True, resolution=resolution)

        plt.title(f"physical side length = {physical_side_length}, number of pixels = {pixel_side_length}")
        plt.loglog(bins_field, p_k_field, label=rf'powerbox', c="m")
        plt.loglog(bins_field, p_spec_cmb.squeeze(), label=rf'truth', ls="--", c="m")
        plt.loglog(kvals_own, pspec_own, label=rf'my pspec')
        plt.legend()
        plt.savefig("pspec_debug_plots/pspec_2D_comp.png")
        plt.close()

        plt.loglog(bins_field, np.abs(pspec_own - p_k_field), label=rf'my pspec - pbox pspec', ls="--", c="m")
        plt.legend()
        plt.savefig("pspec_debug_plots/pspec_1D_diff.png")
        plt.close()

        plt.imshow(pb.delta_x())
        plt.title("truth field")
        plt.colorbar()
        plt.savefig("pspec_debug_plots/truth_field.png")
        plt.close()

        plt.imshow(batt_model_instance.temp_brightness)
        plt.title("temp brightness")
        plt.colorbar()
        plt.savefig("pspec_debug_plots/data.png")
        plt.close()

        k_mag_full, p_spec_2d_cmb = convert_pspec_2_2D(pspec_k_func, pixel_side_length, resolution)

        # p_spec_2d_pbox_full = calc_2D_pspec(pb.delta_x(), area, resolution, pixel_side_length)
        p_spec_2d_pbox_full = pb.power_array() * area
        x = p_spec_2d_pbox_full-p_spec_2d_cmb
        print(np.sum(x))
        print(np.sum((pb_2.power_array() * area) -p_spec_2d_cmb))

        plt.imshow(x, norm=matplotlib.colors.SymLogNorm(linthresh=0.001))
        plt.title("diff pspec")
        plt.colorbar()
        plt.savefig(f"pspec_debug_plots/diff_{z}.png")
        plt.close()

        plt.imshow(p_spec_2d_pbox_full, norm=matplotlib.colors.LogNorm())
        plt.title("2d pbox power spec")
        plt.colorbar()
        plt.savefig(f"pspec_debug_plots/2d_pspec_pbox_{z}.png")
        plt.close()

        plt.imshow(p_spec_2d_cmb, norm=matplotlib.colors.LogNorm())
        plt.title("2d truth power spec")
        plt.colorbar()
        plt.savefig(f"pspec_debug_plots/2d_pspec_cmb_{z}.png")
        plt.close()


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