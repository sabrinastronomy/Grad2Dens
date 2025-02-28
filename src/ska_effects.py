import matplotlib
import matplotlib.pyplot as plt
import jax
import jax.numpy as np  # use jnp for jax numpy, note that not all functionality/syntax is equivalent to normal numpy
import os
import astropy
from astropy.cosmology import Planck18 as cosmo
from astropy import units as u
from jax.scipy.signal import convolve
# import scipy.stats as stats
from matplotlib.colors import LogNorm
import gc

### defaults for paper plots
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams.update({'font.size': 12})


# beam effects (convolving with (12) from Gorce+2021),
# thermal noise (equation (13)
# the wedge (converting to kperp/kparallel pspec, cutting frequencies, and then going back to real space)

class SKAEffects:
    """"
    A class that takes in a brightness temperature field and applies instrumental effects
    # 1) beam effects (convolving with (12) from Gorce+2021),
    # 2) thermal noise (equation (13)
    # 3) the wedge (converting to kperp/kparallel pspec, cutting frequencies, and then going back to real space)
    """

    def __init__(self, brightness_temp, add_all, redshift, physical_side_length=None):
        self.brightness_temp_original = np.copy(brightness_temp)
        self.brightness_temp = brightness_temp
        self.redshift = redshift
        self.resolution = np.shape(self.brightness_temp)[0] / physical_side_length  # pixels/mpc
        self.physical_side_length = physical_side_length
        if add_all:
            self.add_all_effects()

    def add_all_effects(self):
        # jax.debug.print("pre wedge brightness temp: {}", np.max(self.brightness_temp))
        # self.wedge()
        # jax.debug.print("pre convolution brightness temp: {}", np.max(self.brightness_temp))
        self.thermal_noise()

        self.beam_convolution()
        # jax.debug.print("pre thermal noise brightness temp: {}", np.max(self.brightness_temp))
        # jax.debug.print("post thermal noise brightness temp: {}", np.max(self.brightness_temp))

    def convolve_own(self, field_1, kernel):
        # shifting
        # field_1 = np.fft.fftshift(field_1)
        # kernel = np.fft.fftshift(kernel)
        # Compute FFT of both the input and kernel
        input_fft = np.fft.fftn(field_1)
        kernel_fft = np.fft.fftn(kernel)

        # Perform element-wise multiplication in the frequency domain
        result_fft = input_fft * kernel_fft

        # result_fft = np.fft.ifftshift(result_fft)
        # Compute the inverse FFT to obtain the convolved result
        result = np.fft.ifftn(result_fft)
        result = np.fft.ifftshift(result)
        # Take the real part of the result (in case of small imaginary parts due to numerical errors)
        return np.real(result)

    def beam_convolution(self, wavelength=21e-2, b_max=65e3, plot=False):
        """
        Beam effects (convolving with (12) from Gorce+2021) applied to field
        Default parameters are for SKA-low
        baseline = 65km
        wavelength = 21cm
        Input all units in meters # TODO: change size of beam to be just core!!!!!! 2km
        """
        redshifted_wavelength = wavelength * (1 + self.redshift)
        # redshifted_wavelength /= 3.086e+22 # m --> Mpc
        # b_max /= 3.086e+22 # m --> Mpc

        comoving_distance = cosmo.comoving_distance(self.redshift).to(
            u.parsec).value / 1e6  # comoving distance to redshift

        theta_z = 1.22 * redshifted_wavelength / b_max
        # print("theta_z")
        # print(theta_z *180/np.pi)
        fwhm = theta_z * comoving_distance
        self.sigma_gauss = 2.35482004503 * fwhm  # conversion factor from https://brainder.org/2011/08/20/gaussian-kernels-convert-fwhm-to-sigma/
        # jax.debug.print("sigma_gauss: {}", self.sigma_gauss)
        key = jax.random.PRNGKey(0)  # Create a random key
        # gaussian_kernel = sigma_gauss * jax.random.normal(key, np.shape(self.brightness_temp))  # Generate a 3x3 array of random normal numbers
        # gaussian_kernel should be the FUNCTIONAL FORM, NORMALIZED
        # integral of smoothed field to be same of integral of field before smoothing!!

        # Parameters for the Gaussian
        mean = np.array([0.0, 0.0, 0.0])  # Center of the Gaussian
        std_dev = self.sigma_gauss  # Standard deviation of the Gaussian
        pixel_length = np.shape(self.brightness_temp)[0]

        half_pixel_length = self.physical_side_length / 2
        x = np.linspace(-half_pixel_length, half_pixel_length, pixel_length)
        y = np.linspace(-half_pixel_length, half_pixel_length, pixel_length)
        z = np.linspace(-half_pixel_length, half_pixel_length, pixel_length)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

        # Compute the Gaussian field
        gaussian_kernel = np.exp((-1/(2*std_dev**2)) * ((X - mean[0]) ** 2 + (Y - mean[1]) ** 2 + (Z - mean[2]) ** 2)) / (np.sqrt(2 * np.pi * std_dev ** 2))**3
        # gaussian_kernel = np.zeros_like(self.brightness_temp)
        # gaussian_kernel = gaussian_kernel.at[pixel_length // 2, pixel_length // 2, pixel_length // 2].set(1)
        # gaussian_kernel /= np.sum(gaussian_kernel) # normalize



        self.brightness_temp = self.convolve_own(self.brightness_temp, gaussian_kernel)
        if plot:
            cmap = plt.cm.viridis  # Choose a colormap
            slice = gaussian_kernel[:, 64, :]
            # print("min and max")
            # print(np.sum(gaussian_kernel))
            # print(np.min(slice))
            # print(np.max(slice))
            # exit()
            plt.imshow(slice, cmap=cmap)
            plt.colorbar()
            plt.savefig("debug_ska/gauss_kernel_new.png")
            plt.close()

            fig, axs = plt.subplots(1, 2, figsize=(15, 5), constrained_layout=True)
            min_bright_temp = np.min(self.brightness_temp)
            max_bright_temp = np.max(self.brightness_temp)

            im0 = axs[0].imshow(self.brightness_temp_original[0, :, :], origin='lower', cmap='viridis', #vmin=min_bright_temp, vmax=max_bright_temp)
                                vmin=np.min(self.brightness_temp_original[0, :, :]), vmax=np.max(self.brightness_temp_original[0, :, :]))
            axs[0].set_xlabel('Pre-Beam Convolution')

            im1 = axs[1].imshow(self.brightness_temp[0, :, :], origin='lower', cmap='viridis', vmin=min_bright_temp, vmax=max_bright_temp)
            axs[1].set_xlabel('Post SKA1-Low Beam Convolution')

            #Add colorbars to each plot
            fig.colorbar(im0, ax=axs[0], orientation='vertical', label=r'$\rm T_b~[mK]$')
            fig.colorbar(im1, ax=axs[1], orientation='vertical', label=r'$\rm T_b~[mK]$')

            for ax in axs:
                ax.set_xticks([])
                ax.set_yticks([])

            fig.savefig(f"debug_ska/beam_convolution.png", dpi=300)
            plt.close(fig)

    def thermal_noise(self, t_int=1000, area_tot=1e5, delta_theta=10):
        """
        Add thermal noise as in Equation (13) in Gorce+2021
        Default parameters are for SKA-low
        area_tot = 1e5 m^2
        Delta_theta = 10 arcmin
        Delta_nu = simulation frequency resolution (TODO)
        t_int = 1000 hours (TODO)
        """
        print("Adding thermal noise")
        # print("resolution", self.resolution)
        # Parameters
        mpc = 3.086e+22  # m
        rest_freq = 3e8 / 21.0e-2  # Rest wavelength of 21 cm line in Hz

        H_z = cosmo.H(0).value  # Hubble parameter at redshift = 0,  km/s/Mpc
        c = 3e5  # in km
        # Compute observed frequency width corresponding to physical length
        obs_freq_width = H_z * rest_freq * np.sqrt(cosmo.Om0) * 1 / self.resolution
        obs_freq_width /= (c * np.sqrt(1 + self.redshift))  # this is top of page 8 in Gorce+2021
        # obs_freq_width /= 1e6 # in MHz
        # jax.debug.print("obs_freq_width {}", obs_freq_width)
        self.sigma_th = (2.9 * ((1 + self.redshift) / 10) ** 4.6 * np.sqrt(
            (1e6 / obs_freq_width) * (100 / t_int)))  # Equation (13) in Gorce+2021
        # TODO add delta theta: convert to arcmin from rad
        jax.debug.print("var_th : {}", np.max(self.sigma_th))

        key = jax.random.PRNGKey(0)  # Create a random key
        thermal_noise = self.sigma_th * jax.random.normal(key, np.shape(
            self.brightness_temp))  # Generate a 3x3 array of random normal numbers
        jax.debug.print("thermal_noise : {}", np.max(thermal_noise))

        self.brightness_temp += thermal_noise
        # fig, axs = plt.subplots(1, 2, figsize=(15, 5), constrained_layout=True)
        # min_bright_temp = np.min(self.brightness_temp)
        # max_bright_temp = np.max(self.brightness_temp)
        #
        # im0 = axs[0].imshow(self.brightness_temp_original[0, :, :], origin='lower', cmap='viridis', #vmin=min_bright_temp, vmax=max_bright_temp)
        #                     vmin=np.min(self.brightness_temp_original[0, :, :]), vmax=np.max(self.brightness_temp_original[0, :, :]))
        # axs[0].set_xlabel('Pre-Thermal Noise')
        #
        # im1 = axs[1].imshow(self.brightness_temp[0, :, :], origin='lower', cmap='viridis', vmin=min_bright_temp, vmax=max_bright_temp)
        # axs[1].set_xlabel('Post-Thermal Noise')

        # Add colorbars to each plot
        # fig.colorbar(im0, ax=axs[0], orientation='vertical', label=r'$\rm T_b~[mK]$')
        # fig.colorbar(im1, ax=axs[1], orientation='vertical', label=r'$\rm T_b~[mK]$')

        # for ax in axs:
        #     ax.set_xticks([])
        #     ax.set_yticks([])
        #
        # fig.savefig(f"debug_ska/thermal_noise.png", dpi=300)
        # plt.close(fig)

    def cylindrical_spec_normal(self):
        """
        just doing a 2D pspec
        """
        curr_side_length = np.shape(self.brightness_temp)[0]
        # volume = curr_side_length**3

        # take fft of 3D cube
        fft_data = np.fft.fftn(self.brightness_temp)  # 3D
        fft_data_squared = np.abs(fft_data ** 2)  # 3D units pixels^6
        # jax.debug.print("np.max(fft_data_squared): {}", np.max(fft_data_squared))

        # fft_data_squared_collapsed = np.average(fft_data_squared, axis=0) # collapsing to 2D field, but trying to preserve spatial scale by averaging? 0 is depth
        # getting k values
        k_arr = np.fft.fftfreq(
            curr_side_length) * 2 * np.pi * 1 / self.resolution  # follow Fourier conventions and convert from pixel^-1 to Mpc^-1
        # k_arr *= self.resolution  # pixels/side length, changing to Mpc^-1

        angles = np.angle(fft_data)

        # jax.debug.print("Current side length: {}", curr_side_length)
        k_0, k_1, k_2 = np.meshgrid(k_arr, k_arr, k_arr)  # 3D
        k_parallel = np.sqrt(k_0 ** 2)  # depth
        k_perp = np.sqrt(k_1 ** 2 + k_2 ** 2)  # sky plane norm

        # cyl_pspec, kperp_binned, k_parallel_binned = histogram2d(k_perp, k_parallel, weights=fft_data_squared_collapsed, bins=bins) # summing cylinders to get the 2D pspec
        # print(f"resolution, {1 / resolution} Mpc/pixel")  # Mpc/pixel

        cyl_pspec = fft_data_squared / volume  # pixels^6 --> pixels^3
        cyl_pspec *= (self.resolution) ** 3  # converting form pixels^3 to Mpc^3
        # jax.debug.print("np.max(cyl_pspec): {}", np.max(cyl_pspec))
        # cyl_pspec = fft_data
        return fft_data_squared, k_perp, k_parallel, angles

    def wedge(self, angle=75):  # 75 deg in nominal value from Gagnon-Hartman+2021
        cyl_pspec, k_perp, k_parallel, angles = self.cylindrical_spec_normal()
        psi = angle * np.pi / 180  # using 75 degrees as nominal value
        k_parallel_wedge = k_perp * np.tan(psi)  # this is (1) from Gagnon-Hartman+2021

        wedge_mask = k_parallel < k_parallel_wedge  # below this line, we have synchrotron emission, etc. from wedge
        true_count = np.sum(wedge_mask)
        jax.debug.print("Number of True values in wedge_mask: {}", true_count)
        indices = np.where(wedge_mask, size=2057748)  # TODO try flattening
        cyl_pspec = cyl_pspec.at[indices].set(0)
        ### ADD PHASES HERE NP.ANGLE E^(IPHI), TEST WITH A SUPER TINY K WEDGE,
        ### CONSISTENCY CHECK THAT IF
        ### YOU DON'T HAVE ANY CUT NOTHING HAPPENS, CHECK THAT YOU DON'T HAVE TO INVERSE THE NORMALIZATION????
        ### DO EVERYTHING ELSE IN PIXEL SPACE OR LEAVE PSPEC IN PIXEL SPACE

        cyl_pspec = np.sqrt(cyl_pspec)
        cyl_pspec *= np.exp(1j * angles)

        self.brightness_temp = np.real(np.fft.ifftn(cyl_pspec))

        # Create a horizontal 3-panel plot
        # fig, axs = plt.subplots(1, 4, figsize=(15, 5), constrained_layout=True)
        # min_bright_temp = np.min(self.brightness_temp)
        # max_bright_temp = np.max(self.brightness_temp)

        # im0 = axs[0].imshow(self.brightness_temp_original[:, 0, :], origin='lower', cmap='viridis', #vmin=min_bright_temp, vmax=max_bright_temp)
        #                     vmin=np.min(self.brightness_temp_original[0, :, :]), vmax=np.max(self.brightness_temp_original[0, :, :]))
        # axs[0].set_xlabel('Pre-Wedge')
        #
        # im1 = axs[1].imshow(self.brightness_temp[0, :, :], origin='lower', cmap='viridis', vmin=np.min(self.brightness_temp_original[0, :, :]), vmax=np.max(self.brightness_temp_original[0, :, :]))
        # axs[1].set_xlabel('Vertical LOS')
        # # axs[0].set_xlabel(r'$k_y$')
        # # axs[0].set_ylabel(r'$k_z$')
        #
        # im2 = axs[2].imshow(self.brightness_temp[:, :, 0], origin='lower', cmap='viridis', vmin=min_bright_temp, vmax=max_bright_temp)
        # axs[2].set_xlabel('Horizontal LOS')
        #
        # im3 = axs[3].imshow(self.brightness_temp[:, 0, :], origin='lower', cmap='viridis', vmin=min_bright_temp, vmax=max_bright_temp)
        # axs[3].set_xlabel('Transverse')
        #

        # Add colorbars to each plot
        # fig.colorbar(im0, ax=axs[0], orientation='vertical', shrink=0.6, label=r'$\rm T_b~[mK]$')
        # fig.colorbar(im1, ax=axs[1], orientation='vertical', shrink=0.6, label=r'$\rm T_b~[mK]$')
        # fig.colorbar(im2, ax=axs[2], orientation='vertical', shrink=0.6, label=r'$\rm T_b~[mK]$')
        # fig.colorbar(im3, ax=axs[3], orientation='vertical', shrink=0.6, label=r'$\rm T_b~[mK]$')
        # for ax in axs:
        #     ax.set_xticks([])
        #     ax.set_yticks([])
        # # fig.tight_layout()
        # fig.savefig(f"debug_ska/wedge_{angle}.png", dpi=300)
        # plt.close(fig)


if __name__ == "__main__":
    import powerbox as pbox
    from jax_battaglia_full import Dens2bBatt

    k_0_fiducial = 0.185
    alpha_fiducial = 0.564
    b_0_fiducial = 0.593
    midpoint_z_fiducial = 7
    tanh_fiducial = 1

    physical_side_length = 256
    pixel_side_length = 128
    z = 6.5
    pb = pbox.PowerBox(
        N=pixel_side_length,  # number of wavenumbers
        dim=3,  # dimension of box
        pk=lambda k: 0.1 * k ** -2,  # The power-spectrum
        boxlength=physical_side_length,  # Size of the box (sets the units of k in pk)
        seed=1010,  # Use the same seed as our powerbox
        # ensure_physical=True
    )
    fiducial_params = {"b_0": b_0_fiducial, "alpha": alpha_fiducial, "k_0": k_0_fiducial, "tanh_slope": tanh_fiducial,
                       "avg_z": midpoint_z_fiducial, "redshift_run": 6.5}  # b_0=0.5, alpha=0.2, k_0=0.1)

    plt.imshow(pb.delta_x()[0, :, :])
    plt.colorbar()
    plt.title("density")
    plt.savefig("debug_ska/density.png")
    plt.close()

    batt_model_instance = Dens2bBatt(pb.delta_x(), z, physical_side_length, physical_side_length/pixel_side_length, flow=True,
                                     free_params=fiducial_params,
                                     apply_ska=False, debug=True)

    plt.imshow(batt_model_instance.temp_brightness[:, :, 0])
    plt.colorbar()
    plt.title("brightness temp")
    plt.savefig("debug_ska/default.png")
    plt.close()

    # debug = SKAEffects(batt_model_instance.temp_brightness,False, z, physical_side_length=physical_side_length)
    # debug.wedge()
    # debug = SKAEffects(batt_model_instance.temp_brightness,False, z,physical_side_length=physical_side_length)
    # debug.beam_convolution()
    debug = SKAEffects(batt_model_instance.temp_brightness, False, z, physical_side_length=128)
    debug.beam_convolution(plot=True)


