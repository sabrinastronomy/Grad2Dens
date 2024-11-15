import matplotlib
import matplotlib.pyplot as plt
import jax
import jax.numpy as np  # use jnp for jax numpy, note that not all functionality/syntax is equivalent to normal numpy
import os
import astropy
from astropy.cosmology import Planck18 as cosmo
from astropy import units as u
from jax.scipy.signal import convolve
import scipy.stats as stats
import gc

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
    def __init__(self, brightness_temp, add_all, redshift, resolution):
        self.brightness_temp_original = brightness_temp
        self.brightness_temp = brightness_temp
        self.redshift = redshift
        self.resolution = resolution
        if add_all:
            self.add_all_effects()

    def add_all_effects(self):
        jax.debug.print("pre wedge brightness temp: {}", np.max(self.brightness_temp))
        self.wedge()
        jax.debug.print("pre convolution brightness temp: {}", np.max(self.brightness_temp))
        self.beam_convolution()
        jax.debug.print("pre thermal noise brightness temp: {}", np.max(self.brightness_temp))
        self.thermal_noise()
        jax.debug.print("post thermal noise brightness temp: {}", np.max(self.brightness_temp))

    def beam_convolution(self, wavelength=21e-2, b_max=65e3):
        """
        Beam effects (convolving with (12) from Gorce+2021) applied to field
        Default parameters are for SKA-low
        baseline = 65km
        wavelength = 21cm
        Input all units in meters
        """
        redshifted_wavelength = wavelength * (1 + self.redshift)
        redshifted_wavelength /= 3.086e+22 # m --> Mpc
        b_max /= 3.086e+22 # m --> Mpc

        comoving_distance = cosmo.comoving_distance(self.redshift).to(u.parsec).value / 1e6 # comoving distance to redshift

        theta_z = 1.22 * redshifted_wavelength / b_max
        fwhm = theta_z * comoving_distance
        sigma_gauss = 2.35482004503 * fwhm # conversion factor from https://brainder.org/2011/08/20/gaussian-kernels-convert-fwhm-to-sigma/
        jax.debug.print("sigma_gauss: {}", sigma_gauss)
        key = jax.random.PRNGKey(0)  # Create a random key
        # gaussian_kernel = sigma_gauss * jax.random.normal(key, np.shape(self.brightness_temp))  # Generate a 3x3 array of random normal numbers
        # gaussian_kernel should be the FUNCTIONAL FORM, NORMALIZED
        # integral of smoothed field to be same of integral of field before smoothing!!


        x = np.linspace(-10, 10, 7)
        # Compute the 1D Gaussian PDF with given sigma
        gauss_x = stats.norm.pdf(x, scale=sigma_gauss)  # Apply sigma using the scale parameter

        # Extend to a 3D Gaussian
        gaussian_kernel = gauss_x[:, None, None] * gauss_x[None, :, None] * gauss_x[None, None, :]
        jax.debug.print("np.min(gaussian_kernel): {}", np.min(gaussian_kernel))
        jax.debug.print("np.max(gaussian_kernel): {}", np.max(gaussian_kernel))
        self.brightness_temp = convolve(self.brightness_temp, gaussian_kernel, mode='same')
        # jax.debug.print("brightness temp shape: {}", self.brightness_temp.shape)

    def thermal_noise(self, t_int=1000, area_tot=1e5, delta_theta=10):
        """
        Add thermal noise as in Equation (13) in Gorce+2021
        Default parameters are for SKA-low
        area_tot = 1e5 m^2
        Delta_theta = 10 arcmin
        Delta_nu = simulation frequency resolution (TODO)
        t_int = 1000 hours (TODO)
        """
        phys_length = np.shape(self.brightness_temp)[0] * 1/self.resolution

        # Parameters
        mpc = 3.086e+22 # m
        phys_length = phys_length  # Physical length in Mpc
        rest_freq = 3e8 / 21.0e-2  # Rest wavelength of 21 cm line in Hz

        H_z = cosmo.H(self.redshift).value  # Hubble parameter at redshift km/s/Mpc
        c = 3e5 # in km
        # Compute observed frequency width corresponding to physical length
        obs_freq_width = H_z * rest_freq * np.sqrt(cosmo.Om0) * phys_length
        obs_freq_width /=  (c * np.sqrt(1 + self.redshift))  # this is top of page 8 in Gorce+2021
        obs_freq_width /= 1e6 # in MHz
        jax.debug.print("obs_freq_width {}", obs_freq_width)
        var_th = 2.9 * ((1 + self.redshift)/10)**4.6 * np.sqrt((1e6/obs_freq_width) * (100/t_int))
        jax.debug.print("var_th : {}", np.max(var_th))

        self.sigma_th = np.sqrt(var_th)
        key = jax.random.PRNGKey(0)  # Create a random key
        thermal_noise = self.sigma_th * jax.random.normal(key, np.shape(self.brightness_temp))  # Generate a 3x3 array of random normal numbers
        jax.debug.print("thermal_noise : {}", np.max(thermal_noise))

        self.brightness_temp += thermal_noise

    def cylindrical_spec_normal(self, bins=32):
        """
        just doing a 2D pspec
        """
        curr_side_length = np.shape(self.brightness_temp)[0]
        volume = curr_side_length**3
        
        # take fft of 3D cube
        fft_data = np.fft.fftn(self.brightness_temp) # 3D
        fft_data_squared = np.abs(fft_data ** 2)  # 3D units pixels^6
        jax.debug.print("np.max(fft_data_squared): {}", np.max(fft_data_squared))

        # fft_data_squared_collapsed = np.average(fft_data_squared, axis=0) # collapsing to 2D field, but trying to preserve spatial scale by averaging? 0 is depth
        # getting k values
        k_arr = np.fft.fftfreq(curr_side_length) * 2 * np.pi
        k_arr *= self.resolution  # pixels/side length, changing to Mpc^-1
        jax.clear_caches()
        # jax.debug.print("Current side length: {}", curr_side_length)
        k_0, k_1, k_2 = np.meshgrid(k_arr, k_arr, k_arr)  # 3D
        k_perp = np.sqrt(k_1**2 + k_2**2) # sky plane norm
        k_parallel = np.sqrt(k_0**2) # depth

        # cyl_pspec, kperp_binned, k_parallel_binned = histogram2d(k_perp, k_parallel, weights=fft_data_squared_collapsed, bins=bins) # summing cylinders to get the 2D pspec
        # print(f"resolution, {1 / resolution} Mpc/pixel")  # Mpc/pixel

        cyl_pspec = fft_data_squared / volume  # pixels^6 --> pixels^3
        cyl_pspec *= (1 / self.resolution)**3  # converting form pixels^3 to Mpc^3
        jax.debug.print("np.max(cyl_pspec): {}", np.max(cyl_pspec))

        return cyl_pspec, k_perp, k_parallel

    def wedge(self, plot_debug=True):
        cyl_pspec, k_perp, k_parallel = self.cylindrical_spec_normal()
        psi = 75 * np.pi / 180  # using 75 degrees as nominal value
        k_parallel_wedge = k_perp * np.tan(psi)  # this is (1) from Gagnon-Hartman+2021
        # if plot_debug:
        #     plt.imshow(cyl_pspec[0,:,:])
        #     # plt.plot(k_perp[0,:,:], k_parallel_wedge[0,:,:], ls="--")
        #     plt.xlabel("k_{perp}")
        #     plt.ylabel("k_{par}")
        #     plt.savefig("wedge_example.png", dpi=300)
        wedge_mask = k_parallel < k_parallel_wedge # below this line, we have synchrotron emission, etc. from wedge
        true_count = np.sum(wedge_mask)
        jax.debug.print("Number of True values in wedge_mask: {}", true_count)
        indices = np.where(wedge_mask, size=true_count)
        cyl_pspec = cyl_pspec.at[indices].set(0)
        ### ADD PHASES HERE NP.ANGLE E^(IPHI), TEST WITH A SUPER TINY K WEDGE,
        ### CONSISTENCY CHECK THAT IF
        ### YOU DON'T HAVE ANY CUT NOTHING HAPPENS, CHECK THAT YOU DON'T HAVE TO INVERSE THE NORMALIZATION????
        ### DO EVERYTHING ELSE IN PIXEL SPACE OR LEAVE PSPEC IN PIXEL SPACE

        self.brightness_temp = np.sqrt(np.real(np.fft.ifftn(cyl_pspec)))



        


