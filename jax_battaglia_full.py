import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax
from scipy.special import spherical_jn as j_bessel

# Toy Density Field
# nx, ny = (5, 5)
# x = jnp.linspace(0, 1, nx)
# y = jnp.linspace(0, 1, ny)
# xv, yv = jnp.meshgrid(x, y)

# TESTING: density3D = jnp.random.normal(0, 0.1, (64, 64, 64)) # numpy.random.normal(loc=0.0, scale=1.0, size=None)
# density1D = jnp.random.normal(0, 1, 64) # numpy.random.normal(loc=0.0, scale=1.0, size=None)

class Dens2bBatt:
    """
    This class follows the Battaglia et al (2013) model to go from a density field to a temperature brightness field.
    """
    def __init__(self, density, delta_pos, set_z, flow=True, debug=False):
        # go into k-space
        self.debug = debug
        if density.ndim == 1:
            self.one_d = True
            self.two_d = False
            self.three_d = False
        elif density.ndim == 2:
            self.one_d = False
            self.two_d = True
            self.three_d = False
        elif density.ndim == 3:
            self.one_d = False
            self.two_d = False
            self.three_d = True
        else:
            print("Unsupported field dimensions!")
            exit()

        if debug:
            print(f"Density dimensions are {density.ndim}")
        self.density = density
        self.set_z = set_z
        self.delta_pos = delta_pos  # Mpc

        if self.one_d:
            self.cube_len = len(self.density)
            self.integrand = self.density * self.delta_pos  # weird FFT scaling for 1D
            self.ks = jnp.fft.fftfreq(len(self.density), self.delta_pos)
            self.k_mags = jnp.abs(self.ks)
            self.X_HI = jnp.empty(self.cube_len)
            self.delta_k = self.ks[1] - self.ks[0] 

        elif self.two_d: # assuming 2D
            self.cube_len = len(self.density[:, 0])
            self.integrand = self.density * (self.delta_pos**2) # weird FFT scaling for 2D
            self.kx = jnp.fft.fftfreq(self.density.shape[0], self.delta_pos)
            self.ky = jnp.fft.fftfreq(self.density.shape[1], self.delta_pos)
            self.kx *= 2 * jnp.pi  # scaling k modes correctly
            self.ky *= 2 * jnp.pi  # scaling k modes correctly
            self.k_mags = jnp.sqrt(self.kx ** 2 + self.ky ** 2)
            self.delta_k = self.kx[1] - self.kx[0]
            self.xdim = self.ydim = self.cube_len

        elif self.three_d: # assuming 3D
            self.cube_len = len(self.density[:, 0, 0])
            self.integrand = self.density * (self.delta_pos**3) # weird FFT scaling for 3D
            self.kx = jnp.fft.ifftshift(jnp.fft.fftfreq(self.density.shape[0], self.delta_pos))
            self.ky = jnp.fft.ifftshift(jnp.fft.fftfreq(self.density.shape[1], self.delta_pos))
            self.kz = jnp.fft.ifftshift(jnp.fft.fftfreq(self.density.shape[2], self.delta_pos))

            self.kx *= 2 * jnp.pi  # scaling k modes correctly
            self.ky *= 2 * jnp.pi  # scaling k modes correctly
            self.kz *= 2 * jnp.pi  # scaling k modes correctly

            self.k_mags = jnp.sqrt(self.kx ** 2 + self.ky ** 2 + self.kz ** 2)
            self.X_HI = jnp.empty((self.cube_len, self.cube_len, self.cube_len))
            self.delta_k = self.kx[1] - self.kx[0]  
            self.xdim = self.ydim = self.zdim = self.cube_len

        self.density_k = jnp.fft.fftn(self.integrand)

        self.rs_top_hat_3d = lambda k: 3 * (jnp.sin(k) - jnp.cos(k) * k) / k ** 3  # pixelate and smoothing?
        self.rs_top_hat_3d_exp = lambda k: 1 - k ** 2 / 10
        self.rs_top_hat_1d = lambda arg: jnp.sinc(arg / jnp.pi)  # engineer's sinc so divide argument by pi
        if self.two_d:
            self.rs_top_hat_2d_norm = lambda k: 2 * (jnp.sqrt(jnp.pi/(2*k)) * jax.scipy.special.i1(k)) / k # jax modified bessel function of the first kind, TODO this is modified
            self.rs_top_hat_2d_exp = lambda k: 2 * k / 2 - ((k / 2) ** 3) / 2
            self.bias = jnp.where(self.k_mags * self.delta_pos > 1e-6, self.rs_top_hat_2d_norm(self.k_mags),
                                  self.rs_top_hat_2d_exp(
                                      self.k_mags))  # issue with NonConcreteBooleans in JAX, need this syntax
        elif self.one_d:
            self.constant = 1
            print("Unsure what this constant should be in 1D?")
            self.rs_top_hat_1d = lambda arg: self.constant * jnp.sinc(
                arg / jnp.pi)  # engineer's sinc so divide argument by pi
            self.tophatted_ks = self.rs_top_hat_1d(self.k_mags)

        # static
        self.avg_z = 7
        self.b_0 = 0.593
        self.alpha = 0.564
        self.k_0 = 0.185
        self.b_mz = lambda k: self.b_0 / (1 + k / self.k_0) ** self.alpha # bias factor (8) in paper
        if flow:
            self.flow()


    def apply_filter(self):
        w_z = 1 # debug
        w_z = self.b_mz(self.k_mags * self.delta_pos)
        # w_z = self.b_mz(self.k_mags * self.delta_pos) * self.bias

        self.density_k *= w_z
        if self.one_d:
            self.density_k *= self.delta_k
        elif self.two_d:
            self.density_k *= self.delta_k**2
        elif self.three_d:
            self.density_k *= self.delta_k**3 # scaling amplitude in fourier space for 3D

        self.delta_z = jnp.fft.ifftn(self.density_k)

        if self.one_d:
            self.delta_z *= self.cube_len / (2*jnp.pi)
        elif self.two_d:
            self.delta_z *= (self.cube_len**2) / (2*jnp.pi)**2
        elif self.three_d:
            self.delta_z *= (self.cube_len**3)/(2*jnp.pi)**3 # weird FFT scaling for 3D, getting rid of 1/n^3

        if self.debug and self.two_d:
            plt.close()
            plt.imshow(jnp.real(self.delta_z))
            plt.title("delta z smoothed density field")
            plt.colorbar()
            plt.show()

    def get_z_re(self): # average z: order of half ionized half neutral
        self.z_re = self.delta_z * (1 + self.avg_z) + (1 + self.avg_z) - 1

        if self.debug and self.two_d:
            plt.close()
            plt.imshow(jnp.real(self.z_re))
            plt.colorbar()
            plt.title("z_re")
            plt.show()

    def get_x_hi(self):
        if self.one_d:
            i = 0
            for z in self.z_re:
                if z < self.set_z:
                    self.X_HI = self.X_HI.at[i].set(0)
                else:
                    self.X_HI = self.X_HI.at[i].set(1)
                i += 1
        elif self.two_d:
            # vectorized version
            self.X_HI = jnp.where(self.z_re > self.set_z, 0., 1.) # issue with NonConcreteBooleans in JAX, need this syntax
        else:
            raise NotImplementedError
        return self.X_HI

    def get_temp_brightness(self):
        first = 27 * self.X_HI
        second = 1 + self.density
        self.temp_brightness = first*second


    def flow(self):
        self.apply_filter()
        self.get_z_re()
        self.get_x_hi()
        self.get_temp_brightness()



if __name__ == "__main__":
    import jax_main
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    rho = jnp.asarray([1.92688, -0.41562])
    z = 15
    dens2Tb = Dens2bBatt(rho, 1, z)
    Tb = dens2Tb.temp_brightness
    print(Tb)

    for z in [7]:

        # dir = "/Users/sabrinaberger/Library/Mobile Documents/com~apple~CloudDocs/CosmicDawn/building/21cmFASTBoxes_{}/PerturbedField_*".format(z)
        # hf = h5py.File(glob.glob(dir)[0], 'r')
        # data = hf["PerturbedField/density"]
        seed = 1010
        minimizer_functions = jax_main.SwitchMinimizer(seed=seed, z=14, num_bins=128, create_instance=True)
        density = minimizer_functions.create_better_normal_field(seed=seed).delta_x()
        print(jnp.shape(density))
        #### TO RUN A DENSITY FIELD TO TEMPERATURE BRIGHTNESS CONVERSION
        dens2Tb = Dens2bBatt(density, 1, z, debug=True)
        Tb = dens2Tb.temp_brightness
        z_re = dens2Tb.z_re
        ####
        # slice_rho = density2D[:, :, 0]
        # slice_delta_z = dens2Tb.delta_z[:, :, 0]
        # slice_Tb = Tb[:, :, 0]

        plt.close()
        fig, (ax1, ax2) = plt.subplots(ncols=2)
        im1 = ax1.imshow(density)
        divider = make_axes_locatable(ax1)
        cax1 = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im1, cax=cax1)
        ax1.set_title(r"$\delta_{\rho}$" + ", z = {}".format(z))

        im2 = ax2.imshow(Tb)
        divider = make_axes_locatable(ax2)
        cax2 = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im2, cax=cax2)
        ax2.set_title(r"$\delta T_b$" + ", z = {}".format(z))
        plt.tight_layout(h_pad=1)
        fig.savefig("uni_rho_T_b_{}.png".format(z))
        plt.show()
        plt.close()

        print(jnp.min(Tb))