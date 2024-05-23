import jax.random
import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax.scipy.special import i1 as j_bessel
from jax.experimental.host_callback import id_print, id_tap  # this is a way to print in Jax when things are preself.compiledd

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
    def __init__(self, density, delta_pos, set_z, flow=True, debug=False, b_0=0.593, alpha=0.564, k_0=0.185): # , b_0=0.593, alpha=0.564, k_0=0.185):
        # go into k-space
        self.debug = debug
        self.b_0 = b_0
        self.alpha = alpha
        self.k_0 = k_0
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
            self.k_mags = jnp.sqrt(self.kx ** 2 + self.ky[:,None] ** 2)
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
            self.rs_top_hat_2d_norm = lambda k: 2 * j_bessel(k) / k
            self.rs_top_hat_2d_exp = lambda k: 2 * k / 2 - ((k / 2) ** 3) / 2
            x = jnp.linspace(-3, 3, 7)
            import jax.scipy as jsp

            # self.bias = jnp.where(self.k_mags * self.delta_pos > 1e-6, self.rs_top_hat_2d_norm(self.k_mags),
            #                       self.rs_top_hat_2d_exp(
            #                           self.k_mags))  # issue with NonConcreteBooleans in JAX, need this syntax
        elif self.one_d:
            self.rs_top_hat_1d = lambda arg: self.constant * jnp.sinc(
                arg / jnp.pi)  # engineer's sinc so divide argument by pi
            self.tophatted_ks = self.rs_top_hat_1d(self.k_mags)

        # static
        self.avg_z = 7
        self.b_mz = lambda k: self.b_0 / (1 + k / self.k_0) ** self.alpha # bias factor (8) in paper
        if flow:
            self.flow()


    def apply_filter(self):
        seed = 1010
        seed = jax.random.PRNGKey(seed)
        # jax.random.multivariate_normal(seed, 0, [[0, 0], [1, 0], [0, 1]], dtype=)
        # self.bias = jax.random.normal(seed, jnp.shape(self.k_mags))
        # self.bias = jnp.fft.fftn(self.bias)
        # w_z = self.b_mz(self.k_mags * self.delta_pos)

        w_z = self.b_mz(self.k_mags * self.delta_pos)
        #
        self.density_k *= w_z
        if self.one_d:
            self.density_k *= self.delta_k
        elif self.two_d:
            self.density_k *= self.delta_k**2
        elif self.three_d:
            self.density_k *= self.delta_k**3 # scaling amplitude in fourier space for 3D

        self.delta_z = jnp.fft.ifftn(self.density_k)


        ### SMOOTHING
        # import jax.scipy as jsp
        # x = jnp.linspace(-3, 3, 7)
        # window = jsp.stats.norm.pdf(x) * jsp.stats.norm.pdf(x[:, None])
        # self.delta_z = jsp.signal.convolve(self.delta_z, window, mode='same')
        ###

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

    def get_x_hi(self, tanh=True):
        if tanh:
            if self.two_d:
                # plt.close()
                # plt.imshow(jnp.real(self.z_re))
                # plt.colorbar()
                # plt.title("z_re")
                # plt.colorbar()
                # plt.show()
                self.X_HI = jnp.real(jnp.tanh(self.set_z - self.z_re) + 1) / 2.
                # id_print(self.X_HI)
                # plt.close()
                # plt.imshow(self.X_HI)
                # plt.title("X_HI with tanh")
                # plt.colorbar()
                # plt.show()
                # vectorized version
                # self.X_HI = jnp.where(self.z_re > self.set_z, 0., 1.) # issue with NonConcreteBooleans in JAX, need this syntax

        else:
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