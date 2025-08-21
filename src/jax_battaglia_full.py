import jax.random
import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax.scipy.special import i1 as j_bessel
# from jax.experimental.host_callback import id_print, \
#     id_tap  # this is a way to print in Jax when things are preself.compiledd
# try:
#     from .ska_effects import SKAEffects # THIS IS FOR WHEN USING IN JUPYTER NOTEBOOK
# except:
#     from ska_effects import SKAEffects



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

    def __init__(self, density, set_z, physical_side_length, resolution, flow=True, debug=False, free_params={},
                 apply_ska=False):  # b_0=0.5, alpha=0.2, k_0=0.1):         # go into k-space
        self.debug = debug
        self.density = density + 1
        self.dim = self.density.ndim
        self.set_z = set_z
        self.physical_side_length = physical_side_length
        self.density = density
        self.b_0 = free_params['b_0']
        self.alpha = free_params['alpha']
        self.k_0 = free_params['k_0']
        self.tanh_slope = free_params['tanh_slope']
        self.avg_z = free_params['avg_z']
        self.apply_ska = apply_ska
        self.resolution = resolution

        self.integrand = self.density
        self.cube_len = jnp.shape(self.density)[0]

        assert (self.resolution == self.physical_side_length / self.cube_len)  # assert resolution in Mpc/pixel

        if self.dim == 1:
            self.one_d = True
            self.two_d = False
            self.three_d = False
        elif self.dim == 2:
            self.one_d = False
            self.two_d = True
            self.three_d = False
        elif self.dim == 3:
            self.one_d = False
            self.two_d = False
            self.three_d = True
        else:
            print("Unsupported field dimensions!")
            exit()

        if self.one_d:
            self.ks = jnp.fft.fftfreq(
                len(self.density)) * 2 * jnp.pi * 1 / self.resolution  # follow Fourier conventions and convert from pixel^-1 to Mpc^-1
            self.k_mags = jnp.abs(self.ks)
            self.X_HI = jnp.empty(self.cube_len)
            self.delta_k = self.ks[1] - self.ks[0]

        elif self.two_d:  # assuming 2D
            self.kx = self.ky = jnp.fft.fftfreq(self.cube_len, d=self.resolution/2./jnp.pi)  # follow Fourier conventions and convert from pixel^-1 to Mpc^-1
            k1, k2 = jnp.meshgrid(self.kx, self.ky)  # 3D!! meshgrid :)
            self.k_mags = jnp.sqrt(k1 ** 2 + k2 ** 2)
            # self.delta_k = self.kx[1] - self.kx[0]

        elif self.three_d:  # assuming 3D
            self.kx = self.ky = self.kz = jnp.fft.fftfreq(
                self.cube_len) * 2 * jnp.pi * 1 / self.resolution  # follow Fourier conventions and convert from pixel^-1 to Mpc^-1

            k1, k2, k3 = jnp.meshgrid(self.kx, self.ky, self.kz)  # 3D!! meshgrid :)
            self.k_mags = jnp.sqrt(k1 ** 2 + k2 ** 2 + k3 ** 2)

            self.X_HI = jnp.empty((self.cube_len, self.cube_len, self.cube_len))
            self.delta_k = self.kx[1] - self.kx[0]

        self.density_k = jnp.fft.fftn(self.integrand, norm='ortho') * self.resolution ** (self.density.ndim / 2)  # pixels^n -> Mpc^n, get normalization right
        assert self.density_k.ndim == self.dim
        self.rs_top_hat_3d = lambda k: 3 * (jnp.sin(k) - jnp.cos(k) * k) / k ** 3  # pixelate and smoothing?
        self.rs_top_hat_3d_exp = lambda k: 1 - k ** 2 / 10
        self.rs_top_hat_1d = lambda arg: jnp.sinc(arg / jnp.pi)  # engineer's sinc so divide argument by pi
        # if self.two_d:
        #     self.rs_top_hat_2d_norm = lambda k: 2 * j_bessel(k) / k
        #     self.rs_top_hat_2d_exp = lambda k: 2 * k / 2 - ((k / 2) ** 3) / 2
        #     x = jnp.linspace(-3, 3, 7)
        #     import jax.scipy as jsp
        #
        #     # self.bias = jnp.where(self.k_mags * self.delta_pos > 1e-6, self.rs_top_hat_2d_norm(self.k_mags),
        #     #                       self.rs_top_hat_2d_exp(
        #     #                           self.k_mags))  # issue with NonConcreteBooleans in JAX, need this syntax
        # elif self.one_d:
        #     self.rs_top_hat_1d = lambda arg: self.constant * jnp.sinc(
        #         arg / jnp.pi)  # engineer's sinc so divide argument by pi
        #     self.tophatted_ks = self.rs_top_hat_1d(self.k_mags)

        # static
        self.b_mz = lambda k: self.b_0 / (1 + k / self.k_0) ** self.alpha  # bias factor (8) in paper
        if flow:
            self.flow()

    def apply_filter(self):
        w_z = self.b_mz(self.k_mags)
        self.density_k *= w_z
        # if self.one_d:
        #     self.density_k *= self.delta_k
        # elif self.two_d:
        #     self.density_k *= self.delta_k**2
        # elif self.three_d:
        #     self.density_k *= self.delta_k**3 # scaling amplitude in fourier space for 3D

        self.delta_z = jnp.fft.ifftn(self.density_k, norm='ortho')
        ### SMOOTHING
        # import jax.scipy as jsp
        # x = jnp.linspace(-3, 3, 7)
        # window = jsp.stats.norm.pdf(x) * jsp.stats.norm.pdf(x[:, None])
        # self.delta_z = jsp.signal.convolve(self.delta_z, window, mode='same')
        ###

        # if self.one_d:
        #     self.delta_z *= self.cube_len / (2*jnp.pi)
        # elif self.two_d:
        #     self.delta_z *= (self.cube_len**2) / (2*jnp.pi)**2
        # elif self.three_d:
        #     self.delta_z *= (self.cube_len**3)/(2*jnp.pi)**3 # weird FFT scaling for 3D, getting rid of 1/n^3

        if self.debug and self.two_d:
            plt.close()
            plt.imshow(jnp.real(self.delta_z))
            plt.title("delta z smoothed density field")
            plt.colorbar()
            plt.show()

    def get_z_re(self):  # average z: order of half ionized half neutral
        self.z_re = jnp.real(self.delta_z)
        # self.z_re = self.z_re - jnp.mean(self.z_re)
        self.z_re = (1 + self.avg_z) * self.z_re + self.avg_z

        if self.debug and self.two_d:
            plt.close()
            plt.imshow(jnp.real(self.z_re))
            plt.colorbar()
            plt.title("z_re")
            plt.show()
        if self.debug and self.three_d:
            plt.close()
            plt.imshow(jnp.real(self.z_re[0, :, :]))
            plt.colorbar()
            plt.title("z_re")
            plt.savefig("debug_ska/z_re.png")
            plt.close()

    def get_x_hi(self, tanh=True):
        if tanh:
            print("USING TANH SLOPE")
            # self.z_re = jnp.real(self.z_re)
            self.X_HI = (jnp.tanh(self.tanh_slope * (self.set_z - self.z_re)) + 1.) / 2.
        else:
            self.X_HI = jnp.where(self.z_re > self.set_z, 0.,
                                  1.)  # issue with NonConcreteBooleans in JAX, need this syntax

        return self.X_HI

    def get_temp_brightness(self):
        # first = 27 * self.X_HI
        # second = 1 + self.density
        # self.temp_brightness = first * second

        OMb, OMm, h = 0.04897, 0.30966, 0.6766
        # xHI_box = (jnp.tanh((-zre_box + z) * zslope) + 1.) / 2

        cosmo_prefac = 27. * OMb * h ** 2 / 0.023 * \
                       jnp.sqrt(0.15 / OMm / h ** 2) * jnp.sqrt((1. + self.set_z) / 10.)
        self.temp_brightness = cosmo_prefac * self.X_HI * (1. + self.density)

        # self.temp_brightness = self.normalize(self.temp_brightness)

        # print("---------------")
        # print("residual for FFT check")
        # jax.debug.print('"residual for FFT check"? {}',jnp.sum(self.density - self.delta_z))
        # print("---------------")

    # def normalize(self, field):
    #     min_field = jnp.min(field)
    #     max_field = jnp.max(field)
    #     diff = max_field - min_field
    #     return (field - min_field) / diff

    def flow(self):
        self.apply_filter()
        self.get_z_re()
        self.get_x_hi()
        self.get_temp_brightness()
        assert self.temp_brightness.ndim == self.dim
        if self.apply_ska:
            self.ska = SKAEffects(self.temp_brightness, add_all=True, redshift=self.set_z,
                                  physical_side_length=self.physical_side_length)
            self.temp_brightness = self.ska.brightness_temp
            assert self.temp_brightness.ndim == self.dim


 # Example usage:
if __name__ == "__main__":
    k_0_fiducial = 0.185 * 0.676  # changing from Mpc/h to Mpc
    alpha_fiducial = 0.564
    b_0_fiducial = 0.593
    midpoint_z_fiducial = 7
    tanh_fiducial = 2
    fiducial_params = {"b_0": b_0_fiducial, "alpha": alpha_fiducial, "k_0": k_0_fiducial, "tanh_slope": tanh_fiducial,
                       "avg_z": midpoint_z_fiducial}  # b_0=0.5, alpha=0.2, k_0=0.1)
    # Example initialization - adjust parameters as needed
    ## should only be used for setting an initial test field
    import powerbox as pbox
    pb = pbox.PowerBox(
        N=256,  # number of wavenumbers
        dim=2,  # dimension of box
        pk=lambda k: 0.1 * k ** -3.5,  # The power-spectrum
        boxlength=256,  # Size of the box (sets the units of k in pk)
        seed=1010,  # Use the same seed as our powerbox
        # ensure_physical=True
    )

    model = Dens2bBatt(
        free_params=fiducial_params,
        density=pb.delta_x(),
        set_z=6.0,
        physical_side_length=256,
        resolution=1
    )

    # Process the data
    dTb_box = model.temp_brightness
    plt.imshow(dTb_box)
    plt.colorbar()
    plt.show()
    print(f"dTb_box shape: {dTb_box.shape}")
    # print(f"zre_box shape: {zre_box.shape}")