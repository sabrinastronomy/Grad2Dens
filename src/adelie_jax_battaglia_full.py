import jax
# try:
#     from .ska_effects import SKAEffects  # THIS IS FOR WHEN USING IN JUPYTER NOTEBOOK
# except:
#     from ska_effects import SKAEffects
# from .theory_matter_ps import spherical_p_spec_normal, circular_spec_normal, get_truth_matter_pspec
try:
    from .jax_utils import make_initial_density, bias2brightness_2d, bias2zre_2d, ps1d_2d
except:
    from jax_utils import make_initial_density, bias2brightness_2d, bias2zre_2d, ps1d_2d

import jax.numpy as jnp
import matplotlib.pyplot as plt
# You'll need to import or define these missing functions:
# from your_module import bias2brightness_2d, get_truth_matter_pspec
# import powerbox as pbox

class Dens2bBatt:
    """
    This class follows the Battaglia et al (2013) model to go from a density field to a temperature brightness field.
    """

    def __init__(self, density, set_z, physical_side_length, resolution, free_params, flow=True, debug=False,
                 cosmo_params=[0.04897, 0.30966, 0.6766], zslope=50,
                 apply_ska=False):
        # Store input parameters
        self.density = density
        self.set_z = set_z
        self.L = physical_side_length
        self.n = int(physical_side_length / resolution)

        self.resolution = resolution
        self.flow = flow
        self.debug = debug
        self.apply_ska = apply_ska
        self.cosmo_params = cosmo_params

        self.b_0 = free_params['b_0']
        self.alpha = free_params['alpha']
        self.k_0 = free_params['k_0']
        self.zslope = free_params['tanh_slope']
        self.avg_z = free_params['avg_z']
        self.astro_params = [self.b_0, self.k_0, self.alpha] # b0, logk0, alpha

        # Initialize model parameters
        self.ndim = 2  # works on 2d only
        # Set up model parameters dictionary
        self.model_params = {
            'astro_params': self.astro_params,
            'zre': self.avg_z,
            'L': self.L,
            'n': self.n,
            'ndim': self.ndim,
            'zslope': self.zslope,
            'cosmo_params': self.cosmo_params,
            'zdata': self.set_z
        }

        # Initialize the model
        self.process_data()


    def get_model_data(self, dens_model):
        """
        Generate model data from density model
        """
        shape = tuple([self.n for i in range(self.ndim)])

        # You'll need to implement or import bias2brightness_2d
        dTb_box, zre_box, XHI_box = bias2brightness_2d(
            self.model_params['zdata'],
            jnp.reshape(dens_model, shape, order='C'),
            self.model_params['L'],
            self.model_params['astro_params'],
            self.model_params['zre'],
            self.model_params['cosmo_params'],
            zslope=self.model_params['zslope']
        )
        return dTb_box, zre_box, XHI_box

    def process_data(self):
        """
        Main processing function to generate brightness temperature field
        """
        # Generate true density field
        # print('\nGenerating mock data...')
        true_dens = self.density.reshape(self.n ** self.ndim, order='C')
        # print(f'Mean of true density field: {true_dens.mean():.3e}')

        # Update model parameters with true density
        self.model_params.update({'true_dens': true_dens})

        # Generate mock data
        dTb_box, zre_box, xHI_box = self.get_model_data(true_dens)
        self.X_HI = xHI_box
        self.z_re = zre_box
        self.temp_brightness = dTb_box


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
    # print(f"dTb_box shape: {dTb_box.shape}")
    # print(f"zre_box shape: {zre_box.shape}")