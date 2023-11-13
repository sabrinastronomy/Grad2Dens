import powerbox as pbox
import jax.numpy as jnp

def create_better_normal_field(self, seed):
    self.pspec_true = theory_matter_ps.get_truth_matter_pspec(self.z)
    ## should only be used for setting an initial test field
    self.pb = pbox.PowerBox(
        N=self.side_length,  # number of wavenumbers
        dim=self.dim,  # dimension of box
        pk=lambda k: self.pspec_true(self.z, k),  # The power-spectrum
        boxlength=self.side_length,  # Size of the box (sets the units of k in pk)
        seed=seed,  # Use the same seed as our powerbox
        a=0,  # a and b need to be set like this to properly match numpy's fft
        b=2 * jnp.pi,
        ensure_physical=True
    )
    return self.pb