from numpyro.infer import HMC, HMCECS, MCMC, NUTS, SVI, Trace_ELBO, autoguide
import numpyro
import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from numpyro.distributions.distribution import Distribution
from numpyro.distributions.constraints import positive
from numpyro.infer import Predictive
from numpyro import distributions as dist, infer
from src import alternating, jax_battaglia_full, theory_matter_ps, jax_main
from numpyro.infer.initialization import init_to_value


k_0_fiducial = 0.185 * 0.676 # changing from Mpc/h to Mpc
alpha_fiducial = 0.564
b_0_fiducial = 0.593
midpoint_z_fiducial = 7
tan_fiducial = 2

fiducial_params = {"b_0": b_0_fiducial, "alpha": alpha_fiducial, "k_0": k_0_fiducial, "tanh_slope": tan_fiducial,
                   "avg_z": midpoint_z_fiducial}  # b_0=0.5, alpha=0.2, k_0=0.1)

static_redshift = True
ska_effects = True
nominal = True

bins = 8
side_length = 16
physical_side_length = 16
z = 14

key = jax.random.PRNGKey(0)
starting_field = jax.random.normal(key, (side_length, side_length))

iter_num_max = 1
rest_num_max = 1

nothing_off = True

dimensions = 2
cov_matrix_data = False
truth_field = []
brightness_temperature_field = []

params = alternating.ConfigParam(ska_effects=ska_effects, free_params=fiducial_params, z=z, truth_field=truth_field,
                         brightness_temperature_field=brightness_temperature_field, num_bins=bins,
                         nothing_off=nothing_off, plot_direc="", side_length=side_length,
                         physical_side_length=physical_side_length,
                         dimensions=dimensions, iter_num_max=iter_num_max, rest_num_max=rest_num_max, noise_off=True, new_prior=True, cov_matrix_data=False)

DensityBlock = alternating.InferDens(config_params=params, s_field=None)

def generate_density_cov_matrix(n, samples=1000, cov_input_old=None):
    key = jax.random.PRNGKey(0)  # '0' is the seed for reproducibility
    cov_input = jnp.empty((samples, n**dimensions))
    # Generate a random integer between 0 and 1000
    for i in range(samples): # generate n draws of ND data
        key, subkey = jax.random.split(key)  # Update key to ensure new randomness
        random_int = jax.random.randint(subkey, shape=(), minval=0, maxval=samples*4)
        pb_data_unbiased_field = DensityBlock.create_better_normal_field(seed=random_int).delta_x()
        mock_truth_field = jnp.asarray(pb_data_unbiased_field).flatten()
        cov_input = cov_input.at[i, :].set(mock_truth_field) # (num_sample, field)
    cov_input_transposed = cov_input.T # inverting to be (field, num_sample) since we want (random variable, samples)
    if cov_input_old != None:
        cov_input_transposed = jnp.concatenate([cov_input_old, cov_input_transposed], axis=0)
        print("Updating old")
    return cov_input_transposed

### 1) create latent space (density field)
L = DensityBlock.config_params.physical_side_length
n = DensityBlock.config_params.side_length
# DensityBlock.config_params.dim = 2

# truth field is just unbiased version made with pbox
truth_field = DensityBlock.truth_field
data = DensityBlock.data
raw_data = DensityBlock.raw_data

cov_matrix_input = generate_density_cov_matrix(side_length)
cov_matrix_DENSITY = jnp.cov(cov_matrix_input)

#### checking conditioning/stability
print("C^-1 C ~ I?")
difference_cov = jnp.matmul(jnp.linalg.inv(cov_matrix_DENSITY), cov_matrix_DENSITY)
print(difference_cov)
# Compute the error from the identity matrix
identity_error = jnp.linalg.norm(difference_cov - jnp.eye(difference_cov.shape[0]))
print(identity_error)

det = jnp.linalg.det(cov_matrix_DENSITY)
print("Determinant:", det)

eigenvalues = jnp.linalg.eigvalsh(cov_matrix_DENSITY)

class MultivariateLogNormal(Distribution):
    support = positive
    reparametrized_params = ["loc", "covariance_matrix"]

    def __init__(self, loc, covariance_matrix, validate_args=None):
        self.loc = loc
        self.covariance_matrix = covariance_matrix
        self._mvn = dist.MultivariateNormal(loc=loc, covariance_matrix=covariance_matrix)
        batch_shape = self._mvn.batch_shape
        event_shape = self._mvn.event_shape
        super().__init__(batch_shape=batch_shape, event_shape=event_shape, validate_args=validate_args)

    def sample(self, key, sample_shape=()):
        z = self._mvn.sample(key, sample_shape)
        return jnp.exp(z) # exponentiating normal gives you log normal

    def log_prob(self, value):
        log_value = jnp.log(value)
        base_log_prob = self._mvn.log_prob(log_value)
        jacobian = jnp.sum(log_value, axis=-1)
        return base_log_prob - jacobian
# https://mc-stan.org/docs/2_21/stan-users-guide/changes-of-variables.html

def bayesian_model(obs=None):
    density = numpyro.sample("theory", MultivariateLogNormal(loc=0.0, covariance_matrix=cov_matrix_DENSITY)) # SECOND PARAM IS SIGMA
    # density = numpyro.sample("theory", dist.Normal(0, 3).expand((n, n))) # SECOND PARAM IS SIGMA
    density = jnp.reshape(density, (n, n))
    batt_model_instance = jax_battaglia_full.Dens2bBatt(density, resolution=L/n, set_z=z, physical_side_length=L, flow=True, free_params=fiducial_params, apply_ska=DensityBlock.config_params.ska_effects)
    inferred_data = batt_model_instance.temp_brightness
    numpyro.sample("data", dist.Normal(inferred_data, 0.1), obs=obs)


# Using the model above, we can now sample from the posterior distribution using the No
# U-Turn Sampler (NUTS).
nuts_kernel = infer.NUTS(bayesian_model, init_strategy=init_to_value(values={"density": starting_field}))

sampler = infer.MCMC(
    nuts_kernel,
    num_warmup=2000,
    num_samples=10000,
    num_chains=4,
    progress_bar=True,
)

sampler.run(jax.random.PRNGKey(0), obs=data)