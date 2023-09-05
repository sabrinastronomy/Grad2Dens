import random
from typing import Tuple

import optax
import jax.numpy as jnp
import jax
from jax_main import GradDescent
from jax.experimental.host_callback import id_print  # this is a way to print in Jax when things are preself.compiledd

NUM_TRAIN_STEPS = 1_000


holder = GradDescent(12, include_field=True, autorun=False, plot_direc="optax_plots")
holder.check_field(holder.truth_field, "truth_field", show=True, save=True)
holder.check_field(holder.data, "truth_field", show=True, save=True)

initial_params = holder.s_field
TRAINING_DATA = holder.data

def loss(params: optax.Params) -> jnp.ndarray:
    loss_value = holder.chi_sq_jax(params)
    # id_print(loss_value)
    return loss_value


def fit(params: optax.Params, optimizer: optax.GradientTransformation) -> optax.Params:
    opt_state = optimizer.init(params)

    @jax.jit
    def step(params, opt_state):
        loss_value, grads = jax.value_and_grad(loss)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_value
    for i in range(NUM_TRAIN_STEPS):
        params, opt_state, loss_value = step(params, opt_state)
    print(params)
    return params

# Finally, we can fit our parametrized function using the Adam optimizer
# provided by optax.
optimizer = optax.rmsprop(learning_rate=10)
params = fit(initial_params, optimizer)
import matplotlib.pyplot as plt
plt.imshow(jnp.reshape(params, (256, 256)))
plt.colorbar()
plt.savefig("output.png")