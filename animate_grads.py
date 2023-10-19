import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from alternating import InferDens
from jax_battaglia_full import Dens2bBatt

side_length = 256

# grads has one less element than params due to starting field
grads = np.load("/Users/sabrinaberger/just_grad_descent/grad.npy")
params = np.load("/Users/sabrinaberger/just_grad_descent/params.npy")
data = np.load("/Users/sabrinaberger/just_grad_descent/data_adversarial_new_prior_100_bins_both_on" + f"/npy/data_field_8.npy")
truth_field = np.load("/Users/sabrinaberger/just_grad_descent/data_adversarial_new_prior_100_bins_both_on" + f"/npy/truth_field_8.npy").flatten()
isolate_max_index = np.argmax(truth_field)
isolate_min_index = np.argmin(truth_field)

grads_transposed = np.transpose(np.asarray(grads)).flatten()
params_transposed = np.transpose(np.asarray(params)).flatten()

plt.plot(grads_transposed[isolate_max_index], label="max", c="blue")
plt.plot(grads_transposed[isolate_min_index], label="min", c="k")
plt.title("gradients")
plt.legend()
plt.savefig("/Users/sabrinaberger/just_grad_descent/grad.png")
plt.close()

plt.axhline(truth_field[isolate_max_index], label="actual max", ls="--", c="blue")
plt.axhline(truth_field[isolate_min_index], label="actual min", ls="--", c="k")
plt.plot(params_transposed[isolate_max_index], label="max", c="blue")
plt.plot(params_transposed[isolate_min_index], label="min", c="k")
plt.title("density pixels")
plt.legend()
plt.savefig("/Users/sabrinaberger/just_grad_descent/density.png")
plt.close()

fig, ax = plt.subplots()
# ax.axhline(truth_field[isolate_max_index], label="actual max", ls="--", c="blue")
# ax.axhline(truth_field[isolate_min_index], label="actual min", ls="--", c="k")
# ax.plot(grads_transposed[isolate_max_index], params_transposed[isolate_max_index][1:], label="max", c="blue")
# ax.scatter(grads_transposed[isolate_min_index], params_transposed[isolate_min_index][1:], label="min", c="k")
# ax.set_xscale('symlog')
# ax.set_yscale('symlog', linthresh=0.015)
# ax.set_title("density vs grad at corresponding iterations")
# plt.xlabel("gradients")
# plt.ylabel("density field")
# plt.legend()
# plt.savefig("/Users/sabrinaberger/just_grad_descent/density_grad.png")
# plt.close()

# exit()
fig, axes = plt.subplots(1, 2)

divider_1 = make_axes_locatable(axes[0])
cax_1 = divider_1.append_axes('right', '5%', '5%')

divider_2 = make_axes_locatable(axes[1])
cax_2 = divider_2.append_axes('right', '5%', '5%')
# grads = np.transpose(grads)
# params = np.transpose(params)

# grads = np.reshape(grads, (20, 256, 256))
# params = np.reshape(params, (21, 256, 256))

im1 = axes[0].imshow(np.reshape(grads[0], (side_length, side_length)), origin='lower') # Here make an AxesImage rather than contour
cb = fig.colorbar(im1, cax=cax_1)
tx = axes[0].set_title('Frame 0')
fig.colorbar(im1, cax=cax_1, orientation='vertical')

im2 = axes[1].imshow(np.reshape(params[0], (side_length, side_length)), origin='lower') # Here make an AxesImage rather than contour
cb = fig.colorbar(im2, cax=cax_2)
tx = axes[1].set_title('Frame 0')
fig.colorbar(im2, cax=cax_2, orientation='vertical')

def animate(i):
    vmax     = np.max(grads[i])
    vmin     = np.min(grads[i])
    im1.set_clim(vmin, vmax)
    axes[0].set_title(f"Gradients at Iteration # {i}")
    im1.set_data(np.reshape(grads[i], (side_length, side_length)))

    vmax     = np.max(params[i])
    vmin     = np.min(params[i])
    im2.set_clim(vmin, vmax)
    axes[1].set_title(f"Field at Iteration # {i}")
    im2.set_data(np.reshape(params[i], (side_length, side_length)))


plt.tight_layout()

anim = FuncAnimation(fig, animate, frames=20, interval=500, repeat=True)
anim.save("/Users/sabrinaberger/just_grad_descent/test.gif")
plt.close()

likelihoods = []
priors = []
#
# print("getting likelihood")
# samp = InferDens(z=8, num_bins=100, mask_ionized=False, iter_num_max=3,
#                      plot_direc="data_adversarial_new_prior_100_bins_both_on", run_optimizer=False,
#                      weighted_prior=False, new_prior=True, nothing_off=True)
# def get_prior_likelihood_post():
#     likelihoods = []
#     priors = []
#     for g in params:
#         # note param_init was passed in and is a constant
#
#         counts, pspec_box, _ = samp.p_spec_normal(truth_field, samp.num_bins, samp.side_length)
#         discrepancy = data - samp.bias_field(g)
#         likelihoods.append(np.dot(discrepancy.flatten() ** 2, 1. / samp.N_diag))
#         #### prior
#         counts, power_curr, _ = samp.p_spec_normal(g, samp.num_bins, samp.side_length)
#         sigma = counts ** 2
#         x = (pspec_box - power_curr).flatten()
#         priors.append(np.dot(x ** 2, 1 / sigma))
#         return priors, likelihoods
# priors, likelihoods = get_prior_likelihood_post()
# plt.semilogy(priors, label="prior")
# plt.semilogy(likelihoods, label="likelihood")
# plt.legend()
# plt.savefig("prior_likelihood.png")