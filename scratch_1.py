import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1 import make_axes_locatable

grads = np.load("/Users/sabrinaberger/just_grad_descent/grad.npy")
params = np.load("/Users/sabrinaberger/just_grad_descent/params.npy")
fig, axes = plt.subplots(1, 2)

divider_1 = make_axes_locatable(axes[0])
cax_1 = divider_1.append_axes('right', '5%', '5%')

divider_2 = make_axes_locatable(axes[1])
cax_2 = divider_2.append_axes('right', '5%', '5%')

im1 = axes[0].imshow(grads[0], origin='lower') # Here make an AxesImage rather than contour
cb = fig.colorbar(im1, cax=cax_1)
tx = axes[0].set_title('Frame 0')
fig.colorbar(im1, cax=cax_1, orientation='vertical')

im2 = axes[1].imshow(params[0], origin='lower') # Here make an AxesImage rather than contour
cb = fig.colorbar(im2, cax=cax_2)
tx = axes[1].set_title('Frame 0')
fig.colorbar(im2, cax=cax_2, orientation='vertical')

def animate(i):
    vmax     = np.max(grads[i])
    vmin     = np.min(grads[i])
    im1.set_clim(vmin, vmax)
    axes[0].set_title(f"Gradients at Iteration # {i}")
    im1.set_data(grads[i])

    vmax     = np.max(params[i])
    vmin     = np.min(params[i])
    im2.set_clim(vmin, vmax)
    axes[1].set_title(f"Field at Iteration # {i}")
    im2.set_data(params[i])


plt.tight_layout()

anim = FuncAnimation(fig, animate, frames=20, interval=500, repeat=True)
anim.save("/Users/sabrinaberger/just_grad_descent/test.gif")
plt.close()


def get_prior_likelihood_post(self):
    self.like_arr = np.zeros(self.iter_num_max)
    self.prior_arr = np.zeros(self.iter_num_max)

    truth_field = np.load(self.plot_direc + f"/npy/truth_field_{self.z}.npy")
    data = np.load(self.plot_direc + f"/npy/data_field_{self.z}.npy")
    counts, pspec_box, _ = self.p_spec_normal(truth_field, self.num_bins, self.side_length)
    rest_num = 0
    labels = []
    for iter_num_big in range(self.iter_num_max):
        ### annoying labels
        if iter_num_big == 0:
            if self.nothing_off:
                self.prior_off = False
                self.likelihood_off = False
            else:
                # first iteration, we keep just the likelihood on
                self.prior_off = True
                self.likelihood_off = False

        else:
            if not self.nothing_off:  # if both on this is skipped
                if rest_num == self.rest_num_max or self.prior_off:
                    self.prior_off = not self.prior_off
                    self.likelihood_off = not self.likelihood_off
                    rest_num = 0
        rest_num += 1
        if not self.likelihood_off and not self.prior_off:
            self.plot_title = "both_on"
        elif self.likelihood_off:
            self.plot_title = "likelihood_off"
        elif self.prior_off:
            self.plot_title = "prior_off"
        else:
            print("Something is wrong as neither the likelihood or prior is on.")
            exit()

        best_field = np.load(self.plot_direc + f"/npy/{self.plot_title}_best_field_{self.z}_{iter_num_big}.npy")

        candidate_field = jnp.reshape(best_field, self.size)

        # note param_init was passed in and is a constant
        discrepancy = data - self.bias_field(candidate_field)
        likelihood = jnp.dot(discrepancy.flatten() ** 2, 1. / self.N_diag)
        self.like_arr[iter_num_big] = likelihood
        #### prior
        counts, power_curr, _ = self.p_spec_normal(candidate_field, self.num_bins, self.side_length)
        sigma = counts ** 2
        x = (pspec_box - power_curr).flatten()
        prior = jnp.dot(x ** 2, 1 / sigma)
        self.prior_arr[iter_num_big] = prior
        labels.append(self.plot_title)

    print("final prior array")
    print(self.prior_arr)

    self.labels = labels
    np.save(self.plot_direc + f"/npy/prior_arr_{self.z}.npy", self.prior_arr)
    np.save(self.plot_direc + f"/npy/like_arr_{self.z}.npy", self.like_arr)

plt.semilogy(prior, label="prior")
plt.semilogy(likelihood, label="likelihood")
plt.legend()
plt.savefig("prior_likelihood.png")