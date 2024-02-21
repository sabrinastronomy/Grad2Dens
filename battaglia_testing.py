import numpy as np
import matplotlib.pyplot as plt
from jax_battaglia_full import Dens2bBatt
import powerbox as pbox
from matplotlib.ticker import MaxNLocator
from jax_battaglia_full import Dens2bBatt
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable
from theory_matter_ps import circular_spec_normal, after_circular_spec_normal

side_length = 512
physical_side_length = 256
seed = 1010
dim = 2
set_z = 8

pb = pbox.LogNormalPowerBox(
    N=side_length,  # number of wavenumbers
    dim=dim,  # dimension of box
    pk=lambda k: 0.1 * k**-2,  # The power-spectrum
    boxlength=physical_side_length,  # Size of the box (sets the units of k in pk)
    seed=seed  # Use the same seed as our powerbox
    # ensure_physical=True
)
# fit versions, b_0=0.593, alpha=0.564, k_0=0.185
alphas = np.linspace(0.1, 2, 5)
b_0s = np.linspace(0.1, 2, 5)
k_0s = np.linspace(0.1, 2, 5)

# alpha_mesh, b_mesh, k_mesh = np.meshgrid(alphas, b_0s, k_0s)
battaglia_testing = "battaglia_testing"

plt.imshow(pb.delta_x())
plt.colorbar()
plt.title("density")
plt.savefig(battaglia_testing+"/delta_x.png")
plt.close()
fig, axes = plt.subplots(1, 5, figsize=(16, 8))

for i in range(len(alphas)):
    b_0 = 0.593
    k_0 = 0.185
    # for j in range(len(b_0s)):
    #     for k in range(len(k_0s)):
    # b_0 = b_mesh[i, j, k]
    # k_0 = k_mesh[i, j, k]
    batt_model_instance = Dens2bBatt(pb.delta_x(), delta_pos=1, set_z=set_z, flow=True, alpha=alphas[i], b_0=b_0, k_0=k_0)
    im = axes[i].imshow(batt_model_instance.temp_brightness)
    divider = make_axes_locatable(axes[i])
    cax = divider.append_axes('bottom', size='5%', pad=0.05)
    cbar = fig.colorbar(im, cax=cax, fraction=0.046, pad=0.04, orientation="horizontal")
    cbar.ax.tick_params(labelsize=12)
    # axes[i].colorbar()
    axes[i].set_xticks([])
    axes[i].set_aspect('equal')
    axes[i].set_title(f"alpha = {np.round(alphas[i], decimals=1)}")
    # axes[i].set_title(f"T_B, params = alpha = {np.round(alphas[i], decimals=1)}, b = {np.round(b_0, decimals=1)}, k = {np.round(k_0, decimals=1)}")
# plt.savefig(battaglia_testing + f"/z_{np.round(set_z, decimals=1)}_alpha_{np.round(alpha, decimals=1)}_b_0_{np.round(b_0, decimals=1)}_k_0_{np.round(k_0, decimals=1)}_T_b.png")
plt.savefig(battaglia_testing + f"/changing_alpha.png")
plt.close()

fig, axes = plt.subplots(1, 5, figsize=(16, 8))

for i in range(len(b_0s)):
    alpha = 0.564
    k_0 = 0.185
    # for j in range(len(b_0s)):
    #     for k in range(len(k_0s)):
    # b_0 = b_mesh[i, j, k]
    # k_0 = k_mesh[i, j, k]
    batt_model_instance = Dens2bBatt(pb.delta_x(), delta_pos=1, set_z=set_z, flow=True, alpha=alpha, b_0=b_0s[i], k_0=k_0)
    im = axes[i].imshow(batt_model_instance.temp_brightness)
    divider = make_axes_locatable(axes[i])
    cax = divider.append_axes('bottom', size='5%', pad=0.05)
    cbar = fig.colorbar(im, cax=cax, fraction=0.046, pad=0.04, orientation="horizontal")
    cbar.ax.tick_params(labelsize=12)
    # axes[i].colorbar()
    axes[i].set_xticks([])
    axes[i].set_aspect('equal')
    axes[i].set_title(f"beta = {np.round(b_0s[i], decimals=1)}")
    # axes[i].set_title(f"T_B, params = alpha = {np.round(alphas[i], decimals=1)}, b = {np.round(b_0, decimals=1)}, k = {np.round(k_0, decimals=1)}")
# plt.savefig(battaglia_testing + f"/z_{np.round(set_z, decimals=1)}_alpha_{np.round(alpha, decimals=1)}_b_0_{np.round(b_0, decimals=1)}_k_0_{np.round(k_0, decimals=1)}_T_b.png")
plt.savefig(battaglia_testing + f"/changing_beta.png")
plt.close()

fig, axes = plt.subplots(1, 5, figsize=(16, 8))

for i in range(len(k_0s)):
    alpha = 0.564
    b_0 = 0.593
    # for j in range(len(b_0s)):
    #     for k in range(len(k_0s)):
    # b_0 = b_mesh[i, j, k]
    # k_0 = k_mesh[i, j, k]
    batt_model_instance = Dens2bBatt(pb.delta_x(), delta_pos=1, set_z=set_z, flow=True, alpha=alpha, b_0=b_0, k_0=k_0s[i])
    im = axes[i].imshow(batt_model_instance.temp_brightness)
    divider = make_axes_locatable(axes[i])
    cax = divider.append_axes('bottom', size='5%', pad=0.05)
    cbar = fig.colorbar(im, cax=cax, fraction=0.046, pad=0.04, orientation="horizontal")
    cbar.ax.tick_params(labelsize=12)
    # axes[i].colorbar()
    axes[i].set_xticks([])
    axes[i].set_aspect('equal')
    axes[i].set_title(f"k_0 = {np.round(k_0s[i], decimals=1)}")
    # axes[i].set_title(f"T_B, params = alpha = {np.round(alphas[i], decimals=1)}, b = {np.round(b_0, decimals=1)}, k = {np.round(k_0, decimals=1)}")
# plt.savefig(battaglia_testing + f"/z_{np.round(set_z, decimals=1)}_alpha_{np.round(alpha, decimals=1)}_b_0_{np.round(b_0, decimals=1)}_k_0_{np.round(k_0, decimals=1)}_T_b.png")
plt.savefig(battaglia_testing + f"/changing_k_0.png")
plt.close()

# b_0=0.593, alpha=0.564, k_0=0.185
