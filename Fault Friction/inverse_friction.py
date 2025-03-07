# %% [markdown]
# <a href="https://colab.research.google.com/github/napatt2/PINN-SBM/blob/main/Inverse_friction_evolution.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %%
import deepxde as dde
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pandas as pd
import os

# %% [markdown]
# # Import the Dataset
# We use only first 1,000 data points of $u$ and $v$ spanning from time 0 to 100.

# %%
raw = pd.read_csv('../Dataset/sbm_inv.csv')
raw = raw[0:1000]
observe_t = raw['t']
u_ext = raw['u']
v_ext = raw['v']
theta_ext = raw['theta']

# %% [markdown]
# # Formulation of the Problem
# The system of ODEs can be written as follows:
# \begin{gather*}
#         \frac{d\tilde{u}}{d\tilde{t}} = \tilde{v} \\
#         \frac{d\tilde{v}}{d\tilde{t}} = \kappa(v_o \tilde{t} - \tilde{u}) - \alpha(f_o + a \log \tilde{v} + b \log \tilde{\theta}) \\
#         \frac{d\tilde{\theta}}{d\tilde{t}} = 1 - \tilde{v}\tilde{\theta}
#     \end{gather*}
# Here, parameters $\kappa$ and $\alpha$ are defined as 0.5 and 1.0, respectively. However, we do not include measurements of $\theta$ in the training process as it is not available in-situ. Additionally, the form of the rate-and-state friction is simplified to a function of time, under the assumption that we have no prior knowledge of the friction formula. Thus, the loss functions can be written as follows:
# \begin{align*}
# MSE &= MSE_R + MSE_m \\
# MSE_R &= \frac{1}{N_R} \sum_{i=1}^{N_R} \left| \left( \frac{du}{dt} \right) (t_i, \varphi) - v(t_i, \varphi) \right|^2 \\
# &\quad + \frac{1}{N_R} \sum_{i=1}^{N_R} \left| \frac{dv}{dt}(t_i, \varphi) - \kappa(v_0 t_i - u(t_i, \varphi)) + \tau(t_i) \right|^2 \\
# MSE_m &= \frac{1}{N_m} \sum_{i=1}^{N_m} \left| u(t_i, \varphi) - u^*(t_i) \right|^2 + \frac{1}{N_m} \sum_{i=1}^{N_m} \left| v(t_i, \varphi) - v^*(t_i) \right|^2 \\
# \end{align*}
# 

# %%
alpha = 4
kappa = 0.5
v0 = 1
f0 = 0.2
a = 0.2
b = 0.3

# %% [markdown]
# ## Define ODE system
# Three output neurons are `u`, `v`, and `tau`, respectively. We take the derivative for each of the outputs with respect to time `x`.

# %%
def ode_system(x, y):
    u, v, tau = y[:, 0:1], y[:, 1:2], y[:, 2:3]
    du_t = dde.grad.jacobian(y, x, i=0)
    dv_t = dde.grad.jacobian(y, x, i=1)

    return [du_t - v,
            dv_t - kappa * (v0 * x - u) + tau]

# %% [markdown]
# ## Define Geometry and Measurement Loss
# We specify time domain in variable `geom`. Additionally, we assign the measurements of $u$ and $v$ as data loss via variable `observe_y0` and `observe_y1`.

# %%
geom = dde.geometry.TimeDomain(0, 100)

observe_t = observe_t.values.reshape((-1, 1))
u_ext = u_ext.values.reshape((-1, 1))
v_ext = v_ext.values.reshape((-1, 1))

observe_y0 = dde.icbc.PointSetBC(observe_t, u_ext, component=0)
observe_y1 = dde.icbc.PointSetBC(observe_t, v_ext, component=1)


# %% [markdown]
# # Compile and Train Model
# The geometry, equations, measurements, number of collocation points in the domain and on the boundary, achors for the invese problem, and number of test data points are defined in `data`.

# %%
dde.config.set_random_seed(20)

data = dde.data.PDE(
    geom,
    ode_system,
    [observe_y0, observe_y1],
    num_domain=2000,
    num_boundary=1000,
    anchors=observe_t,
    num_test=1000,
)

# %% [markdown]
# The network consists of 1 input neuron of time, 3 hidden layers of 100 neurons, and 3 output neurons of $u, v,$ and $\tau$. The model uses TanH activation function with Glorot uniform initialization. We run for 100,000 iterations using Adam optimization

# %%
net = dde.nn.FNN([1] + [100] * 3 + [3], "tanh", "Glorot uniform")

model = dde.Model(data, net)
model.compile("adam", lr=0.001)

# Create output directory for saving checkpoints
path = "./../output/inverse_friction/model"
os.makedirs(path, exist_ok=True)
checkpoint_path = os.path.join(path, "model.ckpt")
checker = dde.callbacks.ModelCheckpoint(
      checkpoint_path, save_better_only=True, period=500
  )

checker = dde.callbacks.ModelCheckpoint(
    "model/model.ckpt", save_better_only=True, period=1)
losshistory, train_state = model.train(iterations=100000, callbacks=[checker])
dde.saveplot(losshistory, train_state, issave=True, isplot=True, output_dir="./../output/inverse_friction")

# %% [markdown]
# # Prediction and Plotting
# 
# After training the PINN, we predict the solution over the time domain and compare the network predictions with the true measured data. 

# %%
t = np.linspace(0,100,2000).reshape(-1,1)
yhat = model.predict(observe_t)
uhat, vhat, tauhat = yhat[:, 0:1], yhat[:, 1:2], yhat[:, 2:3]

# %%
plt.figure()
plt.plot(observe_t, u_ext, "-", label="u exact")
plt.plot(observe_t, uhat, "--", label="u PINN")
plt.legend()
plt.grid()
plt.xlabel('Time')
plt.ylabel('u')
plt.savefig('./../output/inverse_friction/pred_u.png')

# %%
plt.figure()
plt.plot(observe_t, v_ext, "-", label="v exact")
plt.plot(observe_t, vhat, "--", label="v PINN")
plt.legend(loc = 'upper left')
plt.grid()
plt.xlabel('Time')
plt.ylabel('v')
plt.savefig('./../output/inverse_friction/pred_v.png')

# %%
tau = alpha * (f0 + a * np.log(v_ext) + b * np.log(theta_ext))

plt.figure()
plt.plot(observe_t, tau, "-", label="friction exact")
plt.plot(observe_t, tauhat, "--", label="friction PINN")
plt.legend(loc = 'upper left')
plt.grid()
plt.xlabel('Time')
plt.ylabel('Friction')
plt.savefig('./../output/inverse_friction/pred_friction.png')


