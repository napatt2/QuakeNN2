# %%
import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from deepxde.backend import tf
import pandas as pd
import shutil
import os

# %% [markdown]
# # Import Dataset

# %%
raw = pd.read_csv('../Dataset/sbm1.csv')
raw = raw[0:10000]
observe_t = raw['Var1']
u_ext = raw['y1_1']
v_ext = raw['y1_2']
theta_ext = raw['y1_3']

# %% [markdown]
# # Formulation of the Problem
# 
# The system of ODE is written as follows:
# 
# \begin{gather*}
#         \frac{d\tilde{u}}{d\tilde{t}} = \tilde{v} \\
#         \frac{d\tilde{v}}{d\tilde{t}} = \kappa(v_o \tilde{t} - \tilde{u}) - \alpha(f_o + a \log \tilde{v} + b \log \tilde{\theta}) \\
#         \frac{d\tilde{\theta}}{d\tilde{t}} = -\tilde{v}\tilde{\theta}\ln(\tilde{v}\tilde{\theta})
#     \end{gather*}
# 
# Given the measurements of $\tilde{u}, \tilde{v},$ and $\tilde{\theta}$, our goal is to predict parameter $\kappa$. We initialize parameter $\kappa$ as 0.2. True $\kappa$ is 0.25. Thus, the loss functions can be written as follows:
# 
# \begin{align*}
# MSE &= MSE_R + MSE_m \\
# MSE_R &= \frac{1}{N_R} \sum_{i=1}^{N_R} \left| \left( \frac{du(t_i, \varphi)}{dt} \right) - v(t_i, \varphi) \right|^2 \\
# &\quad + \frac{1}{N_R} \sum_{i=1}^{N_R} \left| \frac{dv(t_i, \varphi)}{dt} - \kappa(v_0 t_i - u(t_i, \varphi)) + \alpha(f_o + a \log v(t_i, \varphi) + b \log \theta(t_i, \varphi)) \right|^2 \\
# &\quad + \frac{1}{N_R} \sum_{i=1}^{N_R} \left| \frac{d\theta (t_i, \varphi)}{dt} + v(t_i, \varphi)\theta(t_i, \varphi) \ln(v(t_i, \varphi)\theta(t_i, \varphi)) \right|^2 \\
# MSE_m &= \frac{1}{N_m} \sum_{i=1}^{N_m} \left| u(t_i, \varphi) - u^*(t_i) \right|^2 + \frac{1}{N_m} \sum_{i=1}^{N_m} \left| v(t_i, \varphi) - v^*(t_i) \right|^2 + \frac{1}{N_m} \sum_{i=1}^{N_m} \left| \theta(t_i, \varphi) - \theta^*(t_i) \right|^2 \\
# \end{align*}

# %%
alpha = 9.81
v0 = 1
f0 = 0.2
a = 0.2
b = 0.3

# %%
def ode_system(x, y):
  u = y[:, 0:1]
  v = y[:, 1:2]
  theta = y[:, 2:3]

  du_t = dde.grad.jacobian(y, x, i=0)
  dv_t = dde.grad.jacobian(y, x, i=1)
  dtheta_t = dde.grad.jacobian(y, x, i=2)

  return     [
      du_t - v ,
      dv_t - C1 * (v0 * x - u) + alpha * (f0 + a * tf.math.log(v) + b * tf.math.log(theta)),
      dtheta_t + (v * theta * tf.math.log(v * theta))]


# %% [markdown]
# # Prepare Mesurements
# 
# Apply measurements of $\tilde{u}$ and $\tilde{v}$ to each component

# %%
arr = observe_t.values
observe_t = arr.reshape((-1, 1))
u_ext = u_ext.values.reshape((-1, 1))
v_ext = v_ext.values.reshape((-1, 1))

# %%
observe_y0 = dde.icbc.PointSetBC(observe_t, u_ext, component=0)
observe_y1 = dde.icbc.PointSetBC(observe_t, v_ext, component=1)

# %%
def output_transform(t, y):
    y1 = y[:, 0:1]
    y2 = y[:, 1:2]
    y3 = y[:, 2:3]

    return tf.concat(
        [y1 * tf.tanh(t) + 1, y2 * tf.tanh(t) + 0.5, y3 * tf.tanh(t) + 1 ], axis=1
    )
     

# %% [markdown]
# ## Compile and Train the PINN Model
# 
# The following steps are performed:
# 
# - **Geometry**: The time domain is set as \([0, 100]\).
# - **Data**: We enforce that the network satisfies the inverse ODE system and that its predictions match the 25 measurement points (using 20,000 residual points).
# - **Network Architecture**: A feed-forward neural network (FNN) with 6 hidden layers of 64 neurons each is used. The network takes time as input and outputs three values: $\tilde{u}, \tilde{v}$, and $\tilde{\theta}$.
# - **Output Transform**: An output transform is applied to help the network meet the initial conditions.
# - **Training**: The model is compiled with the Adam optimizer (learning rate = 0.00005) and trained for 57,000 iterations.
# 
# During training, the trainable variable \(\kappa\) is updated along with the network parameters to minimize the total loss.

# %%
geom = dde.geometry.TimeDomain(0, 100)

# %%
C1 = dde.Variable(0.2)
dde.config.set_random_seed(123)
data = dde.data.PDE(geom, ode_system, [observe_y0, observe_y1], 20000, 0, num_test = 3000, anchors=observe_t)
layer_size = [1] + [64] * 6 + [3]
activation = "tanh"
initializer = "Glorot normal"
net = dde.nn.FNN(layer_size, activation, initializer)
net.apply_output_transform(output_transform)
model = dde.Model(data, net)
external_trainable_variables = [C1]
variable = dde.callbacks.VariableValue(
    external_trainable_variables, period=500, filename="./../output/inverse_kappa/variables.dat"
)

model.compile(
    "adam", lr=0.00005, external_trainable_variables=external_trainable_variables
)

# Create output directory for saving checkpoints
path = "./../output/inverse_kappa/model"
os.makedirs(path, exist_ok=True)
checkpoint_path = os.path.join(path, "model.ckpt")
checker = dde.callbacks.ModelCheckpoint(
      checkpoint_path, save_better_only=True, period=500
  )

losshistory, train_state = model.train(iterations=57000 , callbacks=[variable, checker])

dde.saveplot(losshistory, train_state, issave=True, isplot=True, output_dir="./../output/inverse_kappa/")


# %% [markdown]
# # Prediction and Plotting
# 
# The predictions from output neurons are plotted below, while the prediction of $\kappa$ can be found in file "variables.dat"

# %%
plt.figure()
plt.xlabel("time")
plt.ylabel("y")

plt.plot(observe_t, u_ext, color="black", label="True u")
plt.plot(observe_t, v_ext, color="blue", label="True v")
plt.plot(observe_t, theta_ext, color="brown", label=r'True $\theta$')

t = np.linspace(0, 100, 10000)
t = t.reshape(10000, 1)
sol_pred = model.predict(t)
u_pred = sol_pred[:, 0:1]
v_pred = sol_pred[:, 1:2]
theta_pred = sol_pred[:, 2:3]

plt.plot(t, u_pred, color="red", linestyle="dashed", label="Predict u")
plt.plot(t, v_pred, color="orange", linestyle="dashed", label="Predict v")
plt.plot(t, theta_pred, color="green", linestyle="dashed", label=r"Predict $\theta$")
plt.legend()
plt.grid()
plt.savefig('./../output/inverse_kappa/pred.png')
plt.show()


