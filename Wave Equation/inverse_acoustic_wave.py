# %%
import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import os

# %% [markdown]
# # Formulation of the Problem
# Our goal is to estimate parameter $c$ of the acoustic wave equation, given the dataset of $u$. The formulation can be written as follows:
# 
# \begin{align}
# \frac{\partial^2 u}{\partial t^2} = c^2 \left( \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} \right)
#     \end{align}
# 
# The true value of $c$ is $\sqrt{30/2.7} = 3.33$.

# %%
data = loadmat('./../Dataset/acoustic_c3.mat')
u_exact = data['u_log'].transpose(1, 0, 2)
tspan = data['t'].T

# %%
# Define spatial and temporal domain for the exact solution
x = np.linspace(0, 1, 101)
y = np.linspace(0, 1, 101)
t = np.linspace(0, 0.1, 101)
x_temp, y_temp, t_temp = np.meshgrid(x, y, t)

# %%
# Randomly select 1000 data points in the the datasat
idx = np.random.choice(x_temp.flatten()[:,None].shape[0], 1000, replace=False)
ob_x = x_temp.flatten()[:,None][idx,:]
ob_y = y_temp.flatten()[:,None][idx,:]
ob_t = t_temp.flatten()[:,None][idx,:]
ob_u = u_exact.flatten()[:,None][idx,:]

# %% [markdown]
# # Define Wave Equation

# %%
def Wave_Equation(x, y):
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    dy_yy = dde.grad.hessian(y, x, i=1, j=1)
    dy_tt = dde.grad.hessian(y, x, i=2, j=2)
    return dy_tt - C1**2 * (dy_xx + dy_yy)

def get_initial_loss(model):
    model.compile("adam", lr=0.001)
    losshistory, train_state = model.train(0)
    return losshistory.loss_train[0]

# %% [markdown]
# # Setup and Train Model
# 
# - **True Value & Parameter:**
#   - `C1true = np.sqrt(30/2.7)` sets the true value of the parameter.
#   - `C1 = dde.Variable(2.0)` initializes the parameter to be identified.
# 
# - **Domain Setup:**
#   - **Spatial Domain:** `[0, 1] × [0, 1]` defined via `dde.geometry.Rectangle`.
#   - **Time Domain:** `[0, 0.1]` defined via `dde.geometry.TimeDomain`.
#   - **Spatio-Temporal Domain:** Combined with `dde.geometry.GeometryXTime`.
# 
# - **Training Data:**
#   - `ob_xyt` is created by horizontally stacking `ob_x`, `ob_y`, and `ob_t`.
#   - `observe_u` is generated as a point-set boundary condition using `ob_xyt` and `ob_u` (component 0).
#   - PDE data is prepared using `dde.data.TimePDE` with the `Wave_Equation` and specified numbers of domain, boundary, initial, and anchor points.
# 
# - **Neural Network Setup:**
#   - **Architecture:** `layer_size = [3] + [100] * 3 + [1]` (3 inputs, three hidden layers of 100 neurons each, 1 output).
#   - **Activation & Initializer:** `"tanh"` and `"Glorot uniform"`.
#   - **Network Construction:** Built with `dde.nn.STMsFFN` (with specified `sigmas_x` and `sigmas_t`) and a feature transform.
# 
# - **Model Configuration:**
#   - The model is assembled using `dde.Model(data, net)`.
#   - A checkpoint callback saves the model every 50 iterations.
#   - A variable callback logs the value of `C1` every 100 iterations to a file.
#   - Loss weights are determined by computing `initial_losses` and setting `loss_weights = 5 / initial_losses`.
# 
# - **Compilation & Training:**
#   - Compiled with the Adam optimizer at `lr=0.001`, including `C1` as an external trainable variable, and using an inverse time decay schedule.
#   - Training is performed for 1,000,000 iterations with callbacks for checkpointing, PDE residual resampling (every 100 iterations), and variable logging.
# 
# - **Post-Training:**
#   - Training history and state are saved and plotted using `dde.saveplot` in the specified output directory.

# %%
# true value
C1true = np.sqrt(30/2.7)

# Parameters to be identified
C1 = dde.Variable(2.0)

# Spatial domain: X × Y = [0, 1] × [0, 1]
Lx_min, Lx_max = 0.0, 1.0
Ly_min, Ly_max = 0.0, 1.0
space_domain = dde.geometry.Rectangle([Lx_min, Ly_min], [Lx_max, Ly_max])

# Time domain: T = [0, 0.1]
time_domain = dde.geometry.TimeDomain(0.0, 0.1)

# Spatio-temporal domain
geomtime = dde.geometry.GeometryXTime(space_domain, time_domain)

# Get the training data: num = 7000
ob_xyt = np.hstack((ob_x, ob_y, ob_t))
observe_u = dde.icbc.PointSetBC(ob_xyt, ob_u, component=0)

# Training datasets and Loss
data = dde.data.TimePDE(
    geomtime,
    Wave_Equation,
    [observe_u],
    num_domain=700,
    num_boundary=200,
    num_initial=100,
    anchors=ob_xyt,
)

# Neural Network setup
layer_size = [3] + [100] * 3 + [1]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.nn.STMsFFN(layer_size, activation, initializer, sigmas_x=[1, 5], sigmas_t=[1, 5])
net.apply_feature_transform(lambda x: (x - 0.5) * 2 * np.sqrt(3))
path = "./../output/inverse_acoustic_wave//model"
os.makedirs(path, exist_ok=True)
checkpoint_path = os.path.join(path, "model.ckpt")
checker = dde.callbacks.ModelCheckpoint(
      checkpoint_path, save_better_only=True, period=50
  )
model = dde.Model(data, net)

fnamevar = "./../output/inverse_acoustic_wave/variables.dat"
variable = dde.callbacks.VariableValue([C1], period=100, filename=fnamevar)

initial_losses = get_initial_loss(model)
loss_weights = 5 / initial_losses
model.compile(
    "adam",
    lr=0.001,
    external_trainable_variables=[C1],
    loss_weights=loss_weights,
    decay=("inverse time", 2000, 0.9),
)

pde_residual_resampler = dde.callbacks.PDEPointResampler(period=100) # Use pde residual sampler every 100 iterations
losshistory, train_state = model.train(
    iterations=1000000,  callbacks=[checker, pde_residual_resampler, variable], display_every=500
)

dde.saveplot(losshistory, train_state, issave=True, isplot=True, output_dir="./../output/inverse_acoustic_wave/")


# %% [markdown]
# # Predict Solution
# The prediction of variable $c$ can be founded in the file "variables.dat" which is genereated during the training.

# %%
x = np.linspace(0, 1, 101)
y = np.linspace(0, 1, 101)
t = np.linspace(0, 0.1, 101)
xv, yv, tv = np.meshgrid(x, y, t)

x_test = xv.flatten()[:,None]
y_test = yv.flatten()[:,None]
t_test = tv.flatten()[:,None]

ob_xyt = np.hstack((x_test, y_test, t_test))
pred = model.predict(ob_xyt)
pred2 = pred.reshape((101,101,101))


