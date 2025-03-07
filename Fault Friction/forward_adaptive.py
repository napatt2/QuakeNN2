# %% [markdown]
# # Adaptive Time-Stepping for Solving the Equations of a Spring-Block System
# 
# This notebook demonstrates the formulation of a system of differential equations, the training of a DeepXDE model to approximate its solution, and an adaptive time-stepping strategy that adjusts the time step based on the training loss. Detailed explanations are provided for each part of the code.

# %% [markdown]
# ## 1. Problem Formulation and ODE System Definition
# 
# The modified system of differential equations is given by:
# 
# $$\begin{gather*}
# \frac{dv}{dt} = A, \\
# \frac{dA}{dt} = \kappa(v_o - v) - \alpha\left(\frac{aA}{v} + \frac{b}{\theta}\left(1 - \frac{v\theta}{L}\right)\right), \\
# \frac{d\theta}{dt} = 1 - \frac{v\theta}{L}
# \end{gather*}$$
# 
# ### Variables and Parameters:
# 
# - **v**: Slip rate. 
# - **A**: Acceleration. 
# - **$\theta$**: State variable. 
# - **alpha = 9.81**: Constant representing the normal force. 
# - **kappa = 1**: Spring stiffness. 
# - **v0_const = 1**: Driving plate velocity. 
# - **a = 0.2, b = 0.3**: Frictional parameters in rate-and-state friction. 
# - **L = 0.25**: Characteristic length in rate-and-state friction.
# 
# The function `ode_system(x, y)` defines these equations using DeepXDE's automatic differentiation to compute the necessary time derivatives.

# %%
import deepxde as dde
import tensorflow as tf
import numpy as np
import pandas as pd
import os

# Physical parameters
alpha = 9.81
kappa = 1
v0_const = 1  # Target value for velocity
a = 0.2
b = 0.3
L = 0.25

def ode_system(x, y):
    """
    Defines the modified system of differential equations using DeepXDE's automatic differentiation.
    
    Parameters:
      - x: Independent variable (time)
      - y: Tensor with shape [N, 3] where columns represent [v, A, theta]
    
    Returns:
      A list of residuals for the equations:
        1. dv/dt - A
        2. dA/dt - kappa*(v0 - v) + alpha*(a/v * A + b/theta*(1 - v*theta/L))
        3. dtheta/dt - 1 + (v*theta)/L
    """
    
    # Split the tensor into its components
    v = y[:, 0:1]
    A = y[:, 1:2]
    theta = y[:, 2:3]

    # Compute derivatives using automatic differentiation
    dv_t = dde.grad.jacobian(y, x, i=0)
    dA_t = dde.grad.jacobian(y, x, i=1)
    dtheta_t = dde.grad.jacobian(y, x, i=2)

    # Define residuals for the system
    eq1 = dv_t - A
    eq2 = dA_t - kappa * (v0_const - v) + alpha * (a / v * A + b / theta * (1 - v * theta / L))
    eq3 = dtheta_t - 1 + v * theta / L

    return [eq1, eq2, eq3]

# %% [markdown]
# ## 2. DeepXDE Model Training Function
# 
# The function `train_deepxde` sets up and trains a neural network over a given time interval. It enforces the initial conditions through an output transformation. The neural network architecture includes:
# 
# - **Input layer:** 1 neuron (time)
# - **Hidden layers:** 4 layers with 64 neurons each (using tanh activation)
# - **Output layer:** 3 neurons (corresponding to v, A, theta)
# 
# The network is trained using the Adam optimizer (learning rate = 0.001) for 30,000 iterations. A checkpoint callback is used to save the best model based on the training loss.
# 
# ### Parameters for `train_deepxde`:
# 
# - **num_res:** Number of residual (collocation) points for the training domain.
# - **tmax:** The maximum time value for the current training interval (i.e., training on [0, tmax]).
# - **v0_ic, A0_ic, theta0_ic:** Initial conditions for v, A, and theta respectively.

# %%
def train_deepxde(num_res, tmax, v0_ic, A0_ic, theta0_ic):
    """
    Trains a DeepXDE model for a single time interval with the given initial conditions.
    
    Parameters:
        num_res  : Number of residual points in the domain.
        tmax     : Maximum time value for the current training interval.
        v0_ic    : Initial condition for v.
        A0_ic    : Initial condition for A.
        theta0_ic: Initial condition for theta.
    
    Returns:
        model      : The trained DeepXDE model.
        train_state: Training state with loss history and best training step.
    """
    def output_transform(t, y):
        """
        Output transformation to ensure the network satisfies the initial conditions at t = 0.
        """
        y1 = y[:, 0:1]
        y2 = y[:, 1:2]
        y3 = y[:, 2:3]
        return tf.concat([
            y1 * tf.tanh(t) + v0_ic,
            y2 * tf.tanh(t) + A0_ic,
            y3 * tf.tanh(t) + theta0_ic
        ], axis=1)
    
    # Define the time domain for the current interval
    geom = dde.geometry.TimeDomain(0, tmax)
    data = dde.data.PDE(geom, ode_system, [], num_res, 0, num_test=30000)

    # Neural network architecture: 1 input, 4 hidden layers of 64 neurons, 3 outputs
    layer_size = [1] + [64] * 4 + [3]
    activation = "tanh"
    initializer = "Glorot normal"
    net = dde.nn.FNN(layer_size, activation, initializer)
    net.apply_output_transform(output_transform)

    # Create and compile the model with the Adam optimizer
    model = dde.Model(data, net)
    model.compile("adam", lr=0.001)

    # Create output directory for saving checkpoints
    path = "./../output/forward_adaptive/Model/model" + str(count)
    os.makedirs(path, exist_ok=True)
    checkpoint_path = os.path.join(path, "model.ckpt")
    checker = dde.callbacks.ModelCheckpoint(
        checkpoint_path, save_better_only=True, period=50
    )

    # Train the model for 30,000 iterations
    losshistory, train_state = model.train(iterations=30000, callbacks=[checker])

    # If training did not reach 30,000 steps, retrain
    if losshistory.steps[-1] != 30000:
        model, train_state = train_deepxde(num_res, tmax, v0_ic, A0_ic, theta0_ic)
    else:
        # Save and restore the best model based on training loss
        dde.saveplot(losshistory, train_state, issave=True, isplot=True, output_dir=path)
        model.restore(checkpoint_path + "-" + str(train_state.best_step) + ".ckpt", verbose=1)
        dde.saveplot(losshistory, train_state, issave=True, isplot=True, output_dir=path)

    return model, train_state

# %% [markdown]
# ## 3. Adaptive Time Stepping with Loss Tolerance
# 
# This section implements an adaptive time stepping strategy. For each time interval, the model is trained until the training loss falls below a tolerance of \(10^{-4}\). If the loss is too high, the time step is halved and training is repeated. 
# 
# ### Key Variables:
# 
# - **tol = 0.0001:** Tolerance for the training loss.
# - **tmax_total = 0.5:** Total simulation time.
# - **num_res = 100000:** Number of residual points for each training interval.
# - **t, v, A, theta:** Variables representing the current time and state of the system. 
# - **count:** A counter used for naming model checkpoints uniquely.
# 
# For each time step:
# 1. The current state is logged.
# 2. An initial time step \(h\) is set.
# 3. The model is trained over the interval [0, h]. If the training loss is above the tolerance, \(h\) is halved and the model is retrained.
# 4. Once the loss is acceptable, predictions are made over \([0, 2h]\) and the final values are used as initial conditions for the next step.
# 5. The results are logged and saved to CSV files.

# %%
# Tolerance and simulation parameters
tol = 0.0001
tmax_total = 0.5      # Total simulation time
num_res = 100000      # Residual points for each interval

# Lists to log the time and state at each step
t_values = []
v_values = []
A_values = []
theta_values = []

# Lists to log predictions for each time interval
t_log = []
v_log = []
A_log = []
theta_log = []

# Initial conditions and time
t = 0
v, A, theta = 0.5, 1, 0.5
count = 0  # Used for model checkpoint naming

# Adaptive time stepping loop
while t <= tmax_total:
    # Log the current time and state
    t_values.append(t)
    v_values.append(v)
    A_values.append(A)
    theta_values.append(theta)
    
    # Start with an initial time step h
    h = 0.5
    loss_min = 100  # Initialize loss with a high value
    
    # Adaptive training: reduce h until the loss is below the tolerance
    while loss_min > tol:
        model, train_state = train_deepxde(num_res, h, v, A, theta)
        loss_min = train_state.best_loss_train
        h = h / 2  # Halve the time step if necessary
    
    # Generate test points for prediction over the interval [0, 2h]
    t_test = np.linspace(0, 2 * h, 1000).reshape(1000, 1)
    t_interval = np.linspace(t, t + 2 * h, 1000)

    # Predict the solution using the trained model
    pred = model.predict(t_test)
    v_pred = pred[:, 0].reshape(-1)
    A_pred = pred[:, 1].reshape(-1)
    theta_pred = pred[:, 2].reshape(-1)

    # Update the state for the next interval using the last prediction
    v = v_pred[-1]
    A = A_pred[-1]
    theta = theta_pred[-1]
    t += 2 * h
    count += 1

    # Log the predictions for this interval
    t_log.append(t_interval)
    v_log.append(v_pred)
    A_log.append(A_pred)
    theta_log.append(theta_pred)

    # Save predictions and state logs to CSV files
    dic1 = {
        't_train': np.array(t_log).reshape(-1),
        'pred_v': np.array(v_log).reshape(-1),
        'pred_A': np.array(A_log).reshape(-1),
        'pred_theta': np.array(theta_log).reshape(-1)
    }
    dic2 = {
        't_step': np.array(t_values).reshape(-1),
        'v_step': np.array(v_values).reshape(-1),
        'A_step': np.array(A_values).reshape(-1),
        'theta_step': np.array(theta_values).reshape(-1)
    }

    df1 = pd.DataFrame(dic1)
    df2 = pd.DataFrame(dic2)
    df1.to_csv("./../output/forward_adaptive/pred_" + str(t) + ".csv", index=False)
    df2.to_csv("../../output/forward_adaptive/pred_step_" + str(t) + ".csv", index=False)

# %%



