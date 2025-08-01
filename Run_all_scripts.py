import numpy as np
import numpy.matlib
import scipy.io
import matplotlib.pyplot as plt

from config.cfg import config_vars as cfg

from utility.utils import reduce_spatial_dim, build_koopman_state

from EDMD import edmd_algorithm as edmd


data = scipy.io.loadmat(cfg["path"])
X, Y, U = data["X"], data["Y"], data["U"]
print(X.shape, Y.shape, U.shape)

keylist = [key for key in data]
print(keylist)

N = cfg["N"]
Ntraj = cfg["Ntraj"]
trajLen = cfg["trajLen"]
nd = cfg["nd"]

A, B, C = edmd(X, Y, U, Ntraj, trajLen, nd)
print(f"A : {A.shape},  B : {B.shape},  C : {C.shape}")




import casadi as ca
import numpy as np
from solveNumericallyFDM import solve_it


# System parameters
nx = 100 # state dimension
nxk = 102  # Koopman State dimension
nu = 2    # Number of control inputs (u1 and u2)
N = 10    # Prediction horizon

# Cost matrices
Q = np.eye(nxk)  # State cost matrix (102x102)
R = np.eye(nu)  # Control input cost matrix (2x2)

# Simulation steps
Nsim = 600

# constraints
umin, umax = -0.1, 0.1
lbg, ubg = 0, 0
lbx, ubx = [], []

for i in range(nxk*(N+1)):
    lbx.append(-np.inf)
    ubx.append(np.inf)

for i in range(nu*N):
    lbx.append(umin)
    ubx.append(umax)

# Reference signal
ref = np.ones(int(Nsim))*0.5
for i in range(int(Nsim/3)):
    ref[i+int(Nsim/3)] = 1

X_ref = np.ones((nx, 1))*ref
X_ref = build_koopman_state(X_ref, 0, nd)

# Control profiles
f1 = np.exp(-(((np.linspace(0, 1, nx)) - 0.25) * 15) ** 2)  # Control profile f1
f2 = np.exp(-(((np.linspace(0, 1, nx)) - 0.75) * 15) ** 2)  # Control profile f2

# Initialize CasADi variables
xinit = ca.SX.sym('xinit', nxk, 1)
x = ca.SX.sym('x', nxk, N + 1)  # State trajectory (102 x (N+1))
u1 = ca.SX.sym('u1', 1, N)     # Control trajectory for u1 (1 x N)
u2 = ca.SX.sym('u2', 1, N)     # Control trajectory for u2 (1 x N)

# Cost function
cost = 0
constraints = []

x_ref_sym = 0
flag = 0

constraints.append(x[:, 0] - xinit)
for k in range(N):
    # Reference state at step k
    x_ref_k = ca.SX.sym(f'x_ref_{k}', nxk)
    x_ref_sym = ca.reshape(x_ref_k, -1, 1) if flag==0 else ca.horzcat(x_ref_sym, ca.reshape(x_ref_k, -1, 1))
    flag=1

    # Total control input u(x) = u1 * f1 + u2 * f2
    #u_k = u1[:, k] * f1 + u2[:, k] * f2  # Element-wise multiplication

    # Cost function: tracking error + control effort
    cost += ca.mtimes([(x[:, k] - x_ref_k).T, Q, (x[:, k] - x_ref_k)]) + ca.mtimes([ca.horzcat(u1[:, k], u2[:, k]), R, ca.vertcat(u1[:, k], u2[:, k])])

    # System dynamics constraint: x[k+1] = A * x[k] + B * u(x)
    x_next = ca.mtimes(A, x[:, k]) + B @ ca.vertcat(u1[:, k], u2[:, k])
    constraints.append(x[:, k+1] - x_next)

# print(x_ref_sym.shape)

# Flatten optimization variables and parameters
opt_variables = ca.vertcat(ca.reshape(x, -1, 1), ca.reshape(u1, -1, 1), ca.reshape(u2, -1, 1))
opt_params = ca.vertcat(xinit, ca.vec(x_ref_sym))  # Initial state and reference signal

# print(opt_params.shape)

# Define the NLP
nlp = {'x': opt_variables, 'f': cost, 'g': ca.vertcat(*constraints), 'p': opt_params}

# Solver settings
opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.max_iter': 100}
solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

# Initial conditions
ic1 = np.exp((-1)*(((np.linspace(0, 1, nx))-0.5)*5)**2)
ic2 = (np.sin(4*np.pi*(np.linspace(0, 1, nx))))**2
a = 0.2
x0 = a*ic1 + (1-a)*ic2

U = np.zeros((nu, nd))
X = np.zeros((nx, nd+1))
X[:, 0] = x0

for i in range(nd):
    X[:, i+1] = solve_it(X[:, i], U[:, i][0]*f1+U[:, i][1]*f2)

z0 = build_koopman_state(np.reshape(X[:, -1], (-1, 1)), U, nd).flatten()


u1_0 = np.zeros((1, N))  # Initial guess for u1
u2_0 = np.zeros((1, N))  # Initial guess for u2
z_init = np.zeros((nxk, N + 1))  # Initial guess for koopman state trajectory

# Initialize the optimization problem
opt_init = np.concatenate((z_init.flatten(), u1_0.flatten(), u2_0.flatten()))

# Simulation loop to track the reference signal
for t in range(Nsim):
    # Reference trajectory for the next N steps
    ref = X_ref[:, t]
    Xref_current = np.ones((N, 1))*ref 
    Xref_current_vec = Xref_current.flatten()

    # Solve the MPC problem
    # print(f"hehe {z0.shape}")
    # print(np.concatenate((z0, Xref_current_vec)).shape)
    sol = solver(x0=opt_init, p=np.concatenate((z0, Xref_current_vec)), lbg=lbg, ubg=ubg, lbx=lbx, ubx=ubx)
    sol_x = sol['x'].full()

    # Extract the optimal control inputs and state trajectory
    z_opt = sol_x[:nxk * (N + 1)].reshape((nxk, N + 1))
    u1_opt = sol_x[nxk * (N + 1):nxk * (N + 1) + N].reshape((1, N))
    u2_opt = sol_x[nxk * (N + 1) + N:].reshape((1, N))

    # Apply the first control input
    # u_applied = np.array([u1_opt[:, 0], u2_opt[:, 0]]).T
    # x0 = (A @ x0 + B @ u_applied).flatten()

    u_applied = u1_opt[:, 0]*f1 + u2_opt[:, 0]*f2
    X = np.hstack( (X, np.reshape(solve_it(X[:, -1], u_applied), (-1, 1))) )
    # print(f"opop {u1_opt[:, 0]}")
    U = np.hstack( (U, np.reshape(np.vstack( (u1_opt[:, 0], u2_opt[:, 0]) ), (-1, 1))) )
    z0 = build_koopman_state(np.reshape(X[:, -1], (-1, 1)), U, nd).flatten()
    #print(f"z he vai {z0}")

    # Update initial guess for the next iteration
    opt_init = np.concatenate((z_opt[:, 0:].flatten(), u1_opt[:, 0:].flatten(), u2_opt[:, 0:].flatten()))

    # Print the state and control at the current step
    print(f"Time step {t}: MSE: {sum((X[:, -1]-C@X_ref[:, t])**2)/nx}, Control: u1={u1_opt[:, 0]}, u2={u2_opt[:, 0]}")

    # if t==20:
    #     break


print(U.shape)
print(X.shape)
print(U[:, 0].shape)


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

x_grid = np.linspace(0, 1, nx) 

# Create a figure and axes for the plot with two subplots side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5))

# Initialize lines for the first subplot
line1_1, = ax1.plot([], [], '.', color='blue', label='reference signal')  
line1_2, = ax1.plot([], [], color='orange', linewidth=2, label='system signal')  

# Initialize lines for the second subplot
line2_1, = ax2.plot([], [], color='green', linewidth=2, label='u1') 
line2_2, = ax2.plot([], [], color='red', linewidth=2, label='u2')  

# Configure axes
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1.5)
ax1.legend()
ax1.set_title("velocity profile changing with time")

ax2.set_xlim(0, 1)
ax2.set_ylim(umin, umax)
ax2.legend()
ax2.set_title("Control input generated by KMPC")

# Update function for the animation
def update(frame):
    y1_1 = C@X_ref[:, frame+1]  # reference signal
    y1_2 = X[:, frame+1]  # system signal
    line1_1.set_data(x_grid, y1_1)
    line1_2.set_data(x_grid, y1_2)

    y2_1 = U[:, frame][0]*f1  # control 1
    y2_2 = U[:, frame][1]*f2  # control 2
    line2_1.set_data(x_grid, y2_1)
    line2_2.set_data(x_grid, y2_2)
    
    return line1_1, line1_2, line2_1, line2_2

ani = FuncAnimation(fig, update, frames=np.arange(0, U.shape[1]-1, 1), blit=True)

ani.save('tracking_with_kmpc.mp4', writer='ffmpeg', fps=8)
plt.close(fig) 


""" END """




