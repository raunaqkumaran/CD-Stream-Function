from HW5_Helper import *
import numpy
import matplotlib.pyplot as plt
import copy
import sys

# Define grid sizing, use arguments from terminal if they exist, otherwise default
if len(sys.argv) < 3:
    IL = 101
    JL = 101
else:
    IL = int(sys.argv[1])
    JL = int(sys.argv[2])

# Set time step size and size for figure
delta_t = 0.0001
figure_size = (15, 10)

# Define iterator controls
epsilon = 1000
epsilon_criteria = 1e-4
iter = 0
max_iter = 1000

# Store coordinates for displaying grids
orig_x = []
orig_y = []

transformed_e = []
transformed_n = []

# create 2D matrix indexed in xi, eta, but outputs the cartesian coordinates
xy_mesh = []
[xy_mesh.append((JL + 1) * [1]) for i in range(0, IL + 1)]

# Build grids
for i in range(0, IL + 1):
    # e is xi here
    e = i / IL
    e_inc = (i + 0.01) / IL if i < IL else (i - 1) / IL
    for j in range(0, JL + 1):
        n = j / JL
        transformed_n.append(n)
        transformed_e.append(e)
        if i == 1:
            orig = coord_herm(e, n, e_inc, True)
        else:
            orig = coord_herm(e, n, e_inc)
        orig_x.append(orig[0])
        orig_y.append(orig[1])
        xy_mesh[i][j] = orig

# Display grids
scatter_size = 3
plt.figure("Generalized grid", figsize=figure_size)
plt.title("(xi, eta) grid")
plt.xlabel("xi")
plt.ylabel("eta")
plt.scatter(transformed_e, transformed_n, s=scatter_size)
plt.savefig("Generalized grid with IL = " + str(IL) + " JL = " + str(JL) + ".png")

plt.figure("Original grid", figsize=figure_size)
plt.title("(x, y) grid")
plt.ylabel("y")
plt.xlabel("x")
plt.scatter(orig_x, orig_y, s=scatter_size)
plt.savefig("Original grid with IL = " + str(IL) + " JL = " + str(JL) + ".png")

phi_mat = numpy.zeros([IL + 1, JL + 1])

# Initialize a guess for phi across the entire domain
initialize_phi(phi_mat, IL, JL, xy_mesh)

# Copy the initial condition
original_phi_mat = copy.deepcopy(phi_mat)

# Plot the initial phi guess
fig0 = "initial phi solution with grid size IL = " + str(IL) + " JL = " + str(JL)

plt0 = plt.figure(fig0, figsize=figure_size)
plt.pcolormesh(numpy.transpose(phi_mat))
plt.colorbar()
plt.title("Stream function initializer")
plt.xlabel("chi")
plt.ylabel("eta")
plt.savefig(fig0 + ".png")

epsilon_normal_arr = []
normalizer = 1          # We want to normalize the residual, this later gets updated to normalize future iterations
                        # based on the average residual of the first 3 iterations
prev_epsilon = 0        # Store the residual of the previous iteration
switch_flag = 0         # Debugging
counter = 0             # Debugging
residual_list = []      # Store residuals so we can plot them

# Keep iterating unless the solution violently diverges, or we hit max iterations
while epsilon_criteria < epsilon < 1e10 and iter < max_iter:
    prev_epsilon = epsilon
    epsilon = 0                             # Assume residual is zero
    new_phi_mat = copy.deepcopy(phi_mat)    # Copy the previous guess into a new matrix for the new time step
    delta_chi = 1 / IL                      # These probably should've been outside the loop...
    delta_eta = 1 / JL

    # Iterate through i and j indices. We do not touch phi at the inlet, or along the walls,
    # since those cells impose our boundary conditions.
    for i in range(1, IL + 1):
        for j in range(1, JL):
            if i == 26 and j == 28:     # Debugging
                temp = 42
            # Calculate cell metrics
            metrics = MetricsClass(xy_mesh, i, j, IL, JL)
            arglist = [phi_mat, xy_mesh, IL, JL]    # Easier than having super long function calls
            g_diff_arr = [fg(i, j - 1, *arglist)[1], fg(i, j + 1, *arglist)[1]]     # Calculate g_hat
            g_diff = central_diff(g_diff_arr, [0, delta_eta])                       # partial g_hat
            if i < IL:  # central difference f if possible
                f_diff_arr = [fg(i - 1, j, *arglist)[0], fg(i + 1, j, *arglist)[0]]
                f_diff = central_diff(f_diff_arr, [0, delta_chi])
            else:       # if at the outlet, f must be first order differenced
                f_diff_arr = [fg(i - 1, j, *arglist)[0], fg(i, j, *arglist)[0]]
                f_diff = first_order_diff(f_diff_arr, [0, delta_chi])

            # Debugging
            if switch_flag % 2 == 0:
                new_phi_mat[i][j] = phi_mat[i][j] + (delta_t / metrics.J) * (f_diff + g_diff)
            else:
                new_phi_mat[i][j] = phi_mat[i][j] - (delta_t / metrics.J) * (f_diff + g_diff)

            # Calculate how much we changed the phi value at the cell from one time step to the next
            new_val = new_phi_mat[i][j]
            old_val = phi_mat[i][j]
            epsilon_local = abs(new_val - old_val) / normalizer

            # If this epsilon is greater than every other epsilon calculated for this time step, make it the new epsilon
            if epsilon_local > epsilon:
                epsilon = epsilon_local

            # Debugging
            if epsilon_local > 1e4:
                temp = 40
    iter = iter + 1     # Update iteration count
    residual_list.append(epsilon)
    print(f"Completed iteration {iter:d} with epsilon (absolute) {epsilon:.3e} ({epsilon * normalizer:.2e})")
    phi_mat = new_phi_mat
    if iter > 10 and epsilon > prev_epsilon:
        counter = counter + 1
        # Debugging. Kill the loop if two successive iterations lead to an increase in residual.
        if counter > 2:
            break
            switch_flag = switch_flag + 1
            counter = 0

    # Normalize all residuals based on the average of the first two absolute residuals.
    if len(epsilon_normal_arr) < 2:
        epsilon_normal_arr.append(epsilon)
    else:
        normalizer = sum(epsilon_normal_arr) / len(epsilon_normal_arr)

# Empty 2D grids to store values of U and V
U_grid = numpy.zeros([IL + 1, JL + 1])
V_grid = numpy.zeros([IL + 1, JL + 1])
diff_grid = numpy.zeros([IL + 1, JL + 1])
pressure_grid = numpy.zeros([IL + 1, JL + 1])
p_stag = 101325

# Iterate through all grid elements in the xi, eta grid. Calculate U and V by using second order (where possible)
# difference operators for phi and local cell metrics. First order differencing at boundaries.
for i in range(0, IL + 1):
    for j in range(0, JL + 1):
        diff_grid[i][j] = phi_mat[i][j] - original_phi_mat[i][j]
        metrics = MetricsClass(xy_mesh, i, j, IL, JL)
        if i == 0:
            Y = [phi_mat[0][j], phi_mat[1][j]]
            X = [0, delta_chi]
            phi_chi = first_order_diff(Y, X)
        elif i == IL:
            Y = [phi_mat[IL - 1][j], phi_mat[IL][j]]
            X = [0, delta_chi]
            phi_chi = first_order_diff(Y, X)
        else:
            Y = [phi_mat[i - 1][j], phi_mat[i + 1][j]]
            X = [0, delta_chi]
            phi_chi = central_diff(Y, X)

        if j == 0:
            Y = [phi_mat[i][0], phi_mat[i][1]]
            X = [0, delta_eta]
            phi_eta = first_order_diff(Y, X)
        elif j == JL:
            Y = [phi_mat[i][JL - 1], phi_mat[i][JL]]
            X = [0, delta_eta]
            phi_eta = first_order_diff(Y, X)
        else:
            Y = [phi_mat[i][j - 1], phi_mat[i][j + 1]]
            X = [0, delta_eta]
            phi_eta = central_diff(Y, X)

        U_grid[i][j] = metrics.chi_y * phi_chi + metrics.eta_y * phi_eta
        V_grid[i][j] = - (metrics.chi_x * phi_chi + metrics.eta_x * phi_eta)
        pressure_grid[i][j] = p_stag - 0.5 * 1.225 * U_grid[i][j] ** 2 + V_grid[i][j] ** 2

# Plot everything!!!!
fig1 = "Stream function, transformed grid with grid size IL = " + str(IL) + " JL = " + str(JL)
fig2 = "U (ms-1) velocity, transformed grid with grid size IL = " + str(IL) + " JL = " + str(JL)
fig3 = "V (ms-1) velocity, transformed grid with grid size IL = " + str(IL) + " JL = " + str(JL)
fig4 = "Residuals for IL = " + str(IL) + " JL = " + str(JL)
fig5 = "U (ms-1) velocity, xy grid with grid size IL = " + str(IL) + " JL = " + str(JL)
fig6 = "V (ms-1) velocity, xy grid with grid size IL = " + str(IL) + " JL = " + str(JL)
fig7 = "Final stream function grid - Original stream function grid for grid size IL = " + str(IL) + " JL = " + str(JL)
fig8 = "Static pressure on transformed grid with grid size IL = " + str(IL) + " JL = " + str(JL)
fig9 = "Static pressure on xy grid with grid size grid size IL = " + str(IL) + " JL = " + str(JL)

plt1 = plt.figure(fig1, figsize=figure_size)
plt.pcolormesh(numpy.transpose(phi_mat))
plt.colorbar()
plt.title("Stream function")
plt.xlabel("chi")
plt.ylabel("eta")
plt.savefig(fig1 + ".png")

plt7 = plt.figure(fig7, figsize=figure_size)
plt.pcolormesh(numpy.transpose(diff_grid))
plt.colorbar()
plt.title("Solved stream function - original stream function")
plt.xlabel("chi")
plt.ylabel("eta")
plt.savefig(fig7 + ".png")


plt2 = plt.figure(fig2, figsize=figure_size)
plt.pcolormesh(numpy.transpose(U_grid))
plt.colorbar()
plt.title("U velocity (m/s)")
plt.xlabel("chi")
plt.ylabel("eta")
plt.savefig(fig2 + ".png")

plt3 = plt.figure(fig3, figsize=figure_size)
plt.pcolormesh(numpy.transpose(V_grid))
plt.colorbar()
plt.title("V velocity (m/s)")
plt.xlabel("chi")
plt.ylabel("eta")
plt.savefig(fig3 + ".png")

plt8 = plt.figure(fig8, figsize=figure_size)
plt.pcolormesh(numpy.transpose(pressure_grid))
plt.colorbar()
plt.title("Static pressure (Pa)")
plt.xlabel("chi")
plt.ylabel("eta")
plt.savefig(fig8 + ".png")

plt4 = plt.figure(fig4, figsize=figure_size)
plt.title(fig4)
x_axis = [i for i in range(3, iter)]
plt.plot(x_axis, residual_list[3:iter + 1])
plt.xlabel("Iteration")
plt.ylabel("Residual")
plt.savefig(fig4 + " linear.png")
plt.yscale('log')
plt.savefig(fig4 + " log.png")

plt5 = plt.figure(fig5, figsize=figure_size)
plt.title(fig5)

max_vel = numpy.amax(U_grid)
min_vel = numpy.amin(U_grid)
for i in range(0, IL + 1):
    for j in range(0, JL + 1):
        x = xy_mesh[i][j][0]
        y = xy_mesh[i][j][1]
        plt.scatter(x, y, c=U_grid[i][j], cmap='jet', vmin=min_vel, vmax=max_vel)
plt.colorbar()
plt.savefig(fig5 + ".png")

plt6 = plt.figure(fig6, figsize=figure_size)
plt.title(fig6)

max_vel = numpy.amax(V_grid)
min_vel = numpy.amin(V_grid)
for i in range(0, IL + 1):
    for j in range(0, JL + 1):
        x = xy_mesh[i][j][0]
        y = xy_mesh[i][j][1]
        plt.scatter(x, y, c=V_grid[i][j], cmap='jet', vmin=min_vel, vmax=max_vel)
plt.colorbar()
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.savefig(fig6 + ".png")


plt9 = plt.figure(fig9, figsize=figure_size)
plt.title(fig9)
max_press = numpy.amax(pressure_grid)
min_press = numpy.amin(pressure_grid)
for i in range(0, IL + 1):
    for j in range(0, JL + 1):
        x = xy_mesh[i][j][0]
        y = xy_mesh[i][j][1]
        plt.scatter(x, y, c=pressure_grid[i][j], cmap='jet', vmin=min_press, vmax=max_press)
plt.colorbar()
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.savefig(fig9 + ".png")

plt.show()
