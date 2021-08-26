import math

# Define geometric constants and inlet velocity.
A = 0
B = 2
C = 6
D = 2
E = 3
a1 = 0.35
a2 = 0.75
Vin = 100


# Calculate polynomial terms used in hermite interpolation
def h(n, order):
    if order == 1:
        return 2 * n ** 3 - 3 * n ** 2 + 1
    elif order == 2:
        return -2 * n ** 3 + 3 * n ** 2
    elif order == 3:
        return n ** 3 - 2 * n ** 2 + n
    elif order == 4:
        return n ** 3 - n ** 2
    else:
        raise Exception("Illegal order for h!")


# Functions for defining the boundary curves
def calculate_y1(x_val):
    y = (D - a1) - a1 * math.cos(math.pi * (x_val - A) / (B - A)) if x_val < B else (D - a2) + a2 * math.cos(
        math.pi * (x_val - B) / (C - B))
    return y


def calculate_y2(x_val):
    y = (E + a1) + a1 * math.cos(math.pi * (x_val - A) / (B - A)) if x_val < B else (E + a2) - a2 * math.cos(
        math.pi * (x_val - B) / (C - B))
    return y


# Function for first order differencing
def first_order_diff(Y, X):
    return (Y[0] - Y[1]) / (X[0] - X[1])


# Calculate coordinates for a hermite interpolation at a given xi (e) and eta(n).
# e_inc is the incremented value used for the first order differencing
def coord_herm(e, n, e_inc, straighten=False):
    X1_val = e * C
    X1_inc = e_inc * C

    X2_val = X1_val
    X2_inc = X1_inc

    if (straighten):
        Y1_val = calculate_y1(0)
        Y1_inc = calculate_y1(C * (e_inc - e))

        Y2_val = calculate_y2(0)
        Y2_inc = calculate_y2(C * (e_inc - e))
    else:
        Y1_val = calculate_y1(X1_val)
        Y1_inc = calculate_y1(X1_inc)

        Y2_val = calculate_y2(X2_val)
        Y2_inc = calculate_y2(X2_inc)

    Kf1 = 0.2
    Kf2 = 0.2

    x_ret = X1_val * h(n, 1) + X2_val * h(n, 2) - Kf1 * first_order_diff([Y1_val, Y1_inc], [e, e_inc]) * h(n,
                                                                                                           3) - Kf1 * first_order_diff(
        [Y2_val, Y2_inc], [e, e_inc]) * h(n, 4)

    y_ret = Y1_val * h(n, 1) + Y2_val * h(n, 2) + Kf2 * first_order_diff([X1_val, X1_inc], [e, e_inc]) * h(n,
                                                                                                           3) + Kf2 * first_order_diff(
        [X2_val, X2_inc], [e, e_inc]) * h(n, 4)

    return [x_ret, y_ret]


# Debugging function to revert to lagrange (simpler than hermite, not used)
def coord_lagrange(e, n, e_inc):
    X1_val = e * C
    X2_val = X1_val

    Y1_val = calculate_y1(X1_val)
    Y2_val = calculate_y2(X2_val)

    x_val = X1_val * (1 - n) + X2_val * n
    y_val = Y1_val * (1 - n) + Y2_val * n

    return [x_val, y_val]


# Creates the phy matrix for the initial guess of phi
def initialize_phi(phi_matrix, IL, JL, xy_mesh):
    delta_n = 1 / JL

    for j in range(1, JL + 1):
        metrics = MetricsClass(xy_mesh, 0, j, IL, JL)
        phi_matrix[0][j] = (Vin / metrics.eta_y) * delta_n + phi_matrix[0][j - 1]

    for i in range(1, IL + 1):
        for j in range(0, JL + 1):
            phi_matrix[i][j] = phi_matrix[i - 1][j]

    for i in range(1, IL + 1):
        phi_matrix[i][JL] = phi_matrix[i - 1][JL]
        phi_matrix[i][0] = phi_matrix[i - 1][0]


# Define a function for second order central differencing
def central_diff(Y, X):
    return (Y[1] - Y[0]) / (2 * (X[1] - X[0]))


# Class to calculate cell metrics, given the coordinates of the cell and the number of xi, eta grid points.
class MetricsClass:
    def __init__(self, xy_mesh, i, j, IL, JL):
        delta_e = 1 / IL
        delta_n = 1 / JL

        # Use central differencing where possible, otherwise first order differencing along boundaries.
        if i + 1 > IL:
            coords = [xy_mesh[IL - 1][j], xy_mesh[IL][j]]
            X = [c[0] for c in coords]
            Y = [c[1] for c in coords]

            self.x_chi = first_order_diff(X, [0, delta_e])
            self.y_chi = first_order_diff(Y, [0, delta_e])

        elif i - 1 < 0:

            coords = [xy_mesh[0][j], xy_mesh[1][j]]
            X = [c[0] for c in coords]
            Y = [c[1] for c in coords]

            self.x_chi = first_order_diff(X, [0, delta_e])
            self.y_chi = first_order_diff(Y, [0, delta_e])

        else:

            coords = [xy_mesh[i - 1][j], xy_mesh[i + 1][j]]
            X = [c[0] for c in coords]
            Y = [c[1] for c in coords]

            self.x_chi = central_diff(X, [0, delta_e])
            self.y_chi = central_diff(Y, [0, delta_e])

        if j + 1 > JL:

            coords = [xy_mesh[i][j - 1], xy_mesh[i][j]]
            X = [c[0] for c in coords]
            Y = [c[1] for c in coords]

            self.x_eta = first_order_diff(X, [0, delta_n])
            self.y_eta = first_order_diff(Y, [0, delta_n])

        elif j - 1 < 0:

            coords = [xy_mesh[i][0], xy_mesh[i][1]]
            X = [c[0] for c in coords]
            Y = [c[1] for c in coords]

            self.x_eta = first_order_diff(X, [0, delta_n])
            self.y_eta = first_order_diff(Y, [0, delta_n])

        else:
            coords = [xy_mesh[i][j - 1], xy_mesh[i][j + 1]]
            X = [c[0] for c in coords]
            Y = [c[1] for c in coords]

            self.x_eta = central_diff(X, [0, delta_n])
            self.y_eta = central_diff(Y, [0, delta_n])

        self.J = self.x_chi * self.y_eta - self.y_chi * self.x_eta
        self.chi_x = self.y_eta / self.J
        self.eta_x = -self.y_chi / self.J
        self.chi_y = -self.x_eta / self.J
        self.eta_y = self.x_chi / self.J
        if self.J > 1e5:
            temp = 42


# Caluclate f and g functions used in the governing equation in the xi, eta domain
def fg(i, j, phi_mat, xy_mesh, IL, JL):
    metrics = MetricsClass(xy_mesh, i, j, IL, JL)
    delta_chi = 1 / IL
    delta_eta = 1 / JL

    if i + 1 > IL:
        phi = [phi_mat[IL - 1][j], phi_mat[IL][j]]
        phi_chi = first_order_diff(phi, [0, delta_chi])
    elif i - 1 < 0:
        phi = [phi_mat[0][j], phi_mat[1][j]]
        phi_chi = first_order_diff(phi, [0, delta_chi])
    else:
        phi = [phi_mat[i - 1][j], phi_mat[i + 1][j]]
        phi_chi = central_diff(phi, [0, delta_chi])

    if j + 1 > JL:
        phi = [phi_mat[i][JL - 1], phi_mat[i][JL]]
        phi_eta = first_order_diff(phi, [0, delta_eta])
    elif j - 1 < 0:
        phi = [phi_mat[i][0], phi_mat[i][1]]
        phi_eta = first_order_diff(phi, [0, delta_eta])
    else:
        phi = [phi_mat[i][j - 1], phi_mat[i][j + 1]]
        phi_eta = central_diff(phi, [0, delta_eta])

    f = metrics.chi_x * phi_chi + metrics.eta_x * phi_eta
    g = metrics.chi_y * phi_chi + metrics.eta_y * phi_eta

    f_hat = metrics.J * metrics.chi_x * f + metrics.J * metrics.chi_y * g
    g_hat = metrics.J * metrics.eta_x * f + metrics.J * metrics.eta_y * g

    # Debugging
    if abs(f_hat) > 1e6 or abs(g_hat) > 1e6:
        temp = 42

    return [f_hat, g_hat]
