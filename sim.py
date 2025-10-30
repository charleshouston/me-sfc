# Model SIM
# Simplest Model with Government Money

# The model has 11 equations, 11 unknowns, and 3 exogenous variables
# Y, YD, T_d, T_s, H_s, H_h, G_s, C_s, C_d, N_s, N_d
# Exogenous variables: G_d, theta, W

# Whatever is demanded (services, taxes, and labour) is always supplied
# within the period:
# C_s = C_d
# G_s = G_d
# T_s = T_d
# N_s = N_d

# Diposable income:
# YD = W * N_s - T_s

# Taxes:
# T_d = theta * W * N_s, where theta < 1

# Consumption:
# C_d = c0 * YD + c1 * H_h[-1]

# Budget constraint of government:
# \Delta H_s = H_s - H_s[-1] = G_d - T_d
# H_s = H_s[-1] + G_d - T_d

# Budget contraint of households:
# \Delta H_h = H_h - H_h[-1] = YD - C_d
# H_h = H_h[-1] + YD - C_d

# National income identity
# Y = C_s + G_s
# Y = W * N_d
# N_d = Y / W

import numpy as np
from scipy.optimize import fsolve


class SIM:
    def __init__(self, c0, c1, theta, g0, W):
        # Exogenous constants
        self.c0 = c0  # Consumption from disposable income
        self.c1 = c1  # Wealth effect
        self.theta = theta  # Tax rate
        self.g0 = g0  # Government spending
        self.W = W  # Wage rate

        self.x = [np.zeros(11)]  # solution vector

    def update(self):
        """Update for one time step."""

        # Define the system of equations
        def f(x):
            Y, YD, T_d, T_s, H_s, H_h, G_s, C_s, C_d, N_s, N_d = x

            H_h_prev = self.x[-1][5]
            H_s_prev = self.x[-1][4]

            eq1 = C_s - C_d
            eq2 = G_s - self.g0
            eq3 = T_s - T_d
            eq4 = N_s - N_d
            eq5 = YD - (self.W * N_s - T_s)
            eq6 = T_d - (self.theta * self.W * N_s)
            eq7 = C_d - (self.c0 * YD + self.c1 * H_h_prev)
            eq8 = H_s - (H_s_prev + self.g0 - T_d)
            eq9 = H_h - (H_h_prev + YD - C_d)
            eq10 = Y - (C_s + G_s)
            eq11 = N_d - (Y / self.W)

            return [eq1, eq2, eq3, eq4, eq5, eq6, eq7, eq8, eq9, eq10, eq11]

        initial_guess = self.x[-1]

        Y = fsolve(f, initial_guess)
        self.x.append(Y)
