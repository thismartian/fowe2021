# Copyright 2021 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

# See https://floris.readthedocs.io for documentation


import os

import numpy as np
import matplotlib.pyplot as plt

import floris.tools as wfct
import floris.tools.visualization as vis
from floris.tools.optimization.scipy.layout import LayoutOptimization


def yearly_capacity(n_wt) : return(5 * 8760 * n_wt)
def calc_cf(n_wt, AEP_optimized) : return(100 * AEP_optimized/(10**6 * yearly_capacity(n_wt))) 

# Instantiate the FLORIS object
file_dir = os.path.dirname(os.path.abspath(__file__))
fi = wfct.floris_interface.FlorisInterface(
    os.path.join(file_dir, "../../../example_input.json")
)

# Set turbine locations to 3 turbines in a triangle
D = fi.floris.farm.turbines[0].rotor_diameter
layout_x = [1000, 1000, 500, 1500]
layout_y = [0, 1500, 750, 750]
n_wt = len(layout_x)

fi.reinitialize_flow_field(layout_array=(layout_x, layout_y))

# Define the boundary for the wind farm
boundaries = [[2000.0, 2000.0], [2000.0, 0.0], [0.0, 0.0], [0.0, 2000.0]]

# Generate random wind rose data
# wd = np.arange(0.0, 360.0, 5.0)
# np.random.seed(1)
# ws = 8.0 + np.random.randn(len(wd)) * 0.5
# freq = np.abs(np.sort(np.random.randn(len(wd))))
# freq = freq / freq.sum()


import pandas as pd
df = pd.DataFrame(columns=["wd", "ws", "freq_val"])
wd = np.arange(0.0, 360.0, 45.0/2)
wd = np.repeat(wd, 8)
ws = np.arange(2.5, 42.5, 5) * 5/18
ws = np.repeat(ws, 16)
df["wd"] = wd
df["ws"] = ws
freq_val = [3,	2,	2,	0,	0,	0,	0,	0,
        2,	5,	6,	1,	3,	7,	1,	0,
        4,	12,	5,	8,	1,	1,	0,	1,
        1,	8,	5,	1,	2,	0,	0,	0,
        5,	8,	0,	1,	2,	1,	0,	0,
        5,	5,	1,	10,	3,	2,	0,	0,
        5,	6,	7,	1,	2,	0,	0,	0,
        3,	6,	1,	0,	0,	0,	0,	0,
        2,	2,	0,	0,	0,	0,	0,	0,
        2,	0,	0,	0,	0,	0,	0,	0,
        2,	2,	0,	0,	0,	0,	0,	0,
        2,	1,	0,	0,	0,	0,	0,	0,
        6,	3,	0,	0,	0,	0,	0,	0,
        5,	2,	0,	0,	0,	0,	0,	0,
        3,	3,	2,	0,	0,	0,	0,	0,
        0,	0,	0,	0,	0,	0,	0,	0]
df["freq_val"] = freq_val

freq = df["freq_val"] / df["freq_val"].sum()
print(freq)
# Set optimization options
opt_options = {"maxiter": 50, "disp": True, "iprint": 2, "ftol": 1e-8}

# Compute initial AEP for optimization normalization
AEP_initial = fi.get_farm_AEP(wd, ws, freq)

# Instantiate the layout otpimization object
layout_opt = LayoutOptimization(
    fi=fi,
    boundaries=boundaries,
    wd=wd,
    ws=ws,
    freq=freq,
    AEP_initial=AEP_initial,
    opt_options=opt_options,
)

# Perform layout optimization
layout_results = layout_opt.optimize()

print("=====================================================")
print("Layout coordinates: ")
for i in range(len(layout_results[0])):
    print(
        "Turbine",
        i,
        ": \tx = ",
        "{:.1f}".format(layout_results[0][i]),
        "\ty = ",
        "{:.1f}".format(layout_results[1][i]),
    )

# Calculate new AEP results
fi.reinitialize_flow_field(layout_array=(layout_results[0], layout_results[1]))
AEP_optimized = fi.get_farm_AEP(wd, ws, freq)

print("=====================================================")
print("Total AEP Gain = %.1f%%" % (100.0 * (AEP_optimized - AEP_initial) / AEP_initial))
print("=====================================================")

# Plot the new layout vs the old layout
layout_opt.plot_layout_opt_results()
plt.show()


print("The initial capacity factor for this layout is:")
print(calc_cf(n_wt, AEP_initial))


print("The optimized capacity factor for this layout is:")
print(calc_cf(n_wt, AEP_optimized))
