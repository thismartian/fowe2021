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
def cf(yearly_capacity, AEP_optimized) : return(100 * AEP_optimized/(10**6 * yearly_capacity)) 

# Instantiate the FLORIS object
file_dir = os.path.dirname(os.path.abspath(__file__))
fi = wfct.floris_interface.FlorisInterface(
    os.path.join(file_dir, "../../../example_input.json")
)

# Set turbine locations to 3 turbines in a triangle
D = fi.floris.farm.turbines[0].rotor_diameter
layout_x = [1000, 500, 1500]
layout_y = [1500, 750, 750]
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






# =====================================================
# Optimizing turbine layout...
# Number of parameters to optimize =  6
# =====================================================
#   NIT    FC           OBJFUN            GNORM
#     1     8    -1.017467E+00     1.226139E-01
#     2    15    -1.022468E+00     8.893302E-01
#     3    23    -1.011683E+00     1.538141E-01
#     4    32    -1.043374E+00     1.627747E-01
#     5    39    -1.043427E+00     1.439171E-02
#     6    46    -1.043437E+00     1.174234E-02
#     7    53    -1.043446E+00     1.202273E-02
#     8    60    -1.043455E+00     1.224092E-02
#     9    67    -1.043467E+00     1.189220E-02
#    10    74    -1.043478E+00     1.053980E-02
#    11    81    -1.043497E+00     8.766486E-03
#    12    88    -1.043520E+00     6.387072E-03
#    13    95    -1.043528E+00     4.309324E-03
#    14   102    -1.043534E+00     3.991372E-03
#    15   109    -1.043548E+00     3.892569E-03
#    16   116    -1.043560E+00     3.353964E-03
#    17   123    -1.043565E+00     2.453271E-03
#    18   130    -1.043569E+00     1.876675E-03
#    19   137    -1.043574E+00     1.399938E-03
#    20   144    -1.043576E+00     5.818043E-04
#    21   151    -1.043576E+00     3.238323E-04
#    22   158    -1.043576E+00     2.720643E-04
#    23   165    -1.043576E+00     2.439967E-04
#    24   172    -1.043577E+00     2.026745E-04
#    25   179    -1.043577E+00     1.656379E-04
#    26   186    -1.043577E+00     1.434970E-04
#    27   193    -1.043577E+00     1.363424E-04
#    28   200    -1.043577E+00     1.402541E-04
#    29   207    -1.043577E+00     1.626346E-04
#    30   214    -1.043577E+00     2.203884E-04
#    31   221    -1.043578E+00     3.270087E-04
#    32   228    -1.043580E+00     4.588733E-04
#    33   235    -1.043582E+00     5.195617E-04
#    34   242    -1.043583E+00     4.298881E-04
#    35   249    -1.043584E+00     2.756154E-04
#    36   256    -1.043584E+00     9.209469E-05
#    37   262    -1.043584E+00     7.288678E-05
# Optimization terminated successfully    (Exit mode 0)
#             Current function value: -1.0435842720985886
#             Iterations: 37
#             Function evaluations: 262
#             Gradient evaluations: 37
# Optimization complete!
# =====================================================
# Layout coordinates: 
# Turbine 0 : 	x =  1696.6 	y =  1966.7
# Turbine 1 : 	x =  0.0 	y =  803.4
# Turbine 2 : 	x =  2000.0 	y =  402.9
# =====================================================
# Total AEP Gain = 4.4%
# =====================================================

# AEP_optimized
# Out[165]: 13519274982.540213

# AEP_optimized/10**6
# Out[166]: 13519.274982540213

# n_wt = 3


# def yearly_capacity(n_wt) : return(5 * 8760 * n_wt)
# def cf(yearly_capacity, AEP_optimized) : return(100 * AEP_optimized/(10**6 * yearly_capacity))

# yearly_capacity = yearly_capacity(n_wt=)
#   File "/tmp/ipykernel_540075/4278836357.py", line 1
#     yearly_capacity = yearly_capacity(n_wt=)
#                                            ^
# SyntaxError: invalid syntax


# yearly_capacity = yearly_capacity(n_wt=n_wt)

# cf(yearly_capacity, AEP_optimized)
# Out[171]: 10.288641539223907