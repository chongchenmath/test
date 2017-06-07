# Copyright 2014-2016 The ODL development group
#
# This file is part of ODL.
#
# ODL is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ODL is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with ODL.  If not, see <http://www.gnu.org/licenses/>.

"""
ET image reconstruction using LDDMM.
"""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
import numpy as np
import matplotlib.pyplot as plt
from odl.discr import uniform_discr, Gradient, uniform_partition
from odl.phantom import shepp_logan, white_noise
from odl.tomo import RayTransform, fbp_op, Parallel2dGeometry
from odl.operator import (BroadcastOperator, power_method_opnorm)
from odl.solvers import (CallbackShow, CallbackPrintIteration, ZeroFunctional,
                         L2NormSquared, L1Norm, SeparableSum, 
                         chambolle_pock_solver)
from Recon_LDDMM import snr
standard_library.install_aliases()


# Implementation setting

# Reconstruction space
rec_space = uniform_discr([-16, -16], [16, 16], [128, 128],
                          dtype='float32', interp='linear') 

# Create the ground truth as the Shepp-Logan phantom
ground_truth = shepp_logan(rec_space, modified=True)
#ground_truth.show('ground truth')

# Give the number of directions
num_angles = 20
    
# Create the uniformly distributed directions
angle_partition = uniform_partition(0, np.pi, num_angles,
                                    nodes_on_bdry=[(True, False)])
    
# Create 2-D projection domain
# The length should be 1.5 times of that of the reconstruction space
detector_partition = uniform_partition(-24, 24, 182)
    
# Create 2-D parallel projection geometry
geometry = Parallel2dGeometry(angle_partition, detector_partition)
    
# Ray transform aka forward projection. We use ASTRA CUDA backend.
forward_op = RayTransform(rec_space, geometry, impl='astra_cuda')

# Create projection data by calling the ray transform on the phantom
proj_data = forward_op(ground_truth)
    
# Add white Gaussion noise onto the noiseless data
noise = white_noise(forward_op.range) * 0.8
    
# Add white Gaussion noise from file
#noise = ray_trafo.range.element(np.load('noise_20angles.npy'))
    
# Create the noisy projection data
noise_proj_data = proj_data + noise

# Create the noisy data from file
#noise_proj_data = ray_trafo.range.element(
#    np.load('noise_proj_data_20angles_snr_4_98.npy'))

# Compute the signal-to-noise ratio in dB
snr = snr(proj_data, noise, impl='dB')

# Output the signal-to-noise ratio
print('snr = {!r}'.format(snr))

#%%%
# --- Reconstructing by TV method --- #   

# Initialize gradient operator
grad_op = Gradient(rec_space, method='forward', pad_mode='symmetric')

# Column vector of two operators
op = BroadcastOperator(forward_op, grad_op)

# Do not use the g functional, set it to zero.
g = ZeroFunctional(op.domain)

# Set regularization parameter
lamb = 0.5

# Isotropic TV-regularization i.e. the l1-norm
l1_norm = lamb * L1Norm(grad_op.range)

# l2-squared data matching
l2_norm = L2NormSquared(forward_op.range).translated(noise_proj_data)

# --- Select solver parameters and solve using Chambolle-Pock --- #
# Estimate operator norm, add 10 percent to ensure ||K||_2^2 * sigma * tau < 1
op_norm = 1.1 * power_method_opnorm(op)

niter = 1000  # Number of iterations
tau = 1.0 / op_norm  # Step size for the primal variable
sigma = 1.0 / op_norm  # Step size for the dual variable
gamma = 0.5

# Choose a starting point
x = forward_op.domain.zero()

# Create functionals for the dual variable
# Combine functionals, order must correspond to the operator K
f = SeparableSum(l2_norm, l1_norm)

# Optionally pass callback to the solver to display intermediate results
callback = (CallbackPrintIteration() &
            CallbackShow('iterates'))

# Run the algorithm
chambolle_pock_solver(x, f, g, op, tau=tau, sigma=sigma, niter=niter,
                      gamma=gamma, callback=callback)

rec_result_save = np.asarray(x)

plt.imshow(np.rot90(rec_result_save), cmap='bone'), plt.axis('off')

#%%%
# --- Reconstructing by FBP --- #    
#
#
# Create FBP operator
FBP = fbp_op(forward_op, padding=True, filter_type='Hamming',
             frequency_scaling=0.4)
# Implement FBP method            
rec_result_FBP = FBP(noise_proj_data)
rec_result_FBP_save = np.asarray(rec_result_FBP)

plt.imshow(np.rot90(rec_result_FBP_save), cmap='bone'), plt.axis('off')
