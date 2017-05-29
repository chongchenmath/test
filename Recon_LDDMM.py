import odl
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import compare_ssim, compare_psnr


def snr(signal, noise, impl):
    """Compute the signal-to-noise ratio.

    Parameters
    ----------
    signal : `array-like`
        Noiseless data.
    noise : `array-like`
        Noise.
    impl : {'general', 'dB'}
        Implementation method.
        'general' means SNR = variance(signal) / variance(noise),
        'dB' means SNR = 10 * log10 (variance(signal) / variance(noise)).

    Returns
    -------
    snr : `float`
        Value of signal-to-noise ratio.
        If the power of noise is zero, then the return is 'inf',
        otherwise, the computed value.
    """
    if np.abs(np.asarray(noise)).sum() != 0:
        ave1 = np.sum(signal) / signal.size
        ave2 = np.sum(noise) / noise.size
        s_power = np.sqrt(np.sum((signal - ave1) * (signal - ave1)))
        n_power = np.sqrt(np.sum((noise - ave2) * (noise - ave2)))
        if impl == 'general':
            return s_power / n_power
        elif impl == 'dB':
            return 10.0 * np.log10(s_power / n_power)
        else:
            raise ValueError('unknown `impl` {}'.format(impl))
    else:
        return float('inf')


# --- Give input images --- #

I0name = './pictures/v.png' # 64 * 64 ---> 92
I1name = './pictures/j.png' # 64 * 64

# --- Get digital images --- #

I0 = np.rot90(plt.imread(I0name).astype('float'), -1)[:, :]
I1 = np.rot90(plt.imread(I1name).astype('float'), -1)[:, :]

# Discrete reconstruction space: discretized functions on the rectangle
rec_space = odl.discr.uniform_discr(
    min_pt=[-16, -16], max_pt=[16, 16], shape=[64, 64],
    dtype='float32', interp='linear')

# Create the ground truth as the given image
ground_truth = rec_space.element(I0)

# Create the template as the given image
template = rec_space.element(I1)

# Implementation method for mass preserving or not,
# impl chooses 'mp' or 'geom', 'mp' means mass-preserving deformation method,
# 'geom' means geometric deformation method
impl1 = 'geom'

# Normalize the template's density as the same as the ground truth if consider
# mass preserving method
if impl1 == 'mp':
#    template *= np.sum(ground_truth) / np.sum(template)
    template *= np.linalg.norm(ground_truth, 'fro')/ \
        np.linalg.norm(template, 'fro')

# Implementation method for least square data matching term
impl2 = 'least_square'

# Show intermiddle results
callback = odl.solvers.CallbackShow(
    '{!r} {!r} iterates'.format(impl1, impl2), display_step=5) & \
    odl.solvers.CallbackPrintIteration()

#ground_truth.show('ground truth')
#template.show('template')

# The parameter for kernel function
sigma = 6.0

# Give kernel function
def kernel(x):
    scaled = [xi ** 2 / (2 * sigma ** 2) for xi in x]
    return np.exp(-sum(scaled))

# Maximum iteration number
niter = 200

# Give step size for solver
eps = 0.02

# Give regularization parameter
lamb = 10E-7

# Give the number of directions
num_angles = 10

# Create the uniformly distributed directions
angle_partition = odl.discr.uniform_partition(0.0, np.pi, num_angles,
                                    nodes_on_bdry=[(True, False)])

# Create 2-D projection domain
# The length should be 1.5 times of that of the reconstruction space
detector_partition = odl.discr.uniform_partition(-24, 24, 92)

# Create 2-D parallel projection geometry
geometry = odl.tomo.Parallel2dGeometry(angle_partition, detector_partition)

# Ray transform aka forward projection. We use ASTRA CUDA backend.
forward_op = odl.tomo.RayTransform(rec_space, geometry, impl='astra_cpu')

# Create projection data by calling the op on the phantom
proj_data = forward_op(ground_truth)

# Add white Gaussion noise onto the noiseless data
noise = 1.0 * odl.phantom.white_noise(forward_op.range)

# Create the noisy projection data
noise_proj_data = proj_data + noise

# Compute the signal-to-noise ratio in dB
snr = snr(proj_data, noise_proj_data - proj_data, impl='dB')

# Output the signal-to-noise ratio
print('snr = {!r}'.format(snr))

# Give the number of time points
time_itvs = 20

# Compute by LDDMM solver
image_N0, E = odl.deform.LDDMM_gradient_descent_solver(
        forward_op, noise_proj_data, template, time_itvs, niter, eps, lamb,
        kernel, impl1, impl2, callback)
    
rec_result_1 = rec_space.element(image_N0[time_itvs // 4])
rec_result_2 = rec_space.element(image_N0[time_itvs // 4 * 2])
rec_result_3 = rec_space.element(image_N0[time_itvs // 4 * 3])
rec_result = rec_space.element(image_N0[time_itvs])

# Compute the projections of the reconstructed image
rec_proj_data = forward_op(rec_result)

# Compute the structural similarity index between the result and ground truth
ssim = compare_ssim(np.asarray(rec_result), np.asanyarray(ground_truth))
print('SSIM = {!r}'.format(ssim))

# Compute peak signal-to-noise ratio index between the result and ground truth
psnr = compare_psnr(np.asanyarray(ground_truth), np.asarray(rec_result))
print('PSNR = {!r}'.format(psnr))

# Plot the results of interest
plt.figure(1, figsize=(24, 24))
#plt.clf()

plt.subplot(3, 3, 1)
plt.imshow(np.rot90(template), cmap='bone',
           vmin=np.asarray(template).min(),
           vmax=np.asarray(template).max())
plt.axis('off')
plt.colorbar()
plt.title('Template')

plt.subplot(3, 3, 2)
plt.imshow(np.rot90(rec_result_1), cmap='bone',
           vmin=np.asarray(rec_result_1).min(),
           vmax=np.asarray(rec_result_1).max()) 
plt.axis('off')
plt.colorbar()
plt.title('time_pts = {!r}'.format(time_itvs // 4))

plt.subplot(3, 3, 3)
plt.imshow(np.rot90(rec_result_2), cmap='bone',
           vmin=np.asarray(rec_result_2).min(),
           vmax=np.asarray(rec_result_2).max()) 
plt.axis('off')
plt.colorbar()
plt.title('time_pts = {!r}'.format(time_itvs // 4 * 2))

plt.subplot(3, 3, 4)
plt.imshow(np.rot90(rec_result_3), cmap='bone',
           vmin=np.asarray(rec_result_3).min(),
           vmax=np.asarray(rec_result_3).max()) 
plt.axis('off')
plt.colorbar()
plt.title('time_pts = {!r}'.format(time_itvs // 4 * 3))

plt.subplot(3, 3, 5)
plt.imshow(np.rot90(rec_result), cmap='bone',
           vmin=np.asarray(rec_result).min(),
           vmax=np.asarray(rec_result).max()) 
plt.axis('off')
plt.colorbar()
plt.title('Reconstructed by {!r} iters, '
    '{!r} projs'.format(niter, num_angles))

plt.subplot(3, 3, 6)
plt.imshow(np.rot90(ground_truth), cmap='bone',
           vmin=np.asarray(ground_truth).min(),
           vmax=np.asarray(ground_truth).max())
plt.axis('off')
plt.colorbar()
plt.title('Ground truth')

plt.subplot(3, 3, 7)
plt.plot(np.asarray(proj_data)[0], 'b', linewidth=1.0)
plt.plot(np.asarray(noise_proj_data)[0], 'r', linewidth=0.5)
plt.axis([0, detector_partition.size - 1, -5, 28]), plt.grid(True, linestyle='--')

plt.subplot(3, 3, 8)
plt.plot(np.asarray(proj_data)[2], 'b', linewidth=1.0)
plt.plot(np.asarray(noise_proj_data)[2], 'r', linewidth=0.5)
plt.axis([0, detector_partition.size - 1, -5, 28]), plt.grid(True, linestyle='--')

plt.subplot(3, 3, 9)
plt.plot(np.asarray(proj_data)[4], 'b', linewidth=1.0)
plt.plot(np.asarray(noise_proj_data)[4], 'r', linewidth=0.5)
plt.axis([0, detector_partition.size - 1, -5, 28]), plt.grid(True, linestyle='--')

plt.figure(2, figsize=(8, 1.5))
plt.plot(E)
plt.ylabel('Energy')
plt.gca().axes.yaxis.set_ticklabels([])
plt.grid(True, linestyle='--')
