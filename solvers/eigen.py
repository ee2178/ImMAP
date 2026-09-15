import torch
import torch.nn.functional as F
import numpy as np

def power_method(A, b, num_iter=1000, tol=1e-6, verbose=True):
	"""
	Power method for pytorch operator A and initial vector b.
	"""
	# On `b`'s device, NOT the CPU default: the convergence test compares this
	# against `eig_max`, which lives wherever `A(b)` put it. Every caller that
	# power-iterates at CONSTRUCTION (spectral_normalize, spectral_init) runs
	# before `.to(device)` and so never noticed, but anything measuring a net
	# that is already on the GPU -- `op_norm2`, `cascade_norm` -- hits a device
	# mismatch on the first iteration.
	eig_old = torch.zeros(1, device=b.device)
	flag_tol_reached = False
	for it in range(num_iter):
		b = A(b)
		b = b / torch.norm(b)
		eig_max = torch.sum(b.conj()*A(b))
		if verbose:
			print('i:{0:3d} \t |e_new - e_old|:{1:2.2e}'.format(it,abs(eig_max-eig_old).item()))
		if abs(eig_max-eig_old)<tol:
			flag_tol_reached = True
			break
		eig_old = eig_max
	if verbose:
		print('tolerance reached!',it)
		print(f"L = {eig_max.item():.3e}")
	return eig_max.item(), b, flag_tol_reached

