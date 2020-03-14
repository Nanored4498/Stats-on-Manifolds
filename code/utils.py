import torch
from time import time

def Riemannian_exponential(p, q, invG, n_points=50):
	if not p.requires_grad: p = p.clone().requires_grad_(True)
	if not q.requires_grad: q = q.clone().requires_grad_(True)
	traj_p, traj_q = [p], [q]
	dt = 1. / (n_points-1)
	for i in range(n_points-1):
		p, q = traj_p[-1], traj_q[-1]
		invGp = invG(p)
		H = 0.5 * (q * invGp(q)).sum()
		dq = -torch.autograd.grad(H, p, create_graph=True)[0]
		traj_q.append(q + dq * dt)
		q = traj_q[-1]
		traj_p.append(p + invGp(q) * dt)
	return torch.stack(traj_p), torch.stack(traj_q)

def geod_at(geod, t):
	if t <= 0.0: return geod[0]
	if t >= 1.0: return geod[-1]
	n = len(geod)
	i = int(t*(n-1))
	alpha = t*(n-1) - i
	return (1-alpha) * geod[i] + alpha * geod[i+1]