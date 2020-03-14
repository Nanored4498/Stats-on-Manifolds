import torch
import pylab as pl
from mpl_toolkits.mplot3d import Axes3D
from utils import *

def f(x, y):
	return 0.35*(x-1.5)**2 + 0.2*x - 0.4*(y-1)**2 + 0.4*y + 0.3*torch.cos(3.14*x+1) + 0.1*torch.sin(6.28*y)

def inverse_metric(p):
	invG = torch.empty((2, 2))
	dx_f, dy_f = torch.autograd.grad(f(p[0], p[1]), p, create_graph=True)[0]
	dy_f2 = dy_f**2
	invG[0, 0] = 1. + dy_f2
	invG[0, 1] = invG[1, 0] = - dx_f * dy_f
	invG[1, 1] = 1. + dx_f**2
	det = invG[1, 1] + dy_f2
	invG /= det
	return lambda q: invG @ q

fig = pl.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot manifold
x = torch.linspace(0, 3, 160)
X, Y = torch.meshgrid(x, x)
Z = f(X, Y)
ax.plot_wireframe(X, Y, Z, linewidth=0.5)

# True geodesic
n = 10
p = torch.tensor([0.4, 2.6])
q = torch.tensor([1.3, -3.4])
ps, _ = Riemannian_exponential(p, q, inverse_metric, 10*n)
ps = ps.detach()
# z = f(ps[:,0], ps[:,1])
# ax.plot(ps[:,0], ps[:,1], z, color='r', linewidth=3)

# Noised observed points
k = torch.randint(5, 16, (n,)).cumsum(0)
k = ((k-5) * (10*n-9)) / (k[-1]-5)
times = k.float() / (10*n-1)
points = ps[k]
noise_space = 0.1*torch.normal(torch.zeros(n))
noise_time = 0.02*torch.normal(torch.zeros(n))
points[:,0] += noise_space + noise_time
points[:,1] += noise_space - noise_time
z = f(points[:,0], points[:,1])
ax.scatter(points[:,0], points[:,1], z, color='green', s=40)

def loss_fun(p, q, **kargs):
	geod, _ = Riemannian_exponential(p, q, inverse_metric, **kargs)
	cost = 0
	for t, pt in zip(times, points):
		gt = geod_at(geod, t)
		cost += torch.norm(pt - gt)**2 + (f(pt[0], pt[1]) - f(gt[0], gt[1]))**2
	return cost

n_steps = 40
p0 = (p + 0.005*torch.normal(torch.zeros(2))).requires_grad_(True)
q0 = (q + 0.005*torch.normal(torch.zeros(2))).requires_grad_(True)
optimizer = torch.optim.Adam([p0, q0])
for i in range(n_steps):
	ti = time()
	optimizer.zero_grad()
	loss = loss_fun(p0, q0)
	loss.backward(retain_graph=True)
	print(f"[{i}] Loss: {loss.detach()} \tp: {p0.detach()} \tq: {q0.detach()}t", end=" \t")
	optimizer.step()
	print(f"time: {time() - ti}")
	
ps = Riemannian_exponential(p0, q0, inverse_metric, 100)[0].detach()
z = f(ps[:,0], ps[:,1])
ax.plot(ps[:,0], ps[:,1], z, color='red', linewidth=3)
for t, pt in zip(times, points):
	gt = geod_at(ps, t)
	ax.plot([pt[0], gt[0]], [pt[1], gt[1]], [f(pt[0], pt[1]), f(gt[0], gt[1])], color='orange', linewidth=2)

pl.show()