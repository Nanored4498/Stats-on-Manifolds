from time import time
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

# Data
n = 8
ys = torch.randn((n, 2)) * torch.tensor([0.44, 0.36]) + torch.tensor([1.6, 1.1])
zs = f(ys[:,0], ys[:,1])
ax.scatter(ys[:,0], ys[:,1], zs, color='g', s=32)

gamma = 500
def loss_fun(p, qs, **kargs):
	cost = 0.0
	Gp = inverse_metric(p)
	for i, q in enumerate(qs):
		end = Riemannian_exponential(p, q, inverse_metric, **kargs)[0][-1]
		cost += q.dot(Gp(q))
		cost += gamma * (torch.norm(end - ys[i])**2 + (f(end[0], end[1]) - zs[i])**2)
	return cost

n_steps = 60
p0 = ys.mean(0).requires_grad_(True)
qs0 = (ys - p0.detach()).requires_grad_(True)
optimizer = torch.optim.Adam([p0, qs0])
n_points = 25
for i in range(n_steps):
	ti = time()
	optimizer.zero_grad()
	loss = loss_fun(p0, qs0, n_points=n_points+int(0.33*i))
	loss.backward(retain_graph=True)
	print(f"[{i}] Loss: {loss.detach()} \tp: {p0.detach()} \t", end=" \t")
	optimizer.step()
	print(f"time: {time() - ti}s")

for q in qs0:
	geod = Riemannian_exponential(p0, q, inverse_metric, 60)[0].detach()
	z = f(geod[:,0], geod[:,1])
	ax.plot(geod[:,0], geod[:,1], z, color='orange', linewidth=2.4)
p0 = p0.detach()
ax.scatter(p0[0], p0[1], f(p0[0], p0[1]), color='r', s=45)
pl.show()