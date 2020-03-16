from time import time
from utils import *

X0 = [(-0.842741935483871, -0.6417748917748918), (-0.846774193548387, -0.3441558441558441), (-0.7701612903225806, 0.07792207792207795), (-0.6653225806451613, 0.06168831168831179), (-0.7177419354838709, -0.33333333333333326), (-0.7298387096774193, -0.6201298701298702)]
ys = [
[(-0.6532258064516129, -0.3982683982683982), (-0.4838709677419355, -0.18181818181818188), (-0.2701612903225805, -0.030303030303030276), (-0.217741935483871, -0.18181818181818188), (-0.4193548387096774, -0.30627705627705626), (-0.5524193548387096, -0.45779220779220775)],
[(-0.3870967741935484, -0.1439393939393938), (-0.10483870967741926, -0.04653679653679643), (0.24193548387096775, -0.15476190476190466), (0.16129032258064524, -0.2738095238095237), (-0.07258064516129026, -0.1872294372294372), (-0.31854838709677424, -0.25757575757575757)],
[(0.157258064516129, -0.06818181818181812), (0.35483870967741926, -0.3170995670995671), (0.403225806451613, -0.5714285714285714), (0.282258064516129, -0.5930735930735931), (0.22983870967741948, -0.38203463203463206), (0.08467741935483875, -0.1926406926406925)],
[(0.42741935483870974, -0.051948051948051965), (0.5080645161290323, -0.3387445887445887), (0.5080645161290323, -0.6904761904761905), (0.41532258064516125, -0.6634199134199135), (0.41129032258064524, -0.36038961038961037), (0.36290322580645173, -0.1114718614718615)]
]
man = [(-0.7782258064516129, -0.7012987012987013), (-0.8024193548387096, -0.658008658008658), (-0.7419354838709677, -0.5606060606060607), (-0.7258064516129031, -0.2629870129870129), (-0.8266129032258064, -0.2629870129870129), (-0.8266129032258064, -0.1926406926406925), (-0.7177419354838709, -0.1926406926406925), (-0.6935483870967741, -0.19805194805194803), (-0.6814516129032258, -0.15476190476190466), (-0.7379032258064515, -0.1114718614718615), (-0.7379032258064515, -0.030303030303030276), (-0.6774193548387096, 0.03463203463203479), (-0.596774193548387, 0.02922077922077926), (-0.564516129032258, -0.0032467532467532756), (-0.5927419354838709, -0.1114718614718615), (-0.6370967741935484, -0.1439393939393938), (-0.6451612903225806, -0.19805194805194803), (-0.6088709677419354, -0.20346320346320335), (-0.532258064516129, -0.2629870129870129), (-0.5766129032258064, -0.3225108225108225), (-0.6290322580645161, -0.2629870129870129), (-0.6330645161290323, -0.5119047619047619), (-0.6008064516129031, -0.6363636363636364), (-0.6411290322580645, -0.6742424242424243), (-0.6814516129032258, -0.566017316017316), (-0.7016129032258064, -0.5768398268398268), (-0.7782258064516129, -0.7012987012987013)]

ts = [0.25, 0.5, 0.82, 1.0]

ts = torch.tensor(ts, requires_grad=True)
X0 = torch.tensor(X0)
ys = torch.tensor(ys)
man = torch.tensor(man)

def plot_shape(X, t, **kargs):
	x, y = torch.cat((X, X[:1]), 0).T
	pl.plot(x, y, color=(t, 0, 1-t), **kargs)

sigma = torch.tensor(0.4, requires_grad=True)
def get_K(c, x=None):
	if x == None: x = c
	distances = ((x[:,None] - c[None]) ** 2).sum(-1, keepdim=True)
	K = torch.exp(- 0.5 * distances / sigma**2)
	invG = lambda a: (K * a).sum(1)
	return invG

def eval_geod(geod_c, geod_a, X0, ts):
	len_geod = len(geod_c)
	dt = 1.0 / (len_geod-1)
	X = X0.clone()
	t, j = 0.0, 0
	res = []
	for i in range(len_geod-1):
		ci, ai = 0.5*(geod_c[i]+geod_c[i+1]), 0.5*(geod_a[i]+geod_a[i+1])
		v = get_K(ci, X)(ai)
		t1 = t+dt
		while j < len(ts) and t1 >= ts[j]:
			Xtj = X + ((ts[j] if j != len(ts)-1 else ts[j].detach()) - t) * v
			res.append(Xtj)
			j += 1
		if j >= len(ts): break
		X += v * dt
		t = t1
	return torch.stack(res)

def plot_evol(c0, a0, X0, ts):
	geod_c, geod_a = Riemannian_exponential(c0, a0, get_K)
	with torch.no_grad():
		plot_shape(X0, 0)
		Xs = eval_geod(geod_c, geod_a, X0, ts.detach())
		for t, x in zip(ts, Xs):
			plot_shape(x, t.detach())

def loss_fun(c, a, **kargs):
	geod_c, geod_a = Riemannian_exponential(c, a, get_K, **kargs)
	Xs = eval_geod(geod_c, geod_a, X0, ts)
	cost = 0
	for x, y in zip(Xs, ys):
		cost += (x - y).norm()
	return cost

n = 60
c0 = (torch.rand((n, 2)) - 0.5 + torch.tensor([-0.96, -0.1])) * torch.tensor([1.2, 1])
a0 = (torch.rand((n, 2)) - 0.5) * 0.4
c0.requires_grad_(True)
a0.requires_grad_(True)

figsize = torch.tensor((7, 4.5))
b0, b1 = torch.cat((X0[None], ys)).min(0)[0].min(0)[0], torch.cat((X0[None], ys)).max(0)[0].max(0)[0]
b0[0] -= 0.33
b1[1] += 0.05
mid = 0.5*(b0+b1)
var = (b1-b0)*1.2
var = (var / figsize).max() * figsize
pl.figure(figsize=figsize)
b0, b1 = mid-0.54*var, mid+0.54*var
def step_plot(c0, a0, X0, i):
	pl.clf()
	pl.xlim(b0[0], b1[0])
	pl.ylim(b0[1], b1[1])
	for y, t in zip(ys, ts):
		plot_shape(y, t.detach(), ls='--')
	plot_evol(c0, a0, X0, ts)
	pl.quiver(c0[:,0].detach(), c0[:,1].detach(), 0.12*a0[:,0].detach(), 0.12*a0[:,1].detach(), color='green', width=0.0025)
	pl.scatter(c0[:,0].detach(), c0[:,1].detach(), color='orange', s=12)
	pl.savefig(f"test_{i}.png")

n_steps = 1000
next_plot = 0
dt_plot = 5
optimizer = torch.optim.Adam([c0, a0, sigma, ts])
for i in range(n_steps):
	ti = time()
	optimizer.zero_grad()
	loss = loss_fun(c0, a0)
	loss.backward(retain_graph=True)
	if i % 5 == 0:
		print(f"[{i}] Loss: {loss.detach()}", end=" \t")
		print(sigma.detach(), ts.detach(), time() - ti)
	optimizer.step()
	if i >= next_plot:
		next_plot += dt_plot
		dt_plot *= 1.085
		step_plot(c0, a0, X0, i)
print((a0.detach().norm(dim=1)[:,None] * c0.detach()).sum(0) / a0.detach().norm(dim=1).sum(), c0.detach().max(0)[0] - c0.detach().min(0)[0])

N = 30
geod_c, geod_a = Riemannian_exponential(c0, a0, get_K, n_points=2*N)
with torch.no_grad():
	times = torch.linspace(0, 1, N)
	Xs = eval_geod(geod_c, geod_a, man, times)
	b0, b1 = Xs.min(0)[0].min(0)[0], Xs.max(0)[0].max(0)[0]
	mid = 0.5*(b0+b1)
	var = (b1-b0)*1.2
	var = (var / figsize).max() * figsize
	pl.figure(figsize=figsize)
	b0, b1 = mid-0.54*var, mid+0.54*var
	for t, x in zip(times, Xs):
		pl.clf()
		pl.xlim(b0[0], b1[0])
		pl.ylim(b0[1], b1[1])
		plot_shape(x, t)
		pl.savefig(f"evol_{int(30*t-1e-7)}.png")