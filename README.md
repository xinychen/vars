# vars
[![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)
![Python 3.7](https://img.shields.io/badge/Python-3.7-blue.svg)
[![repo size](https://img.shields.io/github/repo-size/xinychen/vars.svg)](https://github.com/xinychen/vars/archive/master.zip)
[![GitHub stars](https://img.shields.io/github/stars/xinychen/vars.svg?logo=github&label=Stars&logoColor=white)](https://github.com/xinychen/vars)

<h6 align="center">Made by Xinyu Chen • :globe_with_meridians: <a href="https://xinychen.github.io">https://xinychen.github.io</a></h6>

The scientific question is how to **discover dynamic patterns from spatiotemporal data**. We utilize the Vector Autoregression (VAR) as a basic tool to explore the spatiotemporal data in real-world applications.

<br>

## Datasets

### Fluid Dynamics

To analyze the underlying spatiotemporal patterns of fluid dynamics, we consider the cylinder wake dataset in which the flow shows a supercritical Hopf bifurcation. [The cylinder wake dataset](http://dmdbook.com/) is collected from the fluid flow passing a circular cylinder with laminar vortex shedding at Reynolds number Re = 100, which is larger than the critical Reynolds number, using direct numerical simulations of the Navier-Stokes equations. This is a representative three-dimensional flow dataset in fluid dynamics, consisting of matrix-variate time series of vorticity field snapshots for the wake behind a cylinder. The dataset is of size $199\times 449\times 150$, representing 199-by-449 vorticity fields with 150 time snapshots.

```python
import numpy as np
import seaborn as sns
import scipy.io
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
color = scipy.io.loadmat('CCcool.mat')
cc = color['CC']
newcmp = LinearSegmentedColormap.from_list('', cc)

tensor = np.load('tensor.npz')['arr_0']
tensor = tensor[:, :, : 150]
M, N, T = tensor.shape

plt.rcParams['font.size'] = 13
plt.rcParams['mathtext.fontset'] = 'cm'
fig = plt.figure(figsize = (7, 8))
id = np.array([5, 10, 15, 20, 25, 30, 35, 40])
for t in range(8):
    ax = fig.add_subplot(4, 2, t + 1)
    ax = sns.heatmap(tensor[:, :, id[t] - 1], cmap = newcmp, vmin = -5, vmax = 5, cbar = False)
    ax.contour(np.linspace(0, N, N), np.linspace(0, M, M), tensor[:, :, id[t] - 1],
               levels = np.linspace(0.15, 15, 30), colors = 'k', linewidths = 0.7)
    ax.contour(np.linspace(0, N, N), np.linspace(0, M, M), tensor[:, :, id[t] - 1],
               levels = np.linspace(-15, -0.15, 30), colors = 'k', linestyles = 'dashed', linewidths = 0.7)
    plt.xticks([])
    plt.yticks([])
    plt.title(r'$t = {}$'.format(id[t]))
    for _, spine in ax.spines.items():
        spine.set_visible(True)
plt.show()
fig.savefig('fluid_flow_heatmap.png', bbox_inches = 'tight')
```

<p align="center">
<img align="middle" src="graphics/fluid_flow_heatmap.png" alt="drawing" width="300">
</p>

<p align="center"><b>Figure 1</b>: Heatmaps (snapshots) of the fluid flow at times $t=5,10,\ldots,40$. It shows that the snapshots at times $t=5$ and $t=35$ are even same, and the snapshots at times $t=10$ and $t=40$ are also even same, allowing one to infer the seasonality as 30 for the first 50 snapshots.</p>

<br>

### Sea Surface Temperature

[The sea surface temperature dataset](https://psl.noaa.gov/data/gridded/data.noaa.oisst.v2.html) covers weekly means of temperature on the spatial resolution of (1 degree latitude, 1 degree longitude)-grid, and there are $180\times 360$ global grids (i.e., 64,800 grids) in total. The dataset spans a 30-year period from 1990 to 2019, and the time dimension is of length 1,565 (weeks). Therefore, the data can be represented as a matrix of size $64800\times 1565$, which seems to be high-dimensional.

### USA Surface Temperature Data

[Daymet project](https://daac.ornl.gov/DAYMET) provides long-term and continuous estimates of daily weather parameters such as maximum and minimum daily temperature for North America. There are 5,380 stations over the United States Mainland. We use the daily maximum temperature data in the United States Mainland from 2010 to 2021 (i.e., 12 years or 4,380 days in total) for evaluation. The data can be represented as a matrix of size $5380\times 4380$.

### NYC Taxi Trips

We consider to use an [NYC (yellow) taxi trip dataset](https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page). We use 69 zones in Manhattan as pickup/dropoff zones and aggregate daily taxi trip volume of the data from 2012 to 2021. Therefore, the daily trip volume tensor is of size $69\times 69\times 3653$.

<br>

## Algorithm Implementation: Time-Varying Reduced-Rank VAR


```python
import numpy as np

def update_cg(w, r, q, Aq, rold):
    alpha = rold / np.inner(q, Aq)
    w = w + alpha * q
    r = r - alpha * Aq
    rnew = np.inner(r, r)
    q = r + (rnew / rold) * q
    return w, r, q, rnew

def ell_v(Y, Z, W, G, V_transpose, X, temp2, d, T):
    rank, dN = V_transpose.shape
    temp = np.zeros((rank, dN))
    for t in range(d, T):
        temp3 = np.outer(X[t, :], Z[:, t - d])
        Pt = temp2 @ np.kron(X[t, :].reshape([rank, 1]), V_transpose) @ Z[:, t - d]
        temp += np.reshape(Pt, [rank, rank], order = 'F') @ temp3
    return temp

def conj_grad_v(Y, Z, W, G, V_transpose, X, d, T, maxiter = 5):
    rank, dN = V_transpose.shape
    temp1 = W @ G
    temp2 = temp1.T @ temp1
    v = np.reshape(V_transpose, -1, order = 'F')
    temp = np.zeros((rank, dN))
    for t in range(d, T):
        temp3 = np.outer(X[t, :], Z[:, t - d])
        Qt = temp1.T @ Y[:, t - d]
        temp += np.reshape(Qt, [rank, rank], order = 'F') @ temp3
    r = np.reshape(temp - ell_v(Y, Z, W, G, V_transpose, X, temp2, d, T), -1, order = 'F')
    q = r.copy()
    rold = np.inner(r, r)
    for it in range(maxiter):
        Q = np.reshape(q, (rank, dN), order = 'F')
        Aq = np.reshape(ell_v(Y, Z, W, G, Q, X, temp2, d, T), -1, order = 'F')
        v, r, q, rold = update_cg(v, r, q, Aq, rold)
    return np.reshape(v, (rank, dN), order = 'F')

def trvar(mat, d, rank, maxiter = 50):
    N, T = mat.shape
    Y = mat[:, d : T]
    Z = np.zeros((d * N, T - d))
    for k in range(d):
        Z[k * N : (k + 1) * N, :] = mat[:, d - (k + 1) : T - (k + 1)]
    u, _, v = np.linalg.svd(Y, full_matrices = False)
    W = u[:, : rank]
    u, _, _ = np.linalg.svd(Z, full_matrices = False)
    V = u[:, : rank]
    u, _, _ = np.linalg.svd(mat.T, full_matrices = False)
    X = u[:, : rank]
    del u
    loss = np.zeros(maxiter)
    for it in range(maxiter):
        temp1 = np.zeros((N, rank * rank))
        temp2 = np.zeros((rank * rank, rank * rank))
        for t in range(d, T):
            temp = np.kron(X[t, :].reshape([rank, 1]), V.T) @ Z[:, t - d]
            temp1 += np.outer(Y[:, t - d], temp)
            temp2 += np.outer(temp, temp)
        G = np.linalg.pinv(W) @ temp1 @ np.linalg.inv(temp2)
        W = temp1 @ G.T @ np.linalg.inv(G @ temp2 @ G.T)
        V = conj_grad_v(Y, Z, W, G, V.T, X, d, T).T
        temp3 = W @ G
        for t in range(d, T):
            X[t, :] = np.linalg.pinv(temp3 @ np.kron(np.eye(rank), (V.T @ Z[:, t - d]).reshape([rank, 1]))) @ Y[:, t - d]
    return W, G, V, X
```

<br>

## Experiment on Fluid Dynamics

- Model setting: `rank = 7` and `d = 1`.

```python
import numpy as np
import time

tensor = np.load('tensor.npz')['arr_0']
tensor = tensor[:, :, : 150]
M, N, T = tensor.shape
mat = np.zeros((M * N, 100))
mat[:, : 50] = np.reshape(tensor[:, :, : 50], (M * N, 50), order = 'F')
for t in range(50):
    mat[:, t + 50] = np.reshape(tensor[:, :, 50 + 2 * t + 1], (M * N), order = 'F')

for rank in [7]:
    for d in [1]:
        start = time.time()
        W, G, V, X = trvar(mat, d, rank)
        print('rank R = {}'.format(rank))
        print('Order d = {}'.format(d))
        end = time.time()
        print('Running time: %d seconds'%(end - start))
```

- Result visualization: spatial modes

```python
import seaborn as sns
import scipy.io
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
color = scipy.io.loadmat('CCcool.mat')
cc = color['CC']
newcmp = LinearSegmentedColormap.from_list('', cc)

plt.rcParams['font.size'] = 12
fig = plt.figure(figsize = (7, 8))
ax = fig.add_subplot(4, 2, 1)
sns.heatmap(np.mean(tensor, axis = 2), 
            cmap = newcmp, vmin = -5, vmax = 5, cbar = False)
ax.contour(np.linspace(0, N, N), np.linspace(0, M, M), np.mean(tensor, axis = 2), 
            levels = np.linspace(0.15, 15, 20), colors = 'k', linewidths = 0.7)
ax.contour(np.linspace(0, N, N), np.linspace(0, M, M), np.mean(tensor, axis = 2), 
            levels = np.linspace(-15, -0.15, 20), colors = 'k', linestyles = 'dashed', linewidths = 0.7)
plt.xticks([])
plt.yticks([])
plt.title('Mean vorticity field')
for _, spine in ax.spines.items():
    spine.set_visible(True)
for t in range(7):
    if t == 0:
        ax = fig.add_subplot(4, 2, t + 2)
    else:
        ax = fig.add_subplot(4, 2, t + 2)
    ax = sns.heatmap(W[:, t].reshape((199, 449), order = 'F'), 
                     cmap = newcmp, vmin = -0.03, vmax = 0.03, cbar = False)
    if t < 3:
        num = 20
    else:
        num = 10
    ax.contour(np.linspace(0, N, N), np.linspace(0, M, M), W[:, t].reshape((199, 449), order = 'F'),  
               levels = np.linspace(0.0005, 0.05, num), colors = 'k', linewidths = 0.7)
    ax.contour(np.linspace(0, N, N), np.linspace(0, M, M), W[:, t].reshape((199, 449), order = 'F'), 
               levels = np.linspace(-0.05, -0.0005, num), colors = 'k', linestyles = 'dashed', linewidths = 0.7)
    plt.xticks([])
    plt.yticks([])
    plt.title('Spatial mode {}'.format(t + 1))
    for _, spine in ax.spines.items():
        spine.set_visible(True)
plt.show()
fig.savefig("fluid_mode_trvar.png", bbox_inches = "tight")
```

<p align="center">
<img align="middle" src="graphics/fluid_mode_trvar.png" alt="drawing" width="300">
</p>

<p align="center"><b>Figure 2</b>: Mean vorticity field and spatial modes of the fluid flow. Spatial modes are plotted by the columns of $\boldsymbol{W}$ in which seven panels correspond to the rank $R = 7$. Note that the colorbars of all modes are on the same scale.</p>

- Result visualization: temporal modes

```python
import matplotlib.pyplot as plt
import matplotlib.patches as patches

plt.rcParams['font.size'] = 14
plt.rcParams['mathtext.fontset'] = 'cm'
fig = plt.figure(figsize = (7, 9))
for t in range(7):
    ax = fig.add_subplot(7, 1, t + 1)
    plt.plot(np.arange(1, 101, 1), X[:, t], linewidth = 2, alpha = 0.8, color = 'red')
    plt.xlim([1, 100])
    plt.xticks(np.arange(0, 101, 10))
    if t < 6:
        ax.tick_params(labelbottom = False)
    ax.tick_params(direction = "in")
    rect = patches.Rectangle((0, -1), 50, 2, linewidth=2, 
                             edgecolor='gray', facecolor='red', alpha = 0.1)
    ax.add_patch(rect)
    rect = patches.Rectangle((50, -1), 50, 2, linewidth=2, 
                             edgecolor='gray', facecolor='yellow', alpha = 0.1)
    ax.add_patch(rect)
plt.xlabel('$t$')
plt.show()
fig.savefig("fluid_temporal_mode.pdf", bbox_inches = "tight")
```

<br>

## Experiment on Sea Surface Temperature

- Model setting: 
