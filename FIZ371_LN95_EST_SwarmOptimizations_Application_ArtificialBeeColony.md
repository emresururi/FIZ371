---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.5
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

## Swarm Optimizations Hands-on Applications
**FIZ371 - Scientific & Technical Computations | 24/05/2021**

* Ant Colony Optimization
* **Artificial Bee Colony**

Dr. Emre S. Tasci <emre.tasci@hacettepe.edu.tr>

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
```

```{code-cell} ipython3
def fun(x):
    return np.sin(x/2)/x +(-1)**np.round(x*1.25) * np.heaviside(x-2,1.5)*0.20
```

```{code-cell} ipython3
x_min = -300
x_max =  150
x_delta = x_max - x_min
neighbourhood = x_delta / 50

x = np.linspace(x_min,x_max,500)
y = fun(x)
plt.plot(x,y,"r-")
plt.show()
```

```{code-cell} ipython3
N = 100 # Number of bees

# Distribute the bees randomly
xs = x_min + np.random.rand(N) * x_delta
ys = fun(xs)

plt.plot(x,y,"r-")
plt.plot(xs,ys,"bo",markersize=2.0)
plt.show()
```

```{code-cell} ipython3
maxes = -np.sort(-ys)
print(maxes)
```

```{code-cell} ipython3
print(xs[ys > maxes[5]])
```

```{code-cell} ipython3
n_scouts = int(np.ceil(N/10))
n_supers = np.random.randint(n_scouts/2)+1
n_mehs   = n_scouts - n_supers
  
n_super_followers = np.random.randint(N*0.25,N*0.40)
n_meh_followers = int(N/2) - n_super_followers
n_random_fliers = N - n_super_followers - n_meh_followers

n_per_super_min = int(np.floor(n_super_followers/n_supers))
n_super_plus = n_super_followers - n_per_super_min * n_supers

n_per_meh_min = int(np.floor(n_meh_followers/n_mehs))
n_meh_plus = n_meh_followers - n_per_meh_min * n_mehs


print("There are {:} super-sites and {:} meh sites.\n"\
      .format(n_supers,n_mehs))
print("Number of followers allocated for the super sites: {}"\
     .format(n_super_followers))
print("There will be {:} per super site, with an overhead of {:}"\
     .format(n_per_super_min,n_super_plus))
print("-"*45)
print("Number of followers allocated for the meh sites: {}"\
     .format(n_meh_followers))
print("There will be {:} per meh site, with an overhead of {:}"\
     .format(n_per_meh_min,n_meh_plus))
print("\nThe range is [{:},{:}] and the neighbourhood is {:.2f}"\
     .format(x_min,x_max,neighbourhood))
```

```{code-cell} ipython3
# supers
filter_super = ys > maxes[n_supers]
x_supers = xs[filter_super]
y_supers = ys[filter_super]
xy_supers = np.column_stack((x_supers.T,y_supers.T))
print(xy_supers)
print(x_supers,"\n",y_supers)

print("-"*45)

# mehs
filter_meh = np.logical_and(ys > maxes[n_supers+n_mehs]\
                            ,ys <= maxes[n_supers])
x_mehs = xs[filter_meh]
y_mehs = ys[filter_meh]
xy_mehs = np.column_stack((x_mehs,y_mehs))
print(xy_mehs)
print(x_mehs,"\n",y_mehs)

print(xy_mehs[xy_mehs[:,1].argsort()])
```

```{code-cell} ipython3
xy_mehs
```

```{code-cell} ipython3
np.insert(xy_mehs,0,([4,5]),0)
```

```{code-cell} ipython3
5%3
```

```{code-cell} ipython3

```
