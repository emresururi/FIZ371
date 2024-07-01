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

# Optimization, Minimization & Fitting Notes & Recipes

```{code-cell} ipython3
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
```

```{code-cell} ipython3
np.random.seed(371)
```

```{code-cell} ipython3
def g(x,mu,sigma):
    return -np.exp(-(x-mu)**2/(2*sigma**2))
```

```{code-cell} ipython3
x = np.linspace(0,15,100)
mu = 5
sigma = 1
plt.plot(x,g(x,mu,sigma))
plt.show()
```

```{code-cell} ipython3
optimize.minimize(g,x0=2,args=(mu,sigma))
```

# Gradient Descent Algorithm

+++

## Parabol

```{code-cell} ipython3
def f(x,abc):
    return abc[0]*x**2+abc[1]*x+abc[2]
def g(x,abc):
    # Derivative
    return 2*abc[0]*x+abc[1]
```

```{code-cell} ipython3
xx = np.linspace(-5,10,100)
abc = np.array([2,3,-4])
plt.plot(xx,f(xx,abc))
plt.show()
```

```{code-cell} ipython3
x = 5
N = 50
eta = .4
tolerance = 1E-4
xs_so_far = [x]
fs_so_far = [f(x,abc)]
for i in range(N):
    gg = g(x,abc)
    print("Step #{:d}".format(i+1))
    print("The derivative (gradient) at x = {:7.5f} is {:5.3f}"\
          .format(x,gg))
    if(np.abs(gg)<tolerance):
        print("\tAs it is sufficiently close to zero, we have found the minima!")
        break
    elif(gg>0):
        print("\tAs it is positive, go left by: "+
              "(this amount)*eta(={:.2f}).".format(eta))
    else:
        print("\tAs it is negative, go right by: "+
              "|this amount|*eta(={:.2f}).".format(eta))

    delta = -gg*eta
    x0 = x
    x = x + delta
    xs_so_far.append(x)
    fs_so_far.append(f(x,abc))
    print("\t==> The new x is {:7.5f}{:+7.5f}={:7.5f}".format(x0,delta,x))
    plt.plot(xx,f(xx,abc),color="orange")
    plt.plot(xs_so_far,fs_so_far,"*-")
    plt.show()

    print("-"*45)
```

```{code-cell} ipython3
# Real minimum:
np.roots([2*abc[0],abc[1]]) # root of 2ax + b
```

## "Negative" Gaussian

```{code-cell} ipython3
def f(x,mu,sigma):
    return -np.exp(-(x-mu)**2/(2*sigma**2))
def g(x,mu,sigma):
    return (x-mu)/(sigma**2)*np.exp(-(x-mu)**2/(2*sigma**2))
```

```{code-cell} ipython3
mu = 5
sigma = 1
```

```{code-cell} ipython3
xx = np.linspace(0,15,100)
plt.plot(xx,f(xx,mu,sigma))
plt.show()
```

```{code-cell} ipython3
x = 8
N = 60
eta = .4
tolerance = 1E-4
xs_so_far = [x]
fs_so_far = [f(x,mu,sigma)]
for i in range(N):
    gg = g(x,mu,sigma)
    print("Step #{:d}".format(i+1))
    print("The derivative (gradient) at x = {:7.5f} is {:5.4f}"\
          .format(x,gg))
    if(np.abs(gg)<tolerance):
        print("\tAs it is sufficiently close to zero, we have found the minima!")
        break
    elif(gg>0):
        print("\tAs it is positive, go left by: "+
              "(this amount)*eta(={:.2f}).".format(eta))
    else:
        print("\tAs it is negative, go right by: "+
              "|this amount|*eta(={:.2f}).".format(eta))

    delta = -gg*eta
    x0 = x
    x = x + delta
    xs_so_far.append(x)
    fs_so_far.append(f(x,mu,sigma))
    print("\t==> The new x is {:7.5f}{:+7.5f}={:7.5f}".format(x0,delta,x))
    plt.plot(xx,f(xx,mu,sigma),color="orange")
    plt.plot(xs_so_far,fs_so_far,"*-")
    plt.show()

    print("-"*45)
```

# Fitting parameters via gradient descent algorithm

```{code-cell} ipython3
mu = 5
sigma = 1

N = 10
x = np.random.rand(N)*4+3
t = f(x,mu,sigma)

xx = np.linspace(3,7,100)

plt.plot(xx,f(xx,mu,sigma),color="orange")
plt.plot(x,t,"o")
plt.show()
```

$\newcommand{\diff}{\text{d}}
\newcommand{\dydx}[2]{\frac{\text{d}#1}{\text{d}#2}}
\newcommand{\ddydx}[2]{\frac{\text{d}^2#1}{\text{d}#2^2}}
\newcommand{\pypx}[2]{\frac{\partial#1}{\partial#2}}
\newcommand{\unit}[1]{\,\text{#1}}$

We have the data points, we know the function but we don't have the mu & sigma.

$$f(x;\mu,\sigma)=-\exp{\left[-\frac{(x-\mu)^2}{2\sigma^2}\right]}$$

The function we are going to try to minimize will be the difference between the real values ($\{t_i\}$) corresponding to $\{x_i\}$ and the projected values ($\{y_i\}$):

$$F(x_i,t_i,\mu,\sigma) = t_i - f(x_i;\mu,\sigma)$$

Begin by calculating the derivatives:

$$\pypx{F}{\mu}=\frac{x_i-\mu}{\sigma^2}\exp{\left[-\frac{(x_i-\mu)^2}{2\sigma^2}\right]}\\
\pypx{F}{\sigma}=\frac{(x_i-\mu)^2}{\sigma^3}\exp{\left[-\frac{(x_i-\mu)^2}{2\sigma^2}\right]}$$

_(don't forget that $\{x_i\}$ and $\{t_i\}$ are fixed!)_

+++

Can you see the problem in this approach? As $\{t_i\}$ are fixed, the problem is reduced to finding the $(\mu,\sigma)$ set that will make $f(x_i;\mu,\sigma)$ minimum, regardless of $\{t_i\}$ values. If we follow this approach, we will end up with $(\mu,\sigma)$ that will most likely fix the values all very close to 0.

You are invited to try this approach, i.e.,

```python
def F_mu(x,mu,sigma):
    return (x-mu)/sigma**2*np.exp(-(x-mu)**2/(2*sigma**2))
def F_sigma(x,mu,sigma):
    return (x-mu)**2/sigma**3*np.exp(-(x-mu)**2/(2*sigma**2))
```

+++

But what we really have in mind is the fact that, for a given $x_i$, we want to find values as close to the corresponding $t_i$ as possible. One way to obtain this would be to define the error function as:

$$F(x_i,t_i,\mu,\sigma) = \left[t_i - f(x_i;\mu,\sigma)\right]^2=\left\{t_i -
\left[-\exp{\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)}\right]
\right\}^2$$

and then we would have the following derivatives:

$$\pypx{F}{\mu} = \frac{2(x_i-\mu)}{\sigma^2}\exp{\left[-\frac{(x_i-\mu)^2}{2\sigma^2}\right]}\left\{t_i+\exp{\left[-\frac{(x_i-\mu)^2}{2\sigma^2}\right]}\right\}\\
\pypx{F}{\sigma}=\frac{2(x_i-\mu)^2}{\sigma^3}\exp{\left[-\frac{(x_i-\mu)^2}{2\sigma^2}\right]}\left\{t_i+\exp{\left[-\frac{(x_i-\mu)^2}{2\sigma^2}\right]}\right\}$$

_(Evaluated via WolframAlpha: [1](https://www.wolframalpha.com/input?i=d%2Fdu+%28t%2Bexp%28-%28x-u%29%5E2%2F%282*s%5E2%29%29%29%5E2), [2](https://www.wolframalpha.com/input?i=d%2Fds+%28t%2Bexp%28-%28x-u%29%5E2%2F%282*s%5E2%29%29%29%5E2))_

```{code-cell} ipython3
def F_mu(x,t,mu,sigma):
    return 2*(x-mu)/sigma**2*np.exp(-(x-mu)**2/(2*sigma**2))*\
(t+np.exp(-(x-mu)**2/(2*sigma**2)))
def F_sigma(x,t,mu,sigma):
    return 2*(x-mu)**2/sigma**3*np.exp(-(x-mu)**2/(2*sigma**2))*\
(t+np.exp(-(x-mu)**2/(2*sigma**2)))
```

```{code-cell} ipython3
np.array([x,t]).T
```

```{code-cell} ipython3
eta = 1

# Starting values
mu_opt = 2.7
sigma_opt = 2.3
tolerance = 1E-4

for i in range(10000):
    for ii in range(x.size):
        xi = x[ii]
        ti = t[ii]
        #print(xi,ti)
        F_mu_xi = F_mu(xi,ti,mu_opt,sigma_opt)
        F_sigma_xi = F_sigma(xi,ti,mu_opt,sigma_opt)
        mu_opt -= eta*F_mu_xi
        sigma_opt -= eta*F_sigma_xi
    total_absolute_error = np.sum(np.abs(t-f(x,mu_opt,sigma_opt)))
    if(total_absolute_error < tolerance):
        print(("As the sum of the absolute errors is sufficiently close to zero ({:.7f}),\n"+
              "\tbreaking the iteration at the {:d}. step!").
              format(total_absolute_error,i+1))
        break
print("mu: {:.4f}\tsigma: {:.4f}".format(mu_opt,sigma_opt))

plt.plot(xx,f(xx,mu,sigma),color="orange")
plt.plot(xx,f(xx,mu_opt,sigma_opt),":b")
plt.plot(x,t,"o")
plt.show()
```

## Stochastic Gradient Descent Algorithm

In this approach, instead of optimizing the variables at every step for one data point, we optimize them as a whole:

```{code-cell} ipython3
eta = 0.1

# Starting values
mu_opt = 2.7
sigma_opt = 2.3
tolerance = 1E-4
total_absolute_error0 = 1000

for i in range(10000):
    d_mu    = -eta*np.sum(F_mu(x,t,mu_opt,sigma_opt))
    d_sigma = -eta*np.sum(F_sigma(x,t,mu_opt,sigma_opt))

    mu_opt    += d_mu
    sigma_opt += d_sigma

    total_absolute_error = np.sum(np.abs(t-f(x,mu_opt,sigma_opt)))

    if(total_absolute_error < tolerance):
        print(("As the sum of the absolute errors is sufficiently close to zero ({:.7f}),\n"+
              "\tbreaking the iteration at the {:d}. step!").
              format(total_absolute_error,i+1))
        break
print("mu: {:.4f}\tsigma: {:.4f}".format(mu_opt,sigma_opt))

plt.plot(xx,f(xx,mu,sigma),color="orange")
plt.plot(xx,f(xx,mu_opt,sigma_opt),":b")
plt.plot(x,t,"o")
plt.show()
```

```{code-cell} ipython3
def F(musigma,x,t):
    return np.sum((t + np.exp(-(x-musigma[0])**2/(2*musigma[1]**2)))**2)
```

```{code-cell} ipython3
opt = optimize.minimize(F,x0=(2.7,2.3),args=(x,t))
opt.x,opt.fun
```

```{code-cell} ipython3
opt = optimize.minimize(F,x0=(2.7,2.3),args=(x,t),bounds=[(3,6.5),(None,None)])
opt.x,opt.fun
```

```{code-cell} ipython3

```
