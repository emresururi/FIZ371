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

# Example: White Token Out Of A Bag
(MacKay Exercise 3.12)

A bag contains one token, known to be either white or black. A white token is put in, the bag is shaken, and a token is drawn out, which proves to be white. What is now the chance of drawing another white token?


## Analytical Solution

Given that we have picked a white token, the probability that it had already contained another white, thus after adding the white it became WW:

$$P(WW|W) = \frac{P(W|WW).P(WW)}{P(W)} = \frac{1\times1/2}{P(W)}=\frac{1/2}{P(W)}$$

and the probability that it had a black token beforehand, so after adding the white it became WB is:

$$P(WB|W) = \frac{P(W|WB).P(WB)}{P(W)} = \frac{1/2\times1/2}{P(W)}=\frac{1/4}{P(W)}$$

We can evaluate the $P(W)$ in the denominator via two ways: 

* By normalization:  
Since there is no other possibility, $P(WW|W) + P(WS|W)$ should be equal to 1. Hence:

$$\frac{1/2}{P(W)} + \frac{1/4}{P(W)} = \frac{3/4}{P(W)} = 1\Rightarrow P(W)=\frac{3}{4}$$

* By inference:  
As $P(A) = \sum_i{P(A|B_i)P(B_i)}$  
So, expanding and summing over all the possible cases (WW and WB) yield:

$$P(W) = P(W|WW).P(WW) + P(W|WB).P(WB) = (1\times1/2) + (1/2\times1/2)=\frac{3}{4}$$

Substituting $P(W)$ in the above equations:

$$\boxed{P(WW|W)=\frac{1/2}{P(W)}=\frac{2}{3}}$$

$$P(WB|W)=\frac{1/4}{P(W)}=\frac{1}{3}$$

+++

## Monte Carlo Solution

```{code-cell} ipython3
import numpy as np
```

```{code-cell} ipython3
N = 10000000
```

```{code-cell} ipython3
A = np.random.randint(0,2,(N,3))
# 0 : Black
# 1 : White
```

```{code-cell} ipython3
A[:,1] = 1
A[:,2] = 999
```

```{code-cell} ipython3
A[A[:,0] == A[:,1],2] = 1 # If both are white, outcome will be white
B = A.copy()

# If one is black, one is white, the outcome can be either
B[B[:,2]==999,2] = np.random.randint(0,2,np.sum(B[:,2]==999))
```

```{code-cell} ipython3
# Filter out the outcomes with white:
W = B[B[:,2] == 1]
```

```{code-cell} ipython3
num_TOT = W.shape[0]
num_W = np.sum(W[:,0] == 1)
num_B = num_TOT - num_W
print("White: {:d}/{:d} ({:5f})\t|\tBlack: {:d}/{:d} ({:5f})".format(num_W,num_TOT,
                                                                     num_W/num_TOT,
                                                                     num_B,num_TOT,
                                                                     num_B/num_TOT))
```

```{code-cell} ipython3

```
