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

# Monty Hall Problem
**FIZ371 - Scientific & Technical Calculations | 08/03/2021**

Emre S. Tasci <emre.tasci@hacettepe.edu.tr>

```{code-cell} ipython3
import numpy as np
import random

N = 100000

#np.random.seed(371)
```

# 3 doors, 1 door is opened

```{code-cell} ipython3
tries = np.random.randint(1,4,(N,5))
tries[:,2:5] = 0

# 0th col: Door with the prize
# 1st col: Door that the contestant has chosen
# 2nd col: Door that Monty opens
# 3rd col: 1 if the user wins by sticking, 0 otherwise
# 4th col: 1 if the user wins by switching, 0 otherwise
```

```{code-cell} ipython3
# tries
```

```{code-cell} ipython3
filt = tries[:,0]==tries[:,1]
```

```{code-cell} ipython3
#tries[filt]
```

```{code-cell} ipython3
for i in np.array(range(N))[filt]:
    possible = np.array([1,2,3])
    possible = possible[possible != tries[i,0]]
    tries[i,2] = random.choice(possible)

for i in np.array(range(N))[filt]:
    possible = np.array([1,2,3])
    possible = possible[possible != tries[i,0]]
    tries[i,3] = 1
```

```{code-cell} ipython3
nfilt = np.invert(filt)
```

```{code-cell} ipython3
#tries[nfilt]
```

```{code-cell} ipython3
for i in np.array(range(N))[nfilt]:
    possible = np.array([1,2,3])
    possible = possible[possible != tries[i,0]]
    possible = possible[possible != tries[i,1]]
    tries[i,2] = possible[0]
    tries[i,4] = 1
```

```{code-cell} ipython3
tries
```

```{code-cell} ipython3
print("# of total wins if sticks  : {:} [prob: {:.5f}]"\
      .format(np.sum(tries[:,3]),np.sum(tries[:,3])/N))
print("# of total wins if switches: {:} [prob: {:.5f}]"\
      .format(np.sum(tries[:,4]),np.sum(tries[:,4])/N))
```

# 4 doors, 2 are opened simultaneously

```{code-cell} ipython3
tries4 = np.random.randint(1,5,(N,6))
tries4[:,2:] = 0

# 0th col: Door with the prize
# 1st col: Door that the contestant has chosen
# 2nd col: Door that Monty opens
# 3rd col: Door that Monty opens
# 4th col: 1 if the user wins by sticking, 0 otherwise
# 5th col: 1 if the user wins by switching, 0 otherwise
```

```{code-cell} ipython3
# tries4
```

```{code-cell} ipython3
filt4 = tries4[:,0] == tries4[:,1]

tries4[filt4,4] = 1
tries4[np.invert(filt4),5] = 1
```

```{code-cell} ipython3
a = np.array([1,2,3,4])
for i in range(N):
    b = a[a != tries4[i,0]]
    b = b[b != tries4[i,1]]
    b = np.sort(np.random.choice(b,size=2,replace = False))
    #print(b);
    tries4[i,2:4] = b
```

```{code-cell} ipython3
tries4
```

```{code-cell} ipython3
print("# of total wins if sticks  : {:} [prob: {:.5f}]"\
      .format(np.sum(tries4[:,4]),np.sum(tries4[:,4])/N))
print("# of total wins if switches: {:} [prob: {:.5f}]"\
      .format(np.sum(tries4[:,5]),np.sum(tries4[:,5])/N))
```

# 4 doors, a door is opened at each stage

```{code-cell} ipython3
N=10000
tries41 = np.random.randint(1,5,(N,12))
tries41[:,2:] = 0

# 0th  col: Door with the prize
# 1st  col: Door that the contestant has chosen first (stick - *)
# 2nd  col: Door that Monty opens first
# 3rd  col: Door that the contestant switches on the first ask (switch - *)
# 4th  col: Door that Monty opens second (given the contestant has stuck on the 1st)
# 5th  col: Door that Monty opens second (given the contestant has switched on the 1st)
# 6th  col: Door that the contestant switches on the second ask (stick - switch)
# 7th  col: Door that the contestant switches on the second ask (switch - switch)
# 8th  col: 1 if the user has stick-stick win
# 9th  col: 1 if the user has stick-switch win
# 10th col: 1 if the user has switch-stick win
# 11th col: 1 if the user has switch-switch win

tries41[:10]
```

```{code-cell} ipython3
a = np.array([1,2,3,4])

for i in range(N):
    
    # The 1st door Monty opens
    b = a[a != tries41[i,0]] # Can't open the one with the prize
    b = b[b != tries41[i,1]] # Can't open the contestant's pick
    tries41[i,2] = np.random.choice(b)
    
    # The door the contestant switches (1st)
    b = a[a != tries41[i,1]] # Can't switch to the already picked
    b = b[b != tries41[i,2]] # Can't switch to the opened door
    tries41[i,3] = np.random.choice(b)

    # The 2nd door Monty opens (given the contestans has stuck)
    b = a[a != tries41[i,0]] # Can't open the one with the prize
    b = b[b != tries41[i,1]] # Can't open the contestant's pick
    b = b[b != tries41[i,2]] # Can't open the already opened
    tries41[i,4] = np.random.choice(b)    
    
    # The 2nd door Monty opens (given the contestans has switched)
    b = a[a != tries41[i,0]] # Can't open the one with the prize
    b = b[b != tries41[i,3]] # Can't open the contestant's pick
    b = b[b != tries41[i,2]] # Can't open the already opened
    tries41[i,5] = np.random.choice(b)    
    
    # The door the contestant switches (after having stuck)
    b = a[a != tries41[i,2]] # Can't switch to the opened door
    b = b[b != tries41[i,4]] # Can't switch to the opened door
    b = b[b != tries41[i,1]] # Can't switch to the already picked
    tries41[i,6] = np.random.choice(b)
    
    # The door the contestant switches (after having switched once)
    b = a[a != tries41[i,2]] # Can't switch to the opened door
    b = b[b != tries41[i,5]] # Can't switch to the opened door
    b = b[b != tries41[i,3]] # Can't switch to the already picked
    tries41[i,7] = np.random.choice(b)

tries41[:10]
```

```{code-cell} ipython3
# stick - stick wins
tries41[tries41[:,0] == tries41[:,1],8] = 1

# stick - switch wins
tries41[tries41[:,0] == tries41[:,6],9] = 1

# switch - stick wins
# If they first switch, then stick, 
# treat the 3rd column as the final choice
tries41[tries41[:,0] == tries41[:,3],10] = 1

# switch - switch wins
tries41[tries41[:,0] == tries41[:,7],11] = 1

tries41[:10]
```

```{code-cell} ipython3
print("# of total wins for stick-stick   : {:} [prob: {:.5f}]"\
      .format(np.sum(tries41[:,8]),np.sum(tries41[:,8])/N))
print("# of total wins for stick-switch  : {:} [prob: {:.5f}]"\
      .format(np.sum(tries41[:,9]),np.sum(tries41[:,9])/N))
print("# of total wins for switch-stick  : {:} [prob: {:.5f}]"\
      .format(np.sum(tries41[:,10]),np.sum(tries41[:,10])/N))
print("# of total wins for switch-switch : {:} [prob: {:.5f}]"\
      .format(np.sum(tries41[:,11]),np.sum(tries41[:,11])/N))
```

```{code-cell} ipython3
tries41[:10]
```

```{code-cell} ipython3
ss = tries41[:,[0,1,6,3,7]] 
ss[:10] # pr, st-st, st-sw, sw-st, sw-sw
```

```{code-cell} ipython3

```
