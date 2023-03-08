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

# Implementation of Prisonner's Dilemma with Memory

```{code-cell} ipython3
import numpy as np
import pandas as pd
```

```{code-cell} ipython3
history = np.empty((0,2),int)
```

```{code-cell} ipython3
for i in range(10):
    history = np.vstack((history,np.array([np.round(np.random.rand()),0]).astype(int)))
```

```{code-cell} ipython3
history
```

```{code-cell} ipython3
def pick_fight(pl1,pl2,N=10):
    history = np.empty((0,2),int)
    for i in range(N):
        (a,b) = [pl1(0,history),pl2(1,history)]
        #print(a,b)
        history = np.vstack((history,np.array([a,b])))
    return history

def score_history(history):
    # Returns the mean score
    s00 = np.sum((h[:,0] == 0 ) & (h[:,1] == 0)) # Both silent (2,2)
    s01 = np.sum((h[:,0] == 0 ) & (h[:,1] == 1)) # A silent,  B defects (0,3)
    s10 = np.sum((h[:,0] == 1 ) & (h[:,1] == 0)) # A defects, B silent (3,0)
    s11 = np.sum((h[:,0] == 1 ) & (h[:,1] == 1)) # Both defect (1,1)
    
    return np.array([s00*2+s10*3+s11*1,s00*2+s01*3+s11*1])/history.shape[0]
```

```{code-cell} ipython3
def pl_tit4tat(pl_no=0,history=[]):
    # starts with silence (0)
    # afterwards, returns the last action of the opponent
    opp_no = np.mod(pl_no+1,2)
    if(not history.size):
        return 0
    return history[-1,opp_no]

def pl_rando(pl_no=0,history=[]):
    # returns random
    return np.random.randint(0,2)

def pl_good(pl_no=0,history=[]):
    # always silent
    return 0

def pl_evil(pl_no=0,history=[]):
    # always defects
    return 1

def pl_last10avg(pl_no,history=[]):
    # Starts with silence (0)
    # Returns the average of opponent's last 10 turns
    opp_no = np.mod(pl_no+1,2)
    if(not history.size):
        return 0
    return int(np.round(np.mean(history[-10:,opp_no])))
    
```

```{code-cell} ipython3
pls = ['pl_rando','pl_tit4tat','pl_good','pl_evil','pl_last10avg']
```

```{code-cell} ipython3
chart = pd.DataFrame(columns=pls,index=pls)

no_pl = len(pls)
for i in range(no_pl):
    for j in range(no_pl):
        print("{:s} vs {:s}".format(pls[i],pls[j]))
        p1 = locals()[pls[i]]
        p2 = locals()[pls[j]]
        h = pick_fight(p1,p2,N=10000)
        (a,b) = score_history(h)
        print("{:.2f}, {:.2f}".format(a,b))
        if(a > b):
            print("{:s} (1st player) WINS!!!".format(pls[i]))
        elif(a < b):
            print("{:s} (2nd player) WINS!!!".format(pls[j]))
        else:
            print("IT'S A DRAW!!!")
        print("-"*45)
        
        chart.loc[pls[i]][pls[j]] = (a,b)
print(chart)
for pl in pls:
    # Row-wise        
    chart.loc[pl,'Tot'] = np.sum(chart.loc[pl,pls].sum()[0::2])

for pl in pls:
    # Col-wise
    chart.loc['ToT',pl] = np.sum(chart[pl][:-1].sum()[1::2])
    
totscore = pd.DataFrame(columns=pls)
for pl in pls:
    #print(pl,chart.loc[pl,'Tot'], chart.loc['ToT',pl])
    totscore.loc['tot',pl] = chart.loc[pl,'Tot'] + chart.loc['ToT',pl]
print(totscore)
```

```{code-cell} ipython3
chart
```

```{code-cell} ipython3
totscore
```

```{code-cell} ipython3
chart.loc['pl_evil',pls] # row-wise
```

```{code-cell} ipython3
chart.loc['pl_evil',pls].sum()[0::2]
```

```{code-cell} ipython3
chart.loc['pl_evil','tot'] = np.sum(chart.loc['pl_evil',pls].sum()[0::2])
```

```{code-cell} ipython3
chart['pl_evil'] # col-wise
```

```{code-cell} ipython3
chart['pl_evil'].sum()[1::2]
```

```{code-cell} ipython3
chart.loc['ToT','pl_evil'] = np.sum(chart['pl_evil'].sum()[1::2])
```

```{code-cell} ipython3
chart.loc['pl_evil','pl_good']
```

```{code-cell} ipython3
chart.loc['pl_evil','tot'] = 5
```

```{code-cell} ipython3
chart
```

```{code-cell} ipython3
chart2 = chart.copy()
```

```{code-cell} ipython3
chart2
```

```{code-cell} ipython3
chart2.loc['ToT'] = 0
```

```{code-cell} ipython3
chart2
```

```{code-cell} ipython3
chart2.loc['ToT','pl_tit4tat'] = 5
```

```{code-cell} ipython3
chart2
```

```{code-cell} ipython3

```
