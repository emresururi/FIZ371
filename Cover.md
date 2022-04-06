---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.5
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# FIZ371 - Scientific and Technical Calculations
**Dr. Emre S. Tasci** <emre.tasci@hacettepe.edu.tr>  
Department of Engineering Physics  
Hacettepe University  
Ankara / Turkey

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
```

```{code-cell} ipython3
x = np.linspace(-3,3,100)
y = x**2
```

```{code-cell} ipython3
print(np.array([[x],[y]]).T[:10,:])
```

```{code-cell} ipython3
plt.plot(x**2,y,"-")
plt.show()
```

```{code-cell} ipython3

```
