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

# Probabilities Application: Letter Frequencies
**FIZ371 - Scientific & Technical Calculations | 11/10/2023**

Emre S. Tasci <emre.tasci@hacettepe.edu.tr>

Using L. Frank Baum's "The Wonderful Wizard of Oz" book, calculate the frequencies of the letters & bigrams.

(The book, written in 1900, is now in public domain and available from [Project Gutenberg](https://www.gutenberg.org/ebooks/55))

```{code-cell} ipython3
import numpy as np
```

We first read the text into the `data` variable and define the set of letters we are interested in:

```{code-cell} ipython3
fname = "supp/wizardofoz_1990_publicdomain_guthenberg.txt"
alphabet_str = "abcdefghijklmnopqrstuvwxyz "

# Convert the alphabet_str to alphabet array
alphabet = [*alphabet_str]
```

```{code-cell} ipython3
with open(fname, 'r') as myfile:
    data=myfile.read().replace('\n', ' ')
    
# Convert it to all lowercase
data = data.lower()
```

# Letters

Count and store the frequencies into the `count_letter` dictionary:

```{code-cell} ipython3
count_letter = {}
for letter in alphabet:
    count_letter[letter] = data.count(letter)
count_letter
```

To calculate the probabilities, we divide each frequency by the total sum:

```{code-cell} ipython3
tot_count = np.sum(list(count_letter.values()))
print(tot_count)
```

```{code-cell} ipython3
probs_letter = {}
for letter in count_letter.keys():
    probs_letter[letter] = count_letter[letter] / tot_count
probs_letter
```

# Bigrams
We are going to generate all possible bigrams and count them, storing in `count_bigram`:

```{code-cell} ipython3
count_bigram = {}
total_sum_bigram = 0
for x in alphabet:
    for y in alphabet:
        bigram = x+y
        count_bigram[bigram] = data.count(bigram)
        total_sum_bigram += count_bigram[bigram]
```

```{code-cell} ipython3
count_bigram
```

# Exercise

Calculate the probabilities:

$$p("b") = ?$$

$$p("an") = ?$$

```{code-cell} ipython3
p_b = probs_letter["b"]
p_b
```

```{code-cell} ipython3
p_an = count_bigram["an"] / total_sum_bigram
p_an
```

# Exercise

Calculate the probabilities:

$$p(x="a" | y ="n") =?$$

$$p(y="n"|x="a") =?$$

+++

## $p(x="a"|y="n")$

To calculate the first probability, we need to define a subset that contains all the bigrams whose second letter is equal to "n":

```{code-cell} ipython3
subset_x_yn = {}
for x in alphabet:
    bigram = x + "n"
    subset_x_yn[bigram] = count_bigram[bigram]
subset_x_yn
```

The total of the subset, `sum_subset_x_yn` is:

```{code-cell} ipython3
subset_x_yn.values()
```

```{code-cell} ipython3
# We are converting the values to a numpy array
# to directly evaluate using the 'sum()' method:
array_1 = np.array(list(subset_x_yn.values()))
array_1
```

```{code-cell} ipython3
total_subset_x_yn = array_1.sum()
total_subset_x_yn
```

Thus, $p(x="a"|y="n")$ probability is:

$$p(x="a"|y="n") = \frac{3170}{10715}\approx0.296$$

```{code-cell} ipython3
prob_xa_yn = subset_x_yn["an"] / total_subset_x_yn
prob_xa_yn
```

## $p(y="n"|x="a")$

For the second probability, we construct a new subset that contains the bigrams whose first letter is "a":

```{code-cell} ipython3
subset_y_xa = {}
for y in alphabet:
    bigram = "a" + y
    subset_y_xa[bigram] = count_bigram[bigram]
subset_y_xa
```

The rest is similar to the first one:

```{code-cell} ipython3
array_2 = np.array(list(subset_y_xa.values()))
array_2
```

```{code-cell} ipython3
total_subset_y_xa = array_2.sum()
total_subset_y_xa
```

$$p(y="n"|x="a") = \frac{3172}{13863}\approx0.229$$

```{code-cell} ipython3
prob_yn_xa = subset_y_xa["an"] / total_subset_y_xa
prob_yn_xa
```

# Exercise

Calculate the probability:

$$\frac{p(y="n"|x="a")p("a")}{p("n")}$$

For this one, we have all the factors:

```{code-cell} ipython3
prob_yn_xa * probs_letter["a"] / probs_letter["n"]
```

Let's check if this is equal to $p(x="a"|y="n")$ as Bayes Theorem dictates:

```{code-cell} ipython3
prob_xa_yn
```

The two values are close but not equal. I hope that you are able to figure out the reason for this difference. If not, please ponder on it a while before proceeding! 8)
```

















```
<center><i><b>Wait while the reader ponders!<br>(Spoilers ahead!)</center></b></i>

```















```

+++

The reason is due to the characters unaccounted for (e.g., "d."). Consider the following paragraph:

> "My darling child!" she cried, folding the little girl in her arms and covering her face with kisses. "Where in the world did you come from?"

If we were to calculate the marginal probability of $p("n")$, for example, we would first count all the occurences of the letter "n" and then divide by all the characters included in the text:

```{code-cell} ipython3
text = '''"My darling child!" she cried, folding the little girl in her arms and
covering her face with kisses. "Where in the world did you come from?"'''

text = text.lower()
print(text)
```

```{code-cell} ipython3
count_n = text.count("n")
count_n
```

```{code-cell} ipython3
count_all = len(text)
count_all
```

```{code-cell} ipython3
p_n = count_n / count_all
p_n
```

However, this marginal probability includes the uncounted characters such as {"'",""",",",".","!"} which aren't accounted for when we were calculating the bigram probabilities, thus messing with our calculations. This also goes for the bigram probabilities, thus causing an inconsistency.

To remedy this issue, we should have excluded all the characters except the ones we had in our `alphabet` before we had started. This filtering can be done via the regular expression module _re_'s `sub()` method:

```{code-cell} ipython3
import re
```

```{code-cell} ipython3
text_filtered = re.sub('[^a-z ]',' ',text)
print(text_filtered)
```

_Although the regular expressions is a whole topic by itself, to briefly explain, we are first defining a range inside the square brackets to include any letter from a to z ([a-z]), and then also add the space character to this range ([a-z ]). But since these are the characters we want to keep, we negate our statement by putting the negation sign ("^") to mean "every character that is not in this range" ([^a-z ]). The second parameter is the replacement, by putting in " ", we are saying that replace all the matching characters with space. The third parameter is the text we want to operate on._

Thus, we end up with a text that only contains the characters we are taking into account.

Below, we'll repeat the same procedures we did above, only this time we'll filter our text to only include the letters from a to z and the space character:

```{code-cell} ipython3
with open(fname, 'r') as myfile:
    data=myfile.read().replace('\n', ' ')

# Convert it to all lowercase
data = data.lower()

data = re.sub('[^a-z ]',' ',data)

count_letter = {}
for letter in alphabet:
    count_letter[letter] = data.count(letter)
    
tot_count = np.sum(list(count_letter.values()))

probs_letter = {}
for letter in count_letter.keys():
    probs_letter[letter] = count_letter[letter] / tot_count

count_bigram = {}
total_sum_bigram = 0
for x in alphabet:
    for y in alphabet:
        bigram = x+y
        count_bigram[bigram] = data.count(bigram)
        total_sum_bigram += count_bigram[bigram]

p_b = probs_letter["b"]

p_an = count_bigram["an"] / total_sum_bigram

subset_x_yn = {}
for x in alphabet:
    bigram = x + "n"
    subset_x_yn[bigram] = count_bigram[bigram]
    
array_1 = np.array(list(subset_x_yn.values()))

total_subset_x_yn = array_1.sum()

prob_xa_yn = subset_x_yn["an"] / total_subset_x_yn

subset_y_xa = {}
for y in alphabet:
    bigram = "a" + y
    subset_y_xa[bigram] = count_bigram[bigram]

array_2 = np.array(list(subset_y_xa.values()))

total_subset_y_xa = array_2.sum()

prob_yn_xa = subset_y_xa["an"] / total_subset_y_xa

prob_yn_xa * probs_letter["a"] / probs_letter["n"]
```

```{code-cell} ipython3
prob_xa_yn
```

Thus, we have saved the Bayes Theorem! 8)

$$p(a|b) = \frac{p(b|a)p(a)}{p(b)}$$

+++

# Visualizing the frequencies

Using the pandas module, we can also "visualize" the frequencies:

```{code-cell} ipython3
import pandas as pd
```

```{code-cell} ipython3
df_letter = pd.DataFrame(count_letter.values(),
                         index=alphabet,
                         columns=["count"])
df_letter.style.background_gradient(cmap="binary")

#https://matplotlib.org/stable/tutorials/colors/colormaps.html
# https://stackoverflow.com/a/50605020
```

```{code-cell} ipython3
df = pd.DataFrame({"l1":[],"l2":[],"count":[]})
for letter1 in alphabet:
    for letter2 in alphabet:
        word = letter1+letter2
        df.loc[-1] = [letter1,letter2,data.count(word)]
        df.index = df.index + 1
df.insert(loc=2,column="word",value=df["l1"]+df["l2"])
```

```{code-cell} ipython3
df
```

```{code-cell} ipython3
len_alpha = len(alphabet)

mat = np.zeros((len_alpha,len_alpha),int)
for i in range(len_alpha):
    for j in range(len_alpha):
        mat[i,j] = df.loc[df.word==(alphabet[i]+alphabet[j]),"count"]
        
df_bi = pd.DataFrame(mat,index=alphabet,columns=alphabet)

import matplotlib.pyplot as plt
from matplotlib import colors

def background_gradient(s, m, M, cmap='PuBu', low=0, high=0):
    rng = M - m
    norm = colors.Normalize(m - (rng * low),
                            M + (rng * high))
    normed = norm(s.values)
    c = [colors.rgb2hex(x) for x in plt.cm.get_cmap(cmap)(normed)]
    return ['background-color: %s' % color for color in c]


df_bi.style.apply(background_gradient,cmap="binary",
               m=df_bi.min().min(),
               M=df_bi.max().max(),
               low=0,
               high=0.85)

# https://stackoverflow.com/a/42563850
```
