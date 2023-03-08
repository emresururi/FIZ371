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

* **Ant Colony Optimization**
* Artificial Bee Colony

Dr. Emre S. Tasci <emre.tasci@hacettepe.edu.tr>

+++

# Ant Colony Optimization Application to the Travelling Salesman Problem

+++

Start with $N$ cities, randomly placed on the map:

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(371)
```

```{code-cell} ipython3
map_dimensions = 1000
N = 6

alpha = 1.0
beta = 1.1
Q = 10*N # Total pheromone per path
rho = 0.85 # Pheromone persistance coefficient (1-rho)'

# Place the cities
coords_cities = np.random.randint(0,map_dimensions,size=(N,2))
print(coords_cities)

# randomly initialize pheromones along the city paths
matrix_pher = np.random.rand(N,N)*10
# Symmetrize the matrix so that path A->B is equal to path B->A
matrix_pher = np.tril(matrix_pher) + np.tril(matrix_pher, -1).T
print("Path Pheromones:\n",matrix_pher)

# Calculate the distances
matrix_dist = np.zeros((N,N))
for i in range(N):
    for j in range(i+1,N):
        matrix_dist[i,j] = np.linalg.norm(coords_cities[i,:]-coords_cities[j,:])
# Symmetrize
matrix_dist += np.tril(matrix_dist.T)
with np.printoptions(precision=5):
    print("Distances:\n",matrix_dist)
```

```{code-cell} ipython3
alphabet = np.array(list(map(chr, range(65, 91))))

# Show them on the map
fig,ax = plt.subplots()
plt.plot(coords_cities[:,0],coords_cities[:,1],"ob")
plt.xlim(0,map_dimensions+10)
plt.ylim(0,map_dimensions+10)

# Positions for labels - scaled
x_left = np.round(3*map_dimensions/300)
y_up = np.round(7*map_dimensions/300)

# Label the cities correspondingly
for i in range(N):
    ax.annotate(alphabet[i],(coords_cities[i,0]-x_left,\
                             coords_cities[i,1]+y_up))
plt.show()
```

```{code-cell} ipython3
def send_the_ant():
    global matrix_pher
    # Start from a random city
    city_start = np.random.randint(N)
    #d#print("Starting City: {:} (#{:})\n".format(alphabet[city_start],city_start))
    city_sequence = [city_start]
    city_from = city_start
    # At the beginning we haven't visited any cities yet
    # -- except the city we are starting from
    cities_not_visited = np.arange(N)
    cities_not_visited = np.delete(cities_not_visited,\
                                   np.where(cities_not_visited == city_start))

    while (len(cities_not_visited)):
        # Calculate the possibilities
        possibilities = []
        for city_possible in cities_not_visited:
            pher_this = matrix_pher[city_start,city_possible]
            dist_this = matrix_dist[city_start,city_possible]
            p_this = pher_this**alpha / dist_this**beta
            possibilities.append(p_this)
        possibilities /= sum(possibilities)
        #print(cities_not_visited)
        #print(alphabet[cities_not_visited])
        #d#print("Possible Destinations: {:}".format((", ".join(alphabet[cities_not_visited]))))

        #d#with np.printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)}):
            #d#print("Possibilities: ",possibilities)

        # Choose one city in accordance with the possibilities
        city_chosen = np.random.choice(cities_not_visited,p=possibilities)
        city_to = city_chosen
        city_sequence.append(city_chosen)
        #d#print("Next City Chosen: {:} (#{:})".format(alphabet[city_chosen],city_chosen))
        cities_not_visited = np.delete(cities_not_visited,\
                                   np.where(cities_not_visited == city_chosen))
        #d#print("")
        #d#print(city_from,"->",city_to)

        # Increase pheromone in the visited path
        matrix_pher[city_from,city_to] += Q / matrix_dist[city_from,city_to]
        matrix_pher[city_to,city_from] = matrix_pher[city_from,city_to]
        #print(matrix_pher)
        city_from = city_to
    #d#print("Travel Sequence: {:}".format("->".join(alphabet[city_sequence])))

    # Evaporate pheromone
    matrix_pher *= rho
```

```{code-cell} ipython3
for i in range(100000):
    send_the_ant()
```

```{code-cell} ipython3
with np.printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)}):
    print(matrix_pher)
```

```{code-cell} ipython3
# Calculate most efficient path:
winning_path = ""
winning_dist = 1E10
winning_seq = []
for starting_city in range(N):
    pher_clone = np.copy(matrix_pher)
    total_distance = 0
    print("Starting from: {:} (#{:})".\
          format(alphabet[starting_city],starting_city))
    seq = [starting_city]
    this_city = starting_city
    for i in range(N-1):
        next_city = pher_clone[this_city,:].argmax()
        seq.append(next_city)
        print("\t{:} -> {:} ({:.2f})".format(alphabet[this_city],
                                             alphabet[next_city],
                                matrix_dist[this_city,next_city]))
        total_distance += matrix_dist[this_city,next_city]
        pher_clone[:,this_city] = 0.0
        pher_clone[:,next_city] = 0.0
        this_city = next_city
    this_path = " -> ".join(alphabet[seq])
    if(total_distance < winning_dist):
        winning_dist = total_distance
        winning_path = this_path
        winning_seq = seq
    print(this_path)
    print("Total Distance: {:.2f} units.".format(total_distance))
    print("")
print("Winning Path: {:} ({:.2f} units)".format(winning_path,winning_dist))
```

```{code-cell} ipython3
with np.printoptions(formatter={'float': lambda x: "{0:7.2f}".format(x)}):
    print(matrix_dist)
```

```{code-cell} ipython3
# Show them on the map
fig,ax = plt.subplots()
for i in range(N-1):
    plt.plot([coords_cities[winning_seq[i],0],coords_cities[winning_seq[i+1],0]],\
             [coords_cities[winning_seq[i],1],coords_cities[winning_seq[i+1],1]],\
            "r-")
plt.plot(coords_cities[:,0],coords_cities[:,1],"ob")
plt.xlim(0,map_dimensions+10)
plt.ylim(0,map_dimensions+10)

# Positions for labels - scaled
x_left = np.round(3*map_dimensions/300)
y_up = np.round(7*map_dimensions/300)

# Label the cities correspondingly
for i in range(N):
    ax.annotate(alphabet[i],(coords_cities[i,0]-x_left,\
                             coords_cities[i,1]+y_up))
plt.show()
```

```{code-cell} ipython3
def calc_dist(seq):
    total_distance = 0
    for i in range(len(seq)-1):
        total_distance += matrix_dist[seq[i],seq[i+1]]
    return total_distance
```

```{code-cell} ipython3
calc_dist(winning_seq)
```

```{code-cell} ipython3
calc_dist([1,2,0,3,5,4])
```
