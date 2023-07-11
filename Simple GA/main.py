import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

# Initial parameters
WIDTH = 100
HEIGHT = 100
N_CITIES = 40
N_POPULATION = 100
N_ITERATIONS = 300
P_CROSSOVER = 0.75
P_MUTATION = 0.1

# Variables for plot data
time_series_best_results = []
time_series_results = []
best_solution = None


# Class Solution
class Solution:

    # Constructor - permutation of cities
    def __init__(self, n_cities=0, repr=None):
        if repr is None:
            self.representation = np.random.permutation(n_cities)
        else:
            self.representation = repr

    # Calculate distance of road between cities in order
    def evaluate(self, distance_matrix):
        distance = 0
        for i in range(1, len(self.representation)):
            distance += distance_matrix[self.representation[i - 1]][self.representation[i]]
        distance += distance_matrix[self.representation[-1]][self.representation[0]]
        return distance

    # Define crossover as c = a+b
    def __add__(self, other):
        if not isinstance(other, Solution):
            raise TypeError("You can only perform: Solution + Solution")
        # Create child placeholder
        child = np.ones(len(self.representation), dtype=int) * -1

        # Select a piece of one parent1 that will be copied to child
        a, b = np.random.randint(0, len(self.representation), 2)
        idx1 = min(a, b)
        idx2 = max(a, b)
        for i in range(idx1, idx2 + 1):
            child[i] = self.representation[i]
        idx = 0

        # Fill empty places with genom of parent2 in order
        for i in range(len(self.representation)):
            while child[idx] != -1:
                idx += 1
                if idx >= len(child):
                    return Solution(repr=child)
            if other.representation[i] not in child:
                child[idx] = other.representation[i]
                idx += 1
                if idx >= len(child):
                    break

        return Solution(repr=child)

    # Define mutation as a *= b
    def __imul__(self, other):
        if not isinstance(other, float):
            raise TypeError("You can only perform: Solution *= <Double>")
        child = self.representation
        # Check that is mutation or not
        if np.random.sample() < other:
            idx1, idx2 = np.random.randint(0, len(child), 2)
            # Swap to gens randomly
            child[idx1], child[idx2] = child[idx2], child[idx1]
        return Solution(repr=child)


# Create cities points (X, Y)
def init_points(n_cities, width, height):
    x = np.random.sample(n_cities) * width
    y = np.random.sample(n_cities) * height
    return np.column_stack((x, y))


# Create initial solutions
def init_solutions(n_population, n_cities):
    return [Solution(n_cities) for _ in range(n_population)]


# Calculate distance between cities in order represented by solution
def evaluate_solutions(solutions, distance_matrix):
    return [x.evaluate(distance_matrix) for x in solutions]


# Some genetic stuff - selection, crossover, mutation
def genetic_things(solutions, results, p_crossover, p_mutation, n_population):
    output = []

    # There is implemented simple ranking method
    ranks = np.zeros(len(results))

    # Assign rank from 1 (best) to N_CITIES (worst) based on distance
    # It's like sorting alghoritm
    for i in range(len(results)):
        rank = 1
        for j in range(i):
            if results[i] > results[j]:
                rank += 1
            elif results[i] < results[j]:
                ranks[j] += 1
        ranks[i] = rank

    while len(output) != n_population:
        # Select two parents randomly based on 1/rank probability (normalized)
        solution1, solution2 = np.random.choice(solutions, 2, replace=True,
                                                p=1 / np.array(ranks) / sum(1 / np.array(ranks)))
        # Check if crossover condition is fulfilled
        if np.random.sample() < p_crossover:
            # Crossover
            output.append(solution1 + solution2)
            # Mutate child
            output[-1] *= p_mutation
    return output


# Create initial points
points = init_points(N_CITIES, WIDTH, HEIGHT)

# Compute distance matrix
distance_matrix = sp.spatial.distance_matrix(points, points)

# Create initial solutions
solutions = init_solutions(N_POPULATION, N_CITIES)

# Iterate looking for best solution
for j in range(N_ITERATIONS):

    # Some print stuff
    if not j % 10:
        print(f"iteracja: {j}")

    # Calculate results
    results = evaluate_solutions(solutions, distance_matrix)

    # Update for cost function plot
    if len(time_series_best_results) > 1:
        time_series_best_results.append(min(np.min(results), time_series_best_results[-1]))
    else:
        time_series_best_results.append(np.min(results))

    # Update best solution
    if len(time_series_best_results) == 1 or time_series_best_results[-1] != time_series_best_results[-2]:
        best_solution = solutions[np.argmin(results)]

    # Update for quality of solutions plot
    time_series_results.append(results)

    # Create new solutions
    solutions = genetic_things(solutions, results, P_CROSSOVER, P_MUTATION, N_POPULATION)

# Plot 1
fig1, ax = plt.subplots(1, 1, figsize=(12, 12))
ax.plot(time_series_best_results)
ax.set_xlim([0, N_ITERATIONS])
ax.set_ylim([0, np.sqrt(HEIGHT * HEIGHT + WIDTH * WIDTH) * N_CITIES / 2])
ax.set_title(
    f"Funkcja celu w kolejnych iteracjach, miasta = {N_CITIES}, populacja = {N_POPULATION}, p_cross = {P_CROSSOVER}, "
    f"p_mut = {P_MUTATION}"
)
ax.set_xlabel("Iteracja")
ax.set_ylabel(f"Funkcja celu")
plt.savefig("fcwi.png")
plt.close()

# Plot 2
meanlineprops = dict(linewidth=20, color='orange')
fig2, ax = plt.subplots(1, 1, figsize=(12, 12))
ax.boxplot(time_series_results, meanprops=meanlineprops)
ax.set_xlim([0, N_ITERATIONS])
ax.set_ylim([0, np.sqrt(HEIGHT * HEIGHT + WIDTH * WIDTH) * N_CITIES / 2])
ax.set_title(
    f"Jakość rozwiązań w kolejnych iteracjach, miasta = {N_CITIES}, populacja = {N_POPULATION},"
    f" p_cross = {P_CROSSOVER}, p_mut = {P_MUTATION}"
)
ax.set_xlabel("Iteracja")
ax.set_ylabel("Funkcja celu")
ax.set_xticks(np.arange(0,N_ITERATIONS+N_ITERATIONS/10,step=N_ITERATIONS/10, dtype=int))
ax.set_xticklabels(np.arange(0,N_ITERATIONS+N_ITERATIONS/10, step=N_ITERATIONS/10, dtype=int))
plt.savefig("jrwki.png")
plt.close()

# Plot 3
fig3, ax = plt.subplots(1, 1, figsize=(12, 12))
ax.scatter(points[:, 0], points[:, 1])
ax.set_xlim([0, WIDTH])
ax.set_ylim([0, HEIGHT])
ax.set_title(
    f"Najlepsze rozwiązanie, miasta = {N_CITIES}, populacja = {N_POPULATION}, p_cross = {P_CROSSOVER},"
    f" p_mut = {P_MUTATION}"
)
ax.set_xlabel("x")
ax.set_ylabel("y")
for i in range(1, len(best_solution.representation)):
    x1, y1 = points[best_solution.representation[i - 1]]
    x2, y2 = points[best_solution.representation[i]]
    ax.plot([x1, x2], [y1, y2], c="green")
x1, y1 = points[best_solution.representation[-1]]
x2, y2 = points[best_solution.representation[0]]
ax.plot([x1, x2], [y1, y2], c="green")
plt.savefig("nr.png")
plt.close()
