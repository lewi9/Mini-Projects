from utils import *
from nn import *
import imageio
import os
import seaborn as sns

ROWS = 10
COLS = 15

SEE_ROWS = 3
SEE_COLS = 3

EPOCH = 40
POPULATION = 20
SIMULATIONS = 10

W = 0.5
C1 = 0.5
C2 = 1

TIME = 300

LAYERS = 5
# UNITS = (100, 90, 80, 70, 60, 50, 40, 30, 20, 10)
UNITS = (12, 12, 12, 6, 3)
FUNCTIONS = [
    lambda x: x,
    lambda x: (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x)),
    lambda x: (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x)),
    lambda x: (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x)),
    lambda x: (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x)),
]

# SHAPE = ((100, ROWS * COLS), (90, 100), (80, 90), (70, 80), (60, 70), (50, 60), (40, 50), (30, 40), (20, 30), (10, 20), (1, 10))
SHAPE = ((12, SEE_ROWS*SEE_COLS), (12, 12), (12, 12), (6, 12), (3, 6), (1, 3))

pso = PSO(W, C1, C2)

networks = [
    (NN(LAYERS, UNITS, FUNCTIONS, pso.initial_weights(SHAPE), pso.initial_weights(SHAPE)))
    for i in range(POPULATION)
]

best_all_nn = 0
max_fitness = 0
all_fitness = np.array([])
maxes_fitness = []

for i in range(EPOCH):
    fitness = np.zeros(POPULATION)
    for j in range(POPULATION):
        local_fitness = 0
        for s in range(SIMULATIONS):
            map = init_map(ROWS, COLS)
            for k in range(TIME):
                segment = cut_map(map, SEE_ROWS, SEE_COLS)
                if networks[j].calc(segment):
                    map = move(map)
                    if map.size == 0:
                        break
                    elif map.size == 1:
                        break
                    else:
                        local_fitness += 1
                map = roll_map(map)
                local_fitness -= 0.05
        fitness[j] = local_fitness/SIMULATIONS
    index = np.argmax(fitness)
    print(np.max(fitness))
    if np.max(fitness) > max_fitness:
        max_fitness = np.max(fitness)
        best_all_nn = networks[index]
    best_nn = networks[index]
    for j in range(POPULATION):
        networks[j] = pso.optimize(networks[j], fitness[j], best_all_nn)
    try:
        all_fitness = np.vstack((all_fitness, fitness))
    except:
        all_fitness = fitness

    maxes_fitness.append(max_fitness)


for i in range(TIME):
    try:
        os.remove(f"./frames/frame_{i}.png")
    except:
        break

map = init_map(ROWS, COLS)
for i in range(TIME):
    create_frame(map, i)
    segment = cut_map(map, SEE_ROWS, SEE_COLS)
    if best_all_nn.calc(segment):
        map = move(map)
        if map.size == 0:
            break
        if map.size == 1:
            break
    map = roll_map(map)

frames = []
for i in range(TIME):
    try:
        image = imageio.v2.imread(f'./frames/frame_{i}.png')
    except:
        break
    frames.append(image)

imageio.mimsave('./result.gif', frames, fps=2, loop=1)

x = np.arange(0, EPOCH, 1)
x_for_boxplot = np.repeat(x, POPULATION)
y_for_boxplot = all_fitness.flatten()

fig, ax = plt.subplots(1, 1, figsize=(20, 20))
ax.plot(x_for_boxplot, np.repeat(maxes_fitness, POPULATION), c='red')
sns.boxplot(x=x_for_boxplot, y=y_for_boxplot, ax=ax)

ax.set_ylim([-5, 10])
ax.set_xlabel("Epoch")
ax.set_ylabel("Fitness function")
ax.set_title("Learning process")
plt.savefig("./chart.png")
plt.close()