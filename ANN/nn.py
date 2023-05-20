import numpy as np


class Neuron:
    def __init__(self, x, v, function):
        self.x = x
        self.v = v
        self.best_x = x
        self.function = function
        self.fitness = 0

    def calc(self, input):
        return self.function(np.sum(input * self.x))

    def set_fitness(self, fitness):
        self.fitness = fitness

    def set_best_x(self, x):
        self.best_x = x


class NN:
    def __init__(self, layers, units, functions, x, v):
        self.layers = []
        for i in range(layers):
            self.layers.append([Neuron(x[i][j], v[i][j], functions[i]) for j in range(units[i])])
        self.layers.append([Neuron(x[-1][0], v[-1][0], lambda elem: elem > 0)])
        self.fitness = 0

    def calc(self, map):
        input = map.flatten()
        for i in range(len(self.layers)):
            result = []
            for j in range(len(self.layers[i])):
                result.append(self.layers[i][j].calc(input))
            input = result
        output = input[0]
        return output

    def set_fitness(self, fitness):
        self.fitness = fitness
        for i in range(len(self.layers)):
            for j in range(len(self.layers[i])):
                self.layers[i][j].set_fitness(fitness)

    def set_best_x(self, best_nn):
        for i in range(len(self.layers)):
            for j in range(len(self.layers[i])):
                self.layers[i][j].set_best_x(best_nn.layers[i][j].x)


class PSO:
    def __init__(self, w, c1, c2):
        self.w = w
        self.c1 = c1
        self.c2 = c2

    def optimize(self, current_nn, fitness, best_nn):
        if current_nn.fitness < fitness:
            current_nn.set_fitness(fitness)
            current_nn.set_best_x(best_nn)

        if current_nn is best_nn:
            return current_nn

        for i in range(len(current_nn.layers)):
            for j in range(len(best_nn.layers[i])):
                current_nn.layers[i][j].v = (self.w * current_nn.layers[i][j].v
                                             + self.c1 * np.random.rand() * (current_nn.layers[i][j].best_x
                                                                             - current_nn.layers[i][j].x)
                                             + self.c2 * np.random.rand() * (best_nn.layers[i][j].best_x
                                                                             - current_nn.layers[i][j].x))
                current_nn.layers[i][j].x += current_nn.layers[i][j].v

        return current_nn

    def initial_weights(self, shape):
        weights = []
        for elem in shape:
            weights.append(np.random.rand(*elem)*2-1)
        return weights
