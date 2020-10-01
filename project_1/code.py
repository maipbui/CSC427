__author__ = 'Mai Bui'
__version__ = '09/30/2020'

from random import random
import matplotlib.pyplot as plt
import math
import numpy as np


def signum_func(row, weights):
    """
    Determine the output of the neural network
    :param row: row of the dataset
    :param weights: weights of the perceptron
    :return: -1 or 1 value
    """
    activation = weights[0]
    for i in range(len(row) - 1):
        activation += weights[i+1] * row[i]
    return 1.0 if activation >= 0.0 else -1.0


def moon(num_points, distance, radius, width):
    """
    Generate a double-moon classification problem.
    :param num_points: total number of points in each moon.
    :param distance: the distance between 2 moons.
    :param radius: the radius of each moon.
    :param width: the width of each moon.
    :return: dataset
    """
    x1 = [0 for _ in range(num_points)]
    y1 = [0 for _ in range(num_points)]
    x2 = [0 for _ in range(num_points)]
    y2 = [0 for _ in range(num_points)]

    for i in range(num_points):
        d = distance
        r = radius
        w = width
        a = random() * math.pi
        x1[i] = math.sqrt(random()) * math.cos(a) * (w / 2) + \
            ((-(r + w / 2) if (random() < 0.5) else (r + w / 2)) * math.cos(a))
        y1[i] = math.sqrt(random()) * math.sin(a) * w + (r * math.sin(a)) + d

        a = random() * math.pi + math.pi
        x2[i] = (r + w / 2) + math.sqrt(random()) * math.cos(a) * (w / 2) + \
            ((-(r + w / 2)) if (random() < 0.5) else (r + w / 2)) * math.cos(a)
        y2[i] = -(math.sqrt(random()) * math.sin(a) * (-w) + (-r * math.sin(a))) - d
    return [x1, x2, y1, y2]


def train(dataset, epochs, learning_rate):
    """
    :param dataset: the dataset we use to train perceptron
    :param epochs: number of epochs
    :param learning_rate: The learning rate of the neural network
    :return: mean square error values and weights values of the neural network
    """

    weights = [0 for _ in range(3)]
    mse_values = []

    for epoch in range(epochs):
        mse = 0.0
        for row in dataset:
            prediction = signum_func(row, weights)
            expected = row[-1]
            error = expected - prediction
            mse += error ** 2
            weights[0] += learning_rate * error
            for i in range(len(row) - 1):
                weights[i+1] += learning_rate * error * row[i]
        mse /= len(dataset)
        mse_values.append(mse)

        if mse == 0:
            break
    return mse_values, weights


def plot(weights, mse_values, dataset):
    """
    :param mse_values: list of all mean square error values
    :param weights: weights values of the neural network
    :param dataset: The dataset
    :return: None, display the plot of learning curve and result.
    """
    class_1 = dataset[:, 0] * weights[1] + dataset[:, 1] * weights[2] >= -weights[0]
    class_1_dataset = dataset[class_1]
    class_2 = dataset[:, 0] * weights[1] + dataset[:, 1] * weights[2] < -weights[0]
    class_2_dataset = dataset[class_2]

    n_epoch = 10 if len(mse_values) <= 10 else len(mse_values)
    plt.plot(range(1, len(mse_values)+1), mse_values)
    plt.xlabel("Epochs")
    plt.ylabel("MSE")
    plt.title('Learning curve')
    plt.axis([1, n_epoch, 0, max(mse_values)])
    plt.show()

    x = np.asarray([-20, 32])
    y = (-weights[0] - weights[1] * x)/weights[2]
    plt.plot(x, y, c="k")
    plt.xlim(-20, 32)
    plt.title('Testing result')
    plt.scatter(class_1_dataset[:, 0], class_1_dataset[:, 1], c="b", marker='x', s=50)
    plt.scatter(class_2_dataset[:, 0], class_2_dataset[:, 1], c="r", marker='x', s=50)
    plt.show()


if __name__ == "__main__":
    total_points = int(input("enter number of points in each moon: "))
    dist = float(input("enter the distance between 2 moons: "))
    lr = float(input("enter the learning rate: "))
    num_epochs = int(input("enter the number of epochs: "))

    x1_value, x2_value, y1_value, y2_value = moon(total_points, dist, 10, 6)
    data = []
    data.extend([x1_value[i], y1_value[i], -1] for i in range(total_points))
    data.extend([x2_value[i], y2_value[i], 1] for i in range(total_points))
    data = np.asarray(data)
    np.random.shuffle(data)

    mse_values, weights_value = train(data, num_epochs, lr)

    plot(weights_value, mse_values, data)
