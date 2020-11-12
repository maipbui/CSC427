import numpy as np
import math
from random import random


def activation_func_perceptron(row, weights):
    """
    Determine the output of the neural network using Rosenblatt Perceptron
    :param row: row of the dataset
    :param weights: weights of the perceptron
    :return: -1 or 1 value
    """
    activation = weights[0]
    for i in range(len(row) - 1):
        activation += weights[i+1] * row[i]
    return 1.0 if activation >= 0.0 else -1.0


def activation_func_ls(x, w):
    """
    Determine the output of the neural network using Least Square
    :param row: row of the dataset
    :param weights: weights of the Least Square algorithm
    :return: -1 or 1 value
    """
    activation = 0
    for i in range(3):
        activation = activation + x[i] * w[i]
    return 1.0 if activation >= 0.0 else -1.0


def moon(num_points, distance, radius, width):
    """
    Generate a double-moon classification problem
    :param num_points: total number of points in each moon
    :param distance: the distance between 2 moons
    :param radius: the radius of each moon
    :param width: the width of each moon
    :return: dataset
    """
    points = num_points

    x1 = [0 for _ in range(points)]
    y1 = [0 for _ in range(points)]
    x2 = [0 for _ in range(points)]
    y2 = [0 for _ in range(points)]

    for i in range(points):
        d = distance
        r = radius
        w = width
        a = random() * math.pi
        x1[i] = math.sqrt(random()) * math.cos(a) * (w / 2) + (
                    (-(r + w / 2) if (random() < 0.5) else (r + w / 2)) * math.cos(a))
        y1[i] = math.sqrt(random()) * math.sin(a) * w + (r * math.sin(a)) + d

        a = random() * math.pi + math.pi
        x2[i] = (r + w / 2) + math.sqrt(random()) * math.cos(a) * (w / 2) + (
            (-(r + w / 2)) if (random() < 0.5) else (r + w / 2)) * math.cos(a)
        y2[i] = -(math.sqrt(random()) * math.sin(a) * (-w) + (-r * math.sin(a))) - d
    return [x1, x2, y1, y2]


def normalize(dataset):
    """
    :param dataset: the original dataset
    :return: the processed dataset by normalization
    """
    norm_data = np.asarray(dataset)
    sum_column = np.sum(norm_data[:, :2], axis=0)
    mean_column = np.divide(sum_column, len(dataset))
    norm_data[:, 0] = np.subtract(norm_data[:, 0], mean_column[0])
    norm_data[:, 1] = np.subtract(norm_data[:, 1], mean_column[1])
    max_value = np.amax(abs(norm_data[:, :2]))
    norm_data[:, 0] = np.divide(norm_data[:, 0], max_value)
    norm_data[:, 1] = np.divide(norm_data[:, 1], max_value)
    return norm_data


def train_perceptron(dataset, epochs, learning_rate):
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
            prediction = activation_func_perceptron(row, weights)
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


def train_least_square(dataset, label_set, learning_rate):
    """
    :param dataset: the dataset we use to train using Least Square
    :param label_set: the label of the dataset
    :param learning_rate: The learning rate of the neural network
    :return: mean square error values and weights values of the neural network using Least Square
    """
    I = np.identity(3)
    transposed_dataset = dataset.transpose()
    R = transposed_dataset.dot(dataset)
    R = np.reshape(R, (3, 3))
    weights = (np.linalg.inv(R + (learning_rate * I))).dot((transposed_dataset.dot(label_set)))

    MSE = 0
    for i in range(len(dataset)):
        error = abs(activation_func_ls(dataset[i], weights) - label_set[i])
        MSE += error ** 2
    return MSE / len(dataset), weights


def train_lms(dataset, epochs, eta):
    """
    Train the LMS algorithm using double moon dataset
    :param dataset: the dataset we use to train the LMS algorithm
    :param epochs: number of epochs
    :param eta: learning rate
    :return: the lists of values of mse and weights
    """
    weights = np.random.rand(2)/2 - 0.25
    mses = []
    for epoch in range(epochs):
        mse = 0.0
        np.random.shuffle(dataset)
        for row in dataset:
            row_no_label = row[:2]
            row_no_label = np.asarray(row_no_label)
            prediction = np.dot(weights, row_no_label)
            expected = row[-1]
            error = expected - prediction
            mse += error ** 2
            weights = weights + eta*error*row_no_label

        mse /= len(dataset)
        mses.append(mse)

        if mse == 0:
            break
    return mses, weights


if __name__ == "__main__":
    num_points = int(input("enter number of points in each moon: "))
    dist = float(input("enter the distance between 2 moons: "))
    lr = float(input("enter the learning rate: "))
    num_epochs = int(input("enter the number of epochs: "))

    x1_value, x2_value, y1_value, y2_value = moon(num_points, dist, 10, 6)
    data = []
    data.extend([x1_value[i], y1_value[i], -1] for i in range(num_points))
    data.extend([x2_value[i], y2_value[i], 1] for i in range(num_points))
    data = np.asarray(data)
    data_lms = normalize(np.copy(data))

    ones = np.ones(num_points)
    minus_ones = np.ones(num_points) * -1
    label_set = np.concatenate((ones, minus_ones)).reshape((num_points * 2, 1))
    x = x1_value + x2_value
    x = np.asarray(x).reshape(num_points * 2, 1)
    y = y1_value + y2_value
    y = np.asarray(y).reshape(num_points * 2, 1)
    x0 = np.ones((num_points * 2, 1))
    data_least_square = np.concatenate([x0, x, y], axis=1)

    mse_perceptron, weights_perceptron = train_perceptron(data, num_epochs, lr)
    mse_least_square, weights_least_square = train_least_square(data_least_square, label_set, lr)
    mse_lms, weights_lms = train_lms(data_lms, num_epochs, lr)

    mse_perceptron = np.mean(mse_perceptron)
    mse_lms = np.mean(mse_lms)
    print('--------------------------------------')
    print('Rosenblatt Perceptron algorithm')
    print('MSE: ', mse_perceptron)
    print('Least Square algorithm')
    print('MSE: ', mse_least_square)
    print('LMS algorithm')
    print('MSE: ', mse_lms)
    print('--------------------------------------')
