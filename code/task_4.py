from random import random
import matplotlib.pyplot as plt
import math
import numpy as np


def activation_func_perceptron(row, weights):
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


def plot(mse_d1, mse_d0, mse_dm4):
    mse_min = min(np.min(mse_d1), np.min(mse_d0), np.min(mse_dm4))
    mse_max = max(np.max(mse_d1), np.max(mse_d0), np.max(mse_dm4))
    plt.plot(range(1, len(mse_d1)+1), mse_d1, label='d=1', c='r')
    plt.plot(range(1, len(mse_d0) + 1), mse_d0, label='d=0', c='b')
    plt.plot(range(1, len(mse_dm4) + 1), mse_dm4, label='d=-4', c='c')
    plt.legend(loc="upper right")
    plt.xlabel("Epochs")
    plt.ylabel("MSE")
    plt.title('Learning curve')
    plt.axis([1, 50, 0, max(max(mse_d1), max(mse_d0), max(mse_dm4))])
    plt.ylim(mse_min*0.8, mse_max)
    plt.show()


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


if __name__ == "__main__":
    num_points = 1000
    num_epochs = 50
    width = 10
    radius = 6
    lr = 0.1

    x1_d1, x2_d1, y1_d1, y2_d1 = moon(num_points, 1, width, radius)
    data_d1 = []
    data_d1.extend([x1_d1[i], y1_d1[i], -1] for i in range(num_points))
    data_d1.extend([x2_d1[i], y2_d1[i], 1] for i in range(num_points))
    data_d1 = np.asarray(data_d1)
    normalize_data_d1 = normalize(np.copy(data_d1))

    x1_d0, x2_d0, y1_d0, y2_d0 = moon(num_points, 0, width, radius)
    data_d0 = []
    data_d0.extend([x1_d0[i], y1_d0[i], -1] for i in range(num_points))
    data_d0.extend([x2_d0[i], y2_d0[i], 1] for i in range(num_points))
    data_d0 = np.asarray(data_d0)
    normalize_data_d0 = normalize(np.copy(data_d0))

    x1_dm4, x2_dm4, y1_dm4, y2_dm4 = moon(num_points, -4, width, radius)
    data_dm4 = []
    data_dm4.extend([x1_dm4[i], y1_dm4[i], -1] for i in range(num_points))
    data_dm4.extend([x2_dm4[i], y2_dm4[i], 1] for i in range(num_points))
    data_dm4 = np.asarray(data_dm4)
    normalize_data_dm4 = normalize(np.copy(data_dm4))

    mse_lms_d1, weights_lms_d1 = train_lms(normalize_data_d1, num_epochs, lr)
    mse_lms_d0, weights_lms_d0 = train_lms(normalize_data_d0, num_epochs, lr)
    mse_lms_dm4, weights_lms_dm4 = train_lms(normalize_data_dm4, num_epochs, lr)
    plot(mse_lms_d1, mse_lms_d0, mse_lms_dm4)

    mse_perceptron_d1, weights_perceptron_d1 = train_perceptron(data_d1, num_epochs, lr)
    mse_perceptron_d0, weights_perceptron_d0 = train_perceptron(data_d0, num_epochs, lr)
    mse_perceptron_dm4, weights_perceptron_dm4 = train_perceptron(data_dm4, num_epochs, lr)

    mse_lms_d1 = np.mean(mse_lms_d1)
    mse_lms_d0 = np.mean(mse_lms_d0)
    mse_lms_dm4 = np.mean(mse_lms_dm4)
    mse_perceptron_d1 = np.mean(mse_perceptron_d1)
    mse_perceptron_d0 = np.mean(mse_perceptron_d0)
    mse_perceptron_dm4 = np.mean(mse_perceptron_dm4)
    print('--------------------------------------')
    print('Distance d=1')
    print('Rosenblatt Perceptron algorithm')
    print('MSE: ', mse_perceptron_d1)
    print('LMS algorithm')
    print('MSE: ', mse_lms_d1)
    print('--------------------------------------')
    print('Distance d=0')
    print('Rosenblatt Perceptron algorithm')
    print('MSE: ', mse_perceptron_d0)
    print('LMS algorithm')
    print('MSE: ', mse_lms_d0)
    print('--------------------------------------')
    print('Distance d=-4')
    print('Rosenblatt Perceptron algorithm')
    print('MSE: ', mse_perceptron_dm4)
    print('LMS algorithm')
    print('MSE: ', mse_lms_dm4)
    print('--------------------------------------')
