from random import random
import matplotlib.pyplot as plt
import math
import numpy as np


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


def train(dataset, epochs, eta):
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


def plot(weights, mse_values, dataset):
    """
    Plot the desire results
    :param weights: weights values of the neural network
    :param mse_values: list of all mean square error values
    :param dataset: the processed dataset
    :return: None, display the plot of learning curve and result.
    """
    class_1 = dataset[:, 0] * weights[0] + dataset[:, 1] * weights[1] >= 0
    class_1_dataset = dataset[class_1]
    class_2 = dataset[:, 0] * weights[0] + dataset[:, 1] * weights[1] < 0
    class_2_dataset = dataset[class_2]

    n_epoch = 10 if len(mse_values) <= 10 else len(mse_values)

    plt.plot(range(1, len(mse_values)+1), mse_values)
    plt.xlabel("Epochs")
    plt.ylabel("MSE")
    plt.title('Learning curve d = -4')
    plt.axis([1, n_epoch, 0, max(mse_values)])
    plt.ylim(np.min(mse_values), np.max(mse_values))
    plt.show()

    x = np.asarray([-20, 32])
    y = (-weights[0] * x)/weights[1]
    plt.plot(x, y, c="k")
    plt.xlim(-20, 32)
    plt.title('Testing result')
    plt.scatter(class_1_dataset[:, 0], class_1_dataset[:, 1], c="b", marker='x', s=20)
    plt.scatter(class_2_dataset[:, 0], class_2_dataset[:, 1], c="r", marker='x', s=20)
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
    total_points = int(input("enter number of points in each moon: "))
    dist = float(input("enter the distance between 2 moons: "))
    eta_val = float(input("enter the learning rate: "))

    x1_value, x2_value, y1_value, y2_value = moon(total_points, dist, 10, 6)
    data = []
    data.extend([x1_value[i], y1_value[i], -1] for i in range(total_points))
    data.extend([x2_value[i], y2_value[i], 1] for i in range(total_points))
    data = np.asarray(data)
    normalize_data = normalize(np.copy(data))
    mse_vals, weights_vals = train(normalize_data, 50, eta_val)
    plot(weights_vals, mse_vals, data)
