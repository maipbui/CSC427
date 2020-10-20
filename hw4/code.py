__author__ = 'Mai Bui'
__version__ = '09/30/2020'

from random import random
import matplotlib.pyplot as plt
import math
import numpy as np


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


if __name__ == "__main__":
    # Initialize variable
    total_points = int(input("enter number of points in each moon: "))
    dist = float(input("enter the distance between 2 moons: "))
    I = np.identity(3)
    lamda = 0

    # Create dataset
    x1, x2, y1, y2 = moon(total_points, dist, 10, 6)
    dataset = []
    dataset.extend([1, x1[i], y1[i]] for i in range(total_points))
    dataset.extend([1, x2[i], y2[i]] for i in range(total_points))
    dataset = np.asarray(dataset)
    np.random.shuffle(dataset)

    # Processing for correlation matrix and weight vector
    vect = dataset
    vect_transpose = vect.T

    # Compute correlation matrix
    correlation_mat = vect_transpose.dot(vect)
    correlation_mat = np.reshape(correlation_mat, (3, 3))

    # Compute weight vector
    weight = (np.linalg.inv(correlation_mat + lamda * I)).dot(vect_transpose.dot(dataset[:, 2]))

    # Classify 2 moons separately
    class_1 = dataset[:, 1] * weight[1] + dataset[:, 2] * weight[2] >= -weight[0]
    class_1_dataset = dataset[class_1]
    class_2 = dataset[:, 1] * weight[1] + dataset[:, 2] * weight[2] < -weight[0]
    class_2_dataset = dataset[class_2]

    # Visualization the decision boundary and the double moon
    x = np.asarray([-20, 32])
    y = (-weight[0] - weight[1] * x) / weight[2]
    plt.plot(x, y, c="k")
    plt.xlim(-20, 32)
    plt.title('Testing result')
    plt.scatter(class_1_dataset[:, 1], class_1_dataset[:, 2], c="b", marker='x', s=50)
    plt.scatter(class_2_dataset[:, 1], class_2_dataset[:, 2], c="r", marker='x', s=50)
    plt.show()
