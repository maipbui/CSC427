__author__ = 'Mai Bui'
__version__ = '09/09/2020'

from copy import deepcopy


def predict(row, weights):
    activation = weights[0]
    for i in range(len(row)-1):
        activation += weights[i+1] * row[i]
    return 1 if activation >= 0 else 0


def create_table(n):
    if n < 1:
        return [[]]
    table = create_table(n-1)
    return [row + [i] for row in table for i in [0,1]]


def get_and_logic(table):
    result = deepcopy(table)
    for row in result:
        row.append(int(all(row)))
    return result


def get_or_logic(table):
    result = deepcopy(table)
    for row in result:
        row.append(int(any(row)))
    return result

if __name__ == "__main__":
    n = 5
    dataset = []
    dataset = create_table(n)
    dataset_and = get_and_logic(dataset)
    weights_and = [-5]+[1]*n
    dataset_or = get_or_logic(dataset)
    weights_or = [-1]+[1]*n

    print("Create dataset for n = 5 using AND Logic")
    print(dataset_and)
    for row in dataset_and:
        prediction_and = predict(row, weights_and)
        print("Expected=%d, Predicted=%d" % (row[-1], prediction_and))

    print("Create dataset for n = 5 using OR Logic")
    print(dataset_or)
    for row in dataset_or:
        prediction_or = predict(row, weights_or)
        print("Expected=%d, Predicted=%d" % (row[-1], prediction_or))