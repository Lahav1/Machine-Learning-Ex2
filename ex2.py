import numpy as np
from scipy import stats
import random
import os
import sys


def turnToFloat(abalonesArr):
    """
    Cast all the values in the array to float.

    Parameters
    ----------
    abalonesArr : array of strings.

    Returns
    -------
    newArr : array of floats.
    """
    newArr = []
    for line in abalonesArr:
        newLine = []
        for value in line:
            newLine.append(float(value))
        newArr.append(np.asarray(newLine))
    return np.asarray(newArr)


def setIMFValues(abalonesArr):
    """
    Replace the I/M/F by a numeric value.

    Parameters
    ----------
    abalonesArr : array of abalones information.

    Returns
    -------
    abalonesArr : array after all I/M/F are replaced.
    """
    for abalones in abalonesArr:
        if abalones[0] == "I":
            abalones[0] = 0
        if abalones[0] == "F":
            abalones[0] = 1
        if abalones[0] == "M":
            abalones[0] = 2
    return abalonesArr


def normalizeValues(abalonesArr):
    """
    Normalize each column in the array by Z-Score normalization.

    Parameters
    ----------
    abalonesArr : array of abalones information.

    Returns
    -------
    abalonesArr : array with normalized values.
    """
    # transpose the matrix to get each parameter in a separate array.
    transposedArr = np.asarray([*zip(*abalonesArr)])
    # iterate the parameters one by one.
    for line in transposedArr:
        stats.zscore(line)
    # re-transpose matrix.
    return np.asarray([*zip(*transposedArr)])


def perceptron(X_train, Y_train):
    """
    Build a weights vector using the perceptron algorithm.

    Parameters
    ----------
    X_train : array of information (abalones).
    Y_train : array of matching labels (ages).

    Returns
    -------
    weights : weights vector.
    """
    # Hyper-parameters
    epochs = 50
    eta = 0.01
    weights = np.zeros((3, 8))
    # Train the algorithm epochs times.
    for e in range(epochs):
        s = list(zip(X_train, Y_train))
        random.shuffle(s)
        X_train, Y_train = zip(*s)
        for x, y in zip(X_train, Y_train):
            wx = np.dot(weights, x)
            # y_hat is prediction
            y_hat = np.argmax(wx)
            y = int(y[0])
            # Update weights only if the prediction was wrong.
            if y != y_hat:
                weights[y, :] = weights[y, :] + eta * x
                weights[y_hat, :] = weights[y_hat, :] - eta * x
        eta = eta / epochs
    return weights


def SVM(X_train, Y_train):
    """
    Build a weights vector using the SVM algorithm.

    Parameters
    ----------
    X_train : array of information (abalones).
    Y_train : array of matching labels (ages).

    Returns
    -------
    weights : weights vector.
    """
    # Hyper-parameters
    epochs = 50
    eta = 0.01
    lamda = 0.01
    weights = np.zeros((3, 8))
    # Train the algorithm epochs times.
    for e in range(epochs):
        s = list(zip(X_train, Y_train))
        random.shuffle(s)
        X_train, Y_train = zip(*s)
        for x, y in zip(X_train, Y_train):
            # y_hat is prediction
            y_hat = np.argmax(np.dot(weights, x))
            y = int(y[0])
            # Update weights only if the prediction was wrong.
            if y != y_hat:
                weights[y, :] = (1 - eta * lamda) * weights[y, :] + eta * x
                weights[y_hat, :] = (1 - eta * lamda) * weights[y_hat, :] - eta * x
                for i in range (3):
                    if (i != y) and (i != y_hat):
                        weights[i, :] = (1 - eta * lamda) * weights[i, :]
        eta = eta / epochs
    return weights


def tau(weights, x, y, y_hat):
    """
    Calculate the tau value for PA.

    Parameters
    ----------
    weights : weights vector.
    x : specific abalone's information.
    y : specific abalone's label.
    y_hat : abalone's prediction.

    Returns
    -------
    tau value.
    """
    normal_X = 0
    for i in range(len(x)):
        normal_X += np.power(x[i], 2)
    normal_X = np.sqrt(normal_X)
    a = np.dot(weights[y, :], x)
    b = np.dot(weights[y_hat, :], x)
    loss = max(0, 1 - a + b)
    # For case of division by 0.
    if (2 * np.power(normal_X, 2)) == 0:
        return 1
    return loss / (2 * np.power(normal_X, 2))


def PA(X_train, Y_train):
    """
    Build a weights vector using the PA algorithm.

    Parameters
    ----------
    X_train : array of information (abalones).
    Y_train : array of matching labels (ages).

    Returns
    -------
    weights : weights vector.
    """
    # Hyper-parameters
    epochs = 70
    weights = np.zeros((3, 8))
    # Train the algorithm epochs times
    for e in range(epochs):
        s = list(zip(X_train, Y_train))
        random.shuffle(s)
        X_train, Y_train = zip(*s)
        for x, y in zip(X_train, Y_train):
            # y_hat is prediction
            y_hat = np.argmax(np.dot(weights, x))
            y = int(y[0])
            # Update weights only if the prediction was wrong.
            if y != y_hat:
                t = tau(weights, x, y, y_hat)
                weights[y, :] = weights[y, :] + t * x
                weights[y_hat, :] = weights[y_hat, :] - t * x
        return weights


def createResultsList(weights, X_test):
    """
    Multiplies each information vector in the test array by the weights vector, and returns a list of results.

    Parameters
    ----------
    weights : weights vector.
    X_test : array of information (abalones).

    Returns
    -------
    results : list of results.
    """
    results = []
    X_test = list(X_test)
    for t in range (0, len(X_test)):
        results.append(np.argmax(np.dot(weights, X_test[t])))
    return results


def printResults(perceptron_results, svm_results, pa_results):
    """
    Prints all the predictions for each abalones in the following format:
    "perceptron: __, svm: __, pa: __"

    Parameters
    ----------
    perceptron_results : predictions list of perceptron.
    svm_results : predictions list of svm.
    pa_results : predictions list of pa.
    """
    for r1, r2, r3 in zip(perceptron_results, svm_results, pa_results):
        print("perceptron: %d, svm: %d, pa: %d" % (r1, r2, r3))


# get paths from command line arguments.
train_x_path = sys.argv[1]
train_y_path = sys.argv[2]
test_x_path = sys.argv[3]

# open files.
train_x = open(os.path.join(os.path.dirname(__file__), train_x_path), 'r')
train_y = open(os.path.join(os.path.dirname(__file__), train_y_path), 'r')
test_x = open(os.path.join(os.path.dirname(__file__), test_x_path), 'r')

# initialize lists.
abalonesTrainList = []
labelsTrainList = []
abalonesTestList = []

# start reading lines.
xline = train_x.readline()
yline = train_y.readline()
# iterate all the lines in both train files.
while xline and yline:
    # add abalone to abalones list and label to labels list accordingly.
    xarr = str.split(xline, ",")
    abalonesTrainList.append(xarr)
    labelsTrainList.append(yline[0]);
    # move on to next line.
    xline = train_x.readline()
    yline = train_y.readline()

xtestline = test_x.readline()
# iterate all the lines in both train files.
while xtestline:
    # add abalone to abalones list and label to labels list accordingly.
    xarr = str.split(xtestline, ",")
    abalonesTestList.append(xarr)
    # move on to next line.
    xtestline = test_x.readline()

# prepare train inputs array.
abalonesTrainArr = np.asarray(abalonesTrainList)
abalonesTrainArr = setIMFValues(abalonesTrainArr)
abalonesTrainArr = turnToFloat(abalonesTrainArr)
abalonesTrainArr = normalizeValues(abalonesTrainArr)
# prepare train labels array.
labelsTrainArr = np.asarray(labelsTrainList)
labelsTrainArr = turnToFloat(labelsTrainArr)
# prepare test inputs array.
abalonesTestArr = np.asarray(abalonesTestList)
abalonesTestArr = setIMFValues(abalonesTestArr)
abalonesTestArr = turnToFloat(abalonesTestArr)
abalonesTestArr = normalizeValues(abalonesTestArr)

# get weights vector for each of the algorithms.
w_perceptron = perceptron(abalonesTrainArr, labelsTrainArr)
w_svm = SVM(abalonesTrainArr, labelsTrainArr)
w_pa = PA(abalonesTrainArr, labelsTrainArr)

# create lists of results for each algorithm.
perceptron_results = createResultsList(w_perceptron, abalonesTestArr)
svm_results = createResultsList(w_svm, abalonesTestArr)
pa_results = createResultsList(w_pa, abalonesTestArr)

# print the results in the requested format.
printResults(perceptron_results, svm_results, pa_results)