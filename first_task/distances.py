import numpy as np


def euclidean_distance(X, Y):
    A = np.add.outer(np.sum(X ** 2, axis = 1), np.sum(Y ** 2, axis = 1))
    B = np.dot(X, Y.T)
    return np.sqrt(A - 2 * B)


def cosine_distance(X, Y):
    A = X.dot(Y.T)
    A /= np.outer(np.linalg.norm(X, axis=1), np.linalg.norm(Y, axis=1))
    return 1 - A
