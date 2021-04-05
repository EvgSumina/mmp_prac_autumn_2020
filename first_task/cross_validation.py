import numpy as np
import nearest_neighbors as nn


def kfold(n, n_folds = 3):
    indices = np.random.permutation(np.arange(n))
    indices = np.array(np.array_split(indices, n_folds))
    res = []
    for i in range(len(indices)):
        a = np.concatenate([indices[:i], indices[i + 1:]], axis = 0)
        res.append((np.hstack(a), indices[i]))
    return res


def accuracy(y, y_test):
    return np.array([y == y_test]).mean()


def knn_cross_val_score(X, y, k_list, score, cv = None, **kwargs):
    res = {}
    if cv == None:
        cv = kfold(X.shape[0])
    i = 0
    start = True
    for indices in cv:
        ind_0 = indices[0]
        ind_1 = indices[1]
        for k_neighbors in k_list[::-1]:
            classifier = nn.KNNClassifier(k_neighbors, **kwargs)
            classifier.fit(X[ind_0], y[ind_0])
            if start:
                distance, nearest_neighbors = classifier.find_kneighbors(X[ind_1], True)
                start = False
            else:
                distance = distance[:, :k_neighbors]
                nearest_neighbors = nearest_neighbors[:, :k_neighbors]
            predict = np.zeros(len(ind_1))
            j = 0
            for row in nearest_neighbors:
                if classifier.weights:
                    w = list(map(lambda x: 1 / (x + 10 ** (-5)), distance[j]))
                    predict[j] = np.argmax(np.bincount(classifier.y[row], weights = w))
                else:
                    predict[j] = np.argmax(np.bincount(classifier.y[row]))
                j += 1
            if k_neighbors in res.keys():
                res[k_neighbors][i] = accuracy(y[ind_1], predict)
            else:
                res[k_neighbors] = np.zeros(len(cv))
                res[k_neighbors][0] = accuracy(y[ind_1], predict)
        i += 1
        start = True
    return res
