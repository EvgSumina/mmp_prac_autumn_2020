from sklearn.neighbors import NearestNeighbors
import numpy as np
import distances


class KNNClassifier:
    def __init__(self, k = 1, strategy = 'my_own', metric = 'euclidean', 
                 weights = False, test_block_size = 30):
        self.n_neighbors = k
        self.strategy = strategy
        self.metric = metric
        self.weights = weights
        self.test_block_size = test_block_size


    def fit(self, X, y):
        self.y = np.array(y)
        if self.strategy != 'my_own':
            self.neigh = NearestNeighbors(n_neighbors = self.n_neighbors, 
                                          algorithm = self.strategy, 
                                          metric = self.metric)
            self.neigh = self.neigh.fit(X)
        else:
            self.X = np.array(X)


    def find_kneighbors(self, X, return_distance = False):
        n = self.n_neighbors
        if self.strategy != 'my_own':
            return self.neigh.kneighbors(X, n, return_distance)
        else:
            right_bond = self.test_block_size
            if self.metric == 'euclidean':
                D = distances.euclidean_distance(X[0 : right_bond], self.X)
            elif self.metric == 'cosine':
                D = distances.cosine_distance(X[0 : right_bond], self.X)
            indices = D.argsort(axis = 1)[:,:n]
            array = np.sort(D, axis = 1)[:,:n]
            test_size = right_bond
            while test_size < X.shape[0]:
                right_bond = self.test_block_size + test_size
                if self.metric == 'euclidean':
                    D = distances.euclidean_distance(X[test_size : right_bond], self.X)
                elif self.metric == 'cosine':
                    D = distances.cosine_distance(X[test_size : right_bond], self.X)
                indices = np.vstack((indices, D.argsort(axis = 1)[:,:n]))
                array = np.vstack((array, np.sort(D, axis = 1)[:,:n]))
                test_size = right_bond
        if return_distance:
            return (array, indices)
        else:
            return indices


    def predict(self, X):
        distance, nearest_neighbors = self.find_kneighbors(X, True)
        i = 0
        predict = np.zeros(X.shape[0])
        for row in nearest_neighbors:
            if self.weights:
                w = list(map(lambda x: 1 / (x + 10 ** (-5)), distance[i]))
                predict[i] = np.argmax(np.bincount(self.y[row], weights = w))
            else:
                predict[i] = np.argmax(np.bincount(self.y[row]))
            i += 1
        return predict
