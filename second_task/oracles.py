import numpy as np
from scipy import sparse
from scipy import special


class BaseSmoothOracle:
    """
    Базовый класс для реализации оракулов.
    """
    def func(self, w):
        """
        Вычислить значение функции в точке w.
        """
        raise NotImplementedError('Func oracle is not implemented.')

    def grad(self, w):
        """
        Вычислить значение градиента функции в точке w.
        """
        raise NotImplementedError('Grad oracle is not implemented.')


class BinaryLogistic(BaseSmoothOracle):
    """
    Оракул для задачи двухклассовой логистической регрессии.
    
    Оракул должен поддерживать l2 регуляризацию.
    """

    def __init__(self, l2_coef=0):
        """
        Задание параметров оракула.
        
        l2_coef - коэффициент l2 регуляризации
        """
        self.alpha = l2_coef

    def func(self, X, y, w):
        """
        Вычислить значение функционала в точке w на выборке X с ответами y.
        
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        
        y - одномерный numpy array
        
        w - одномерный numpy array
        """
        M =  - X.dot(w) * y
        helper = np.zeros(M.shape)
        loss = np.sum(np.logaddexp(helper, M)) / M.shape[0]
        loss += np.sum(w ** 2) * self.alpha / 2
        return loss
    
    def mul_help(self, X, y):
        if sparse.issparse(X):
            return X.multiply(y[:, np.newaxis])
        else:
            return (X * y[:, np.newaxis])

    def grad(self, X, y, w):
        """
        Вычислить градиент функционала в точке w на выборке X с ответами y.
        
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        
        y - одномерный numpy array
        
        w - одномерный numpy array
        """
        M =  - X.dot(w) * y
        gradient = self.mul_help(self.mul_help(X, y), special.expit(M))
        gradient = self.alpha * w - gradient.sum(axis=0) / M.shape[0]
        return np.ravel(gradient)
