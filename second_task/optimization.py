import time
import numpy as np
from scipy import sparse
from scipy import special
import oracles


class GDClassifier:
    """
    Реализация метода градиентного спуска для произвольного
    оракула, соответствующего спецификации оракулов из модуля oracles.py
    """

    def __init__(self, loss_function='binary_logistic', step_alpha=1, step_beta=0, 
                 tolerance=1e-5, max_iter=1000, **kwargs):
        """
        loss_function - строка, отвечающая за функцию потерь классификатора. 
        Может принимать значения:
        - 'binary_logistic' - бинарная логистическая регрессия
                
        step_alpha - float, параметр выбора шага из текста задания
        
        step_beta - float, параметр выбора шага из текста задания
        
        tolerance - точность, по достижении которой, необходимо прекратить оптимизацию.
        Необходимо использовать критерий выхода по модулю разности соседних значений функции:
        если |f(x_{k+1}) - f(x_{k})| < tolerance: то выход
        
        max_iter - максимальное число итераций
        
        **kwargs - аргументы, необходимые для инициализации   
        """
        if loss_function == 'binary_logistic':
            self.loss_function = loss_function
        else:
            raise NotImplementedError('GDClassifier is not implemented.')
        self.step_alpha = step_alpha
        self.step_beta = step_beta
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.kwargs = kwargs
        self.threshold = 0.5
        self.history = {}

    def fit(self, X, y, w_0=None, trace=False):
        """
        Обучение метода по выборке X с ответами y
        
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        
        y - одномерный numpy array
        
        w_0 - начальное приближение в методе
        
        trace - переменная типа bool
      
        Если trace = True, то метод должен вернуть словарь history, содержащий информацию
        о поведении метода. Длина словаря history = количество итераций + 1 (начальное приближение)
        
        history['time']: list of floats, содержит интервалы времени между двумя итерациями метода
        history['func']: list of floats, содержит значения функции на каждой итерации
        (0 для самой первой точки)
        """
        if not w_0 is None:
            self.weights = w_0
        else:
            self.weights = np.random.uniform(-1, 1, X.shape[1])
        my_oracle = oracles.BinaryLogistic(**self.kwargs)
        self.history['time'] = [0]
        self.history['func'] = [0]
        for n in range(0, self.max_iter + 1):
            start_time = time.time()
            grad = my_oracle.grad(X, y, self.weights)
            loss = my_oracle.func(X, y, self.weights)
            self.weights -= (self.step_alpha / (n + 1) ** self.step_beta) * grad
            if n > 0:
                self.history['func'].append(loss)
                self.history['time'].append(time.time() - start_time)
                if np.abs(loss - self.history['func'][n - 1]) < self.tolerance:
                    break
            else:
                self.history['func'] = [loss]
                self.history['time'] = [time.time() - start_time]
        if trace:
            return self.history

    def predict(self, X):
        """
        Получение меток ответов на выборке X
        
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        
        return: одномерный numpy array с предсказаниями
        """
        if sparse.issparse(X):
            X = X.toarray()
        y_pred = (special.expit(np.dot(X, self.weights)) > self.threshold).astype(int)
        return y_pred

    def predict_proba(self, X):
        """
        Получение вероятностей принадлежности X к классу k
        
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        
        return: двумерный numpy array, [i, k] значение соответветствует вероятности
        принадлежности i-го объекта к классу k
        """
        if sparse.issparse(X):
            X = X.toarray()
        p_pred = special.expit(np.dot(X, self.weights))
        return np.vstack((1 - p_pred, p_pred))

    def get_objective(self, X, y):
        """
        Получение значения целевой функции на выборке X с ответами y
        
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        y - одномерный numpy array
        
        return: float
        """
        my_oracle = oracles.BinaryLogistic(self.kwargs)
        return my_oracle.func(X, y, self.weights)

    def get_gradient(self, X, y):
        """
        Получение значения градиента функции на выборке X с ответами y
        
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        y - одномерный numpy array
        
        return: numpy array, размерность зависит от задачи
        """
        my_oracle = oracles.BinaryLogistic(self.kwargs)
        return my_oracle.grad(X, y, self.weights)

    def get_weights(self):
        """
        Получение значения весов функционала
        """    
        return self.weights


class SGDClassifier(GDClassifier):
    """
    Реализация метода стохастического градиентного спуска для произвольного
    оракула, соответствующего спецификации оракулов из модуля oracles.py
    """

    def __init__(self, loss_function, batch_size, step_alpha=1, step_beta=0, 
                 tolerance=1e-5, max_iter=1000, random_seed=153, **kwargs):
        """
        loss_function - строка, отвечающая за функцию потерь классификатора. 
        Может принимать значения:
        - 'binary_logistic' - бинарная логистическая регрессия
        
        batch_size - размер подвыборки, по которой считается градиент
        
        step_alpha - float, параметр выбора шага из текста задания
        
        step_beta- float, параметр выбора шага из текста задания
        
        tolerance - точность, по достижении которой, необходимо прекратить оптимизацию
        Необходимо использовать критерий выхода по модулю разности соседних значений функции:
        если |f(x_{k+1}) - f(x_{k})| < tolerance: то выход 
        
        
        max_iter - максимальное число итераций (эпох)
        
        random_seed - в начале метода fit необходимо вызвать np.random.seed(random_seed).
        Этот параметр нужен для воспроизводимости результатов на разных машинах.
        
        **kwargs - аргументы, необходимые для инициализации
        """
        if loss_function == 'binary_logistic':
            self.loss_function = loss_function
        else:
            raise NotImplementedError('SGDClassifier is not implemented.')
        self.batch_size = batch_size
        self.step_alpha = step_alpha
        self.step_beta = step_beta
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.random_seed = random_seed
        self.kwargs = kwargs
        self.threshold = 0.5
        self.history = {}

    def fit(self, X, y, w_0=None, trace=False, log_freq=1):
        """
        Обучение метода по выборке X с ответами y
        
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        
        y - одномерный numpy array
                
        w_0 - начальное приближение в методе
        
        Если trace = True, то метод должен вернуть словарь history, содержащий информацию 
        о поведении метода. Если обновлять history после каждой итерации, метод перестанет 
        превосходить в скорости метод GD. Поэтому, необходимо обновлять историю метода лишь
        после некоторого числа обработанных объектов в зависимости от приближённого номера эпохи.
        Приближённый номер эпохи:
            {количество объектов, обработанных методом SGD} / {количество объектов в выборке}
        
        log_freq - float от 0 до 1, параметр, отвечающий за частоту обновления. 
        Обновление должно проиходить каждый раз, когда разница между двумя значениями приближённого номера эпохи
        будет превосходить log_freq.
        
        history['epoch_num']: list of floats, в каждом элементе списка будет записан приближённый номер эпохи:
        history['time']: list of floats, содержит интервалы времени между двумя соседними замерами
        history['func']: list of floats, содержит значения функции после текущего приближённого номера эпохи
        history['weights_diff']: list of floats, содержит квадрат нормы разности векторов весов с соседних замеров
        (0 для самой первой точки)
        """
        np.random.seed(self.random_seed)
        if sparse.issparse(X):
            X = X.toarray()
        if not w_0 is None:
            self.weights = w_0
        else:
            self.weights = np.random.uniform(-1, 1, X.shape[1])
        my_oracle = oracles.BinaryLogistic(**self.kwargs)
        self.history['time'] = []
        self.history['func'] = []
        self.history['epoch_num'] = []
        self.history['weights_diff'] = []
        size = X.shape[0]
        count = 0
        time_helper = []
        func_helper = []
        epoch_helper = []
        weights_helper = None
        old_loss = 0
        flag = False
        for n in range(0, self.max_iter + 1):
            ind = np.arange(size)
            np.random.shuffle(ind)
            for i in range(0, size, self.batch_size):
                count += 1
                start_time = time.time()
                X_batch = X[ind[i : min(size, i + self.batch_size)]]
                y_batch = y[ind[i : min(size, i + self.batch_size)]]
                grad = my_oracle.grad(X_batch, y_batch, self.weights)
                loss = my_oracle.func(X_batch, y_batch, self.weights)
                self.weights -= (self.step_alpha / n ** self.step_beta) * grad
                func_helper.append(loss)
                if weights_helper is None:
                    weights_helper = [np.linalg.norm(self.weights) ** 2]
                else:
                    weights_helper.append(np.linalg.norm((self.step_alpha / n ** self.step_beta) * grad) ** 2)
                epoch_num = count / size
                epoch_helper.append(epoch_num)
                if (epoch_num) > log_freq:
                    self.history['func'].append(func_helper)
                    self.history['time'].append(time_helper)
                    self.history['epoch_num'].append(epoch_num_helper)
                    self.history['weights_diff'].append(weights_diff_helper)
                if np.abs(loss - old_loss) < self.tolerance:
                    flag = True
                    break
                old_loss = loss
            if flag:
                break
        if trace:
            return self.history
