import numpy as np

def grad_finite_diff(function, w, eps=1e-8):
    """
    Возвращает численное значение градиента, подсчитанное по следующией формуле:
        result_i := (f(w + eps * e_i) - f(w)) / eps,
        где e_i - следующий вектор:
        e_i = (0, 0, ..., 0, 1, 0, ..., 0)
                          >> i <<
    """
    helper = np.zeros(w.shape)
    result = np.zeros(w.shape)
    for i in range(w.shape[0]):
        helper[i] = eps
        result[i] = (function(w + helper) - function(w)) / eps
        helper[i] = 0
    return result