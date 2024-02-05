import numpy as np
import scipy.linalg
from copy import deepcopy


class UKF:
    def __init__(self, timestep, num_states, process_noise, initial_state, initial_covar, alpha, betta, k, iterate_function):
        """
        :param timestep временной шаг работы фильтра
        :param num_states: int, размер вектора состояния(ВС)
        :param process_noise: матрица ковариаций процесса
        :param initial_state: начальное состояние ВС
        :param initial_covar: начальное значние матрицы ковариации (при отсутствии априориных данных задается большыми величинами)
        :param alpha: настроечный параметр UKF (0..1)
        :param k: настроечный параметр UKF (обычно выбирается как 3 - num_states)
        :param beta: настроечный параметр UKF, beta = 2 является лучшим выбором для нормального распределения
        :param iterate_function: функция, описывающая процесс
        """
        self.n_dim = int(num_states)
        self.timestep = timestep
        self.n_sig = 1 + num_states * 2
        self.q = process_noise
        self.x = initial_state
        self.p = initial_covar
        self.betta = betta
        self.alpha = alpha
        self.k = k
        self.iterate = iterate_function

        self.lambd = pow(self.alpha, 2) * (self.n_dim + self.k) - self.n_dim

        self.covar_weights = np.zeros(self.n_sig)
        self.mean_weights = np.zeros(self.n_sig)

        self.covar_weights[0] = (
            self.lambd / (self.n_dim + self.lambd)) + (1 - pow(self.alpha, 2) + self.betta)
        self.mean_weights[0] = (self.lambd / (self.n_dim + self.lambd))

        for i in range(1, self.n_sig):
            self.covar_weights[i] = 1 / (2*(self.n_dim + self.lambd))
            self.mean_weights[i] = 1 / (2*(self.n_dim + self.lambd))

        self.sigmas = self.__get_sigmas()

    def __get_sigmas(self):
        # генерация сигма-точек c использованием разложения Холецкого
        ret = np.zeros((self.n_sig, self.n_dim))
        chol = (np.linalg.cholesky(self.p))
        tmp_mat = chol

        ret[0] = self.x
        for i in range(self.n_dim):

            ret[i+1] = self.x + np.sqrt(self.n_dim + self.lambd) * tmp_mat[i]
            ret[i+1+self.n_dim] = self.x - \
                np.sqrt(self.n_dim + self.lambd) * tmp_mat[i]

        return ret.T

    def predict(self, *inputs):
        # обсчет шага прогноза UKF
        self.sigmas = self.__get_sigmas()
        sigmas_out = np.array(
            [self.iterate(x, self.timestep, inputs) for x in self.sigmas.T]).T
        x_out = np.zeros(self.n_dim)

        for i in range(self.n_dim):
            # среднее значение представляет собой сумму
            # взвешенных значений этой переменной для каждой сигма-точки
            x_out[i] = sum((self.mean_weights[j] * sigmas_out[i][j]
                           for j in range(self.n_sig)))

        p_out = np.zeros((self.n_dim, self.n_dim))

        for i in range(self.n_sig):

            diff = sigmas_out.T[i] - x_out
            diff = np.atleast_2d(diff)
            p_out += self.covar_weights[i] * np.dot(diff.T, diff)

        # добавление шума процесса
        p_out += self.timestep * self.q

        self.x = x_out
        self.p = p_out
        self.sigmas = sigmas_out

    def update(self, states, data, r_matrix):
        """
        performs a measurement update
        :param states: список, содержащий индексы обновляемых состояний
        :param data: измерения обновляемых состояний
        :param r_matrix: матрица ошибок процесса
        """

        num_states = len(states)

        # обсчет сигма-точек только для обновляемых состояний
        sigmas_split = np.split(self.sigmas, self.n_dim)
        y = np.concatenate([sigmas_split[i] for i in states])

        x_split = np.split(self.x, self.n_dim)
        y_mean = np.concatenate([x_split[i] for i in states])

        y_diff = deepcopy(y)
        x_diff = deepcopy(self.sigmas)
        for i in range(self.n_sig):
            for j in range(num_states):
                y_diff[j][i] -= y_mean[j]
            for j in range(self.n_dim):
                x_diff[j][i] -= self.x[j]

        # вычисление ковариации измерений
        p_yy = np.zeros((num_states, num_states))
        for i, val in enumerate(np.array_split(y_diff, self.n_sig, 1)):
            p_yy += self.covar_weights[i] * val.dot(val.T)

        # добавление шума измерений
        p_yy += r_matrix
        p_xy = np.zeros((self.n_dim, num_states))
        for i, val in enumerate(zip(np.array_split(y_diff, self.n_sig, 1), np.array_split(x_diff, self.n_sig, 1))):
            p_xy += self.covar_weights[i] * val[1].dot(val[0].T)

        k = np.dot(p_xy, np.linalg.inv(p_yy))

        y_actual = data

        self.x += np.dot(k, (y_actual - y_mean))
        self.p -= np.dot(k, np.dot(p_yy, k.T))

    def get_state(self, index=-1):
        # получение вектора состояния
        if index >= 0:
            return self.x[index]
        else:
            return self.x

    def get_covar(self):
        # получение матрицы ковариации
        return self.p
