import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from utilities import fromtxt2py
from ukf import UKF

'''парсинг файла с данными text.txt при помощи вспомогательной функции txt2py_dict.py из файла utilities.py'''
file_name = 'test.txt'  # Имя лога
data = fromtxt2py(file_name)
time = data['time']
imu_yaw_rate = data['YawRate_ESP']
sigma_yaw_angle = data['sigma_yaw_angle']
yaw_angle_gnss = data['yaw_angle_gnss']


def iterate_x(x, timestep, inputs):
    ''' в конкретном примере, модель сиcтемы линейна, но при необходимости, она может быть нелинейна'''
    x[0] = x[0] + timestep * inputs[0]
    return x


def main():
    np.set_printoptions(precision=3)

    n_dim = 1 #размерность ВС
    dt = 0.005 #временной шаг работы фильтра
    q = np.array([0.005]) #матрица шумов системы
    result = np.array([])
    alpha = 1 #настроечные параметры фильтра
    betta = 2 
    k = 3 - n_dim
    state_estimator = UKF(dt, n_dim, q, yaw_angle_gnss[0], 0.001 *
                          np.eye(1), alpha, betta, k, iterate_x)

    for i in range(len(time)):
        result = np.append(result, state_estimator.get_state())
        #этап прогноза
        state_estimator.predict(imu_yaw_rate[i])
        #этап обновления, матрица r задается данными о дисперсии измерения курсового угла из файла tets.txt
        r = np.array(np.rad2deg([sigma_yaw_angle[i]]))
        state_estimator.update([0], yaw_angle_gnss[i], r)

    plt.ylim(-100, 20)
    plt.ylabel('ψ, °', fontsize=14)
    plt.xlabel('время, с', fontsize=12)
    legend = plt.legend(loc='upper right')
    plt.plot(time, result, label='UKF')
    plt.plot(time, yaw_angle_gnss, label='Измерение')
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()