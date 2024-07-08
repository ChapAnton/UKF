import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from utilities import *
from ukf import UKF
from ckf import CKF
'''парсинг файла с данными text.txt при помощи вспомогательной функции txt2py_dict.py из файла utilities.py'''
file_name = 'test.txt'  # Имя лога
data = fromtxt2py(file_name)
time = data['time']
imu_yaw_rate = data['YawRate_ESP']
sigma_yaw_angle = data['sigma_yaw_angle']
yaw_angle_gnss = unbounded(data['yaw_angle_gnss']) # использование вспомогателной функции из utilities.py


def iterate_x(x, timestep, inputs):
    x[0] = x[0] + timestep * inputs[0]
    return x


def main():

    n_dim = 1 # размерность ВС
    dt = 0.005 #в ременной шаг работы фильтра
    q = np.array([0.001]) # матрица шумов системы
    P_kk = np.diag([0.001])
    result = np.array([])
    alpha = 1 # настроечные параметры фильтра
    betta = 0 
    k = 0#3 - n_dim
    x = np.array([yaw_angle_gnss[0]])
    state_estimator = UKF(dt, n_dim, q, yaw_angle_gnss[0], 0.001 *
                          np.eye(1), alpha, betta, k, iterate_x)
    ckf = CKF(x_kk=x, P_kk=P_kk, Q_approx=q, R_approx=np.array([10]))
    x_filter_array = []
    for i in range(len(time)):
        result = np.append(result, state_estimator.get_state()) 
        state_estimator.predict(imu_yaw_rate[i])
        r = np.array(np.rad2deg([sigma_yaw_angle[i]]))
        state_estimator.update([0], yaw_angle_gnss[i], r)
        ckf.predict(ckf.x_kk, ckf.P_kk)
        ckf.update(ckf.x_kk, ckf.P_kk, np.array([yaw_angle_gnss[i]]), r)
        #print(ckf.x_kk.shape)
        x_filter_array.append(ckf.x_kk.reshape(1))
    plt.ylim(-100, 20)
    plt.ylabel('ψ, °', fontsize=14)
    plt.xlabel('время, с', fontsize=14)
    plt.plot(time, result, label='UKF')
    plt.plot(time, x_filter_array, label = "CKF")
    plt.plot(time, yaw_angle_gnss, label='Measurements')
    legend = plt.legend(loc='upper right', fontsize = 12)
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()
