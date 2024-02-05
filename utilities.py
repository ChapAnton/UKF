from array import array


def fromtxt2py(file_name):
	with open(file_name) as f:
		variables = f.readline().rstrip().split(',')
		data_dict = {variables[i]: array('d', []) for i in range(len(variables))}
		for i, line in enumerate(f):
			data = line.rstrip().split(',')
			data_dict['time'].append(float(data[0]))
			data_dict['x_ecef'].append(float(data[1]))
			data_dict['y_ecef'].append(float(data[2]))
			data_dict['z_ecef'].append(float(data[3]))
			data_dict['lat'].append(float(data[4]))
			data_dict['long'].append(float(data[5]))
			data_dict['yaw_angle_gnss'].append(float(data[6]))
			data_dict['YawRate_ESP'].append(float(data[7]))
			data_dict['v_can'].append(float(data[8]))
			data_dict['v_hor'].append(float(data[9]))
			data_dict['v_vert'].append(float(data[10]))
			data_dict['w_k1'].append(float(data[11]))
			data_dict['w_k2'].append(float(data[12]))
			data_dict['w_k3'].append(float(data[13]))
			data_dict['w_k4'].append(float(data[14]))
			data_dict['tetta_r'].append(float(data[15]))
			data_dict['gnss_status'].append(float(data[16]))
			data_dict['a_x_ESP'].append(float(data[17]))
			data_dict['a_y_ESP'].append(float(data[18]))
			data_dict['sigma_x'].append(float(data[19]))
			data_dict['sigma_y'].append(float(data[20]))
			data_dict['sigma_yaw_angle'].append(float(data[21]))
			data_dict['satellite_num'] = (float(data[22]))
		
	return data_dict

def unbounded(head):
    '''функция снимает ограничения 0-360 с курсового угла, измеренного СНС'''
    k = 0
    course_un = [head[0], ]
    for i in range(len(head) - 1):
        if head[i + 1] - head[i] > 350:
            k -= 1
        if head[i + 1] - head[i] < -350:
            k += 1
        course_un.append(head[i + 1] + k * 360)
    return course_un
