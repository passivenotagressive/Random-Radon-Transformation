import numpy as np
import math
def generate_line_points(im_len, k, b, thickness=1):
    x = np.array([], 'int64')
    y = np.array([], 'int64')

    for i in range(0, thickness):

        x_i = np.arange(0, im_len, 1)
        y_i = k * x_i + b + i
        points = [(x_i[j], y_i[j]) for j in range(im_len)]
        filtered_points = np.array(list(filter(lambda x: (x[1] >= 0) * (x[1] < im_len), points)))
        if (filtered_points.shape[0] != 0):
            x_i = filtered_points[:, 0]
            y_i = filtered_points[:, 1]
            x = np.hstack([x, x_i])
            y = np.hstack([y, y_i])
    return x, y


def generate_line_points_angle(im_length, im_hight, r, theta, thickness=1):
    x = np.array([], 'int64')
    y = np.array([], 'int64')
    x0 = im_length // 2
    y0 = im_hight // 2

    for i in range(0, thickness):
        x_i = np.array([])
        y_i = np.array([])
        if (abs(theta - np.pi / 2) < 1e-4):
            x_i = np.array([x0 + i] * im_length)
            y_i = np.arange(0, im_hight, 1)
            x = np.hstack([x, x_i])
            y = np.hstack([y, y_i])
        else:
            x_i = np.arange(0, im_length, 1)
            y_i = np.array(y0 + r / (math.cos(theta)) + (x_i - x0) * math.tan(theta)).astype('int') + i
            points = [(x_i[j], y_i[j]) for j in range(im_length)]
            filtered_points = np.array(list(filter(lambda pixel: (pixel[1] >= 0) * (pixel[1] < im_hight), points)))
            if (filtered_points.shape[0] != 0):
                x_i = filtered_points[:, 0]
                y_i = filtered_points[:, 1]
                x = np.hstack([x, x_i])
                y = np.hstack([y, y_i])
    return x, y


def random_img_one_line(im_length, im_hight):
    img = np.zeros((im_length, im_hight), dtype=np.uint8)
    max_rho = (im_length ** 2 + im_hight ** 2) ** 0.5
    rho = (np.random.random() - 0.5) * max_rho
    theta = np.random.random() * np.pi
    x_points, y_points = generate_line_points_angle(im_length, im_hight, rho, theta, 1)
    img[x_points, y_points] = 255
    img = 255 - img
    return img

def check_identical(list_1, list_2):
    return all(x == y for x, y in zip(list_1, list_2))

def generate_one_line(im_length, im_hight, rho, theta, thickness=2):
    """Generate one line on the plot of given size and with given rho and theta """
    x_points, y_points = generate_line_points_angle(im_length, im_hight, rho, theta, thickness)
    img = np.zeros((im_length, im_hight), dtype=np.uint8)
    img[x_points, y_points] = 255
    img = 255 - img
    return img
