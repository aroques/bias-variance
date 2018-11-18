import numpy as np

def main():
    p1, p2 = get_experiment_data(num_samples=20000)
    print(p1[0])
    print(p2[0])

def get_y_intercept(p, m):
    # y - y1 = m(x - x1)
    # y - y1 = mx - mx1
    # y = mx - mx1 + y1
    # let b = -mx1 + y1
    return -m * p[0] + p[1]


def get_slope(p1, p2):
    # slope = (y2 - y1) / (x2 - x1)
    return p2[1] - p1[1] / p2[0] - p1[0]


def get_experiment_data(num_samples):
    s = np.random.uniform(-1, 1, num_samples)
    num_rows = int(num_samples / 2)
    s = s.reshape(num_rows, 2)

    x1, x2 = s[:, 0], s[:, 1]
    x1_squared, x2_squared = np.square(x1), np.square(x2)

    p1 = np.column_stack((x1, x1_squared))
    p2 = np.column_stack((x2, x2_squared))

    return p1, p2

if __name__ == '__main__':
    main()
 