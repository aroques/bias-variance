import numpy as np
import matplotlib.pyplot as plt

def main():
    num_datasets = 10000000
    target_fn = np.square
    
    p1, p2 = get_experiment_data(num_datasets, target_fn)
    m = get_slope(p1, p2)
    b = get_y_intercept(p1, m)

    m_avg = round(np.average(m), 2)
    b_avg = round(np.average(b), 2)

    avg_g = 'y = {}x + {}'.format(m_avg, b_avg)
    bias = calculate_bias(get_x(num_datasets), m_avg, b_avg, target_fn)
    var = calculate_var(get_x(num_datasets), m, b, m_avg, b_avg)
    
    print_data('avg g(x)', avg_g)
    print_data('bias', bias)
    print_data('var', var)
    print_data('e_out', round(bias + var, 2))

    plot_exp(m_avg, b_avg, target_fn)


def print_data(label, value, width=10):
    print('{:{}}: {}'.format(label, width, value))


def get_experiment_data(num_datasets, target_fn):
    num_samples = num_datasets * 2
    s = np.random.uniform(-1, 1, num_samples)
    s = s.reshape(num_datasets, 2)

    x1, x2 = s[:, 0], s[:, 1]
    y1, y2 = target_fn(x1), target_fn(x2)

    p1 = np.column_stack((x1, y1))
    p2 = np.column_stack((x2, y2))

    return p1, p2


def get_slope(p1, p2):
    # slope = (y2 - y1) / (x2 - x1)
    return p2[:, 1] - p1[:, 1] / p2[:, 0] - p1[:, 0]


def get_y_intercept(p, m):
    # y - y1 = m(x - x1)
    # y - y1 = mx - mx1
    # y = mx - mx1 + y1
    # let b = -mx1 + y1
    return -m * p[:, 0] + p[:, 1]


def calculate_bias(x, m_avg, b_avg, target_fn):
    g_avg = hypothesis_fn(m_avg, x, b_avg)
    f_x = target_fn(x)
    return round(mean_squared_error(g_avg, f_x), 2)


def calculate_var(x, m, b, m_avg, b_avg):
    g_x = hypothesis_fn(m, x, b)
    g_avg = hypothesis_fn(m_avg, x, b_avg)
    return round(mean_squared_error(g_x, g_avg) / x.shape[0], 2)


def mean_squared_error(x1, x2):
    return np.average(np.square(x1 - x2))


def get_x(num_pts):
    return np.linspace(-1, 1, num_pts)


def hypothesis_fn(m, x, b):
    return m * x + b


def plot_exp(m_avg, b_avg, target_fn):
    plt.style.use('seaborn-whitegrid')
    fig, ax = plt.subplots()
    
    ax.set(title='Problem 2.24, p. 75')

    x = np.linspace(-1, 1, 30)
    
    ax.plot(x, target_fn(x), label='f(x)')
    ax.plot(x, hypothesis_fn(m_avg, x, b_avg), color='r', label='avg g(x)')
    
    ax.legend(facecolor='w', fancybox=True, frameon=True, edgecolor='black', borderpad=1)
    plt.show()


if __name__ == '__main__':
    main()
 