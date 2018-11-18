import numpy as np
import matplotlib.pyplot as plt

def main():
    p1, p2 = get_experiment_data(num_datasets=10000000)
    m = get_slope(p1, p2)
    b = get_y_intercept(p1, m)

    m_avg = round(np.average(m), 2)
    b_avg = round(np.average(b), 2)

    avg_g = 'y = {}x + {}'.format(m_avg, b_avg)
    bias = calculate_bias(get_x(), m_avg, b_avg)
    var = calculate_var(get_x(), m, b, m_avg, b_avg)
    
    print_data('avg g(x)', avg_g)
    print_data('bias', bias)
    print_data('var', var)
    print_data('eout', round(bias + var, 2))

    plot_exp(m_avg, b_avg)


def print_data(label, value, width=10):
    print('{:{}}: {}'.format(label, width, value))


def get_experiment_data(num_datasets):
    num_samples = num_datasets * 2
    s = np.random.uniform(-1, 1, num_samples)
    s = s.reshape(num_datasets, 2)

    x1, x2 = s[:, 0], s[:, 1]
    x1_squared, x2_squared = np.square(x1), np.square(x2)

    p1 = np.column_stack((x1, x1_squared))
    p2 = np.column_stack((x2, x2_squared))

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


def calculate_bias(x, m_avg, b_avg):
    g_avg = m_avg * x + b_avg
    f_x = np.square(x)
    return round(np.average(np.square(g_avg - f_x)), 2)


def calculate_var(x, m, b, m_avg, b_avg):
    g_x = m * x + b
    g_avg = m_avg * x + b_avg
    return round(np.average(np.square(g_x - g_avg)) / x.shape[0], 2)


def get_x():
    return np.linspace(-1, 1, 10000000)


def plot_exp(m_avg, b_avg):
    plt.style.use('seaborn-whitegrid')
    fig, ax = plt.subplots()
    
    ax.set(title='Problem 2.24, p. 75')

    x = np.linspace(-1, 1, 30)
    
    ax.plot(x, np.square(x), label='f(x)')
    ax.plot(x, m_avg * x + b_avg, color='r', label='avg g(x)')
    
    ax.legend(facecolor='w', fancybox=True, frameon=True, edgecolor='black', borderpad=1)
    plt.show()


if __name__ == '__main__':
    main()
 