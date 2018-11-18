import numpy as np

def main():
    x = get_experiment_data(num_samples=20000)
    print(x[0])


def get_experiment_data(num_samples):
    s = np.random.uniform(-1, 1, num_samples)
    num_rows = int(num_samples / 2)
    s = s.reshape(num_rows, 2)

    x1, x2 = s[:, 0], s[:, 1]
    x1_squared, x2_squared = np.square(x1), np.square(x2) 

    return np.column_stack((x1, x1_squared, x2, x2_squared))

if __name__ == '__main__':
    main()
 