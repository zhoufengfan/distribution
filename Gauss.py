import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    v = np.random.normal(loc=0, scale=1, size=2000)
    plt.hist(v, 128)
    plt.show()
