from matplotlib import pyplot as plt
import numpy as np

if __name__ == '__main__':
    a = np.random.rand(210,210,78)
    plt.plot(a[3,4,:])
    plt.show()
