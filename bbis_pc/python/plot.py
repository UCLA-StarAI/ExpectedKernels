# unused
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np

if __name__ == "__main__":
    axis = np.load("../axis.npz")
    direct = np.load("../direct.npz")
    collapsed = np.load("../collapsed.npz")
    plt.cla()
    plt.plot(axis, direct, label="gibbs inference")
    plt.plot(axis, collapsed, label="collapsed inference")
    plt.xlabel("sample size")
    plt.ylabel("estimation error")
    plt.legend(loc=1)
    plt.savefig("../err.jpg")
    print("figure save to ../err.jpg")
