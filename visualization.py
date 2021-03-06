import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from math import cos
from math import sin
from math import pi
from math import asin

matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = 20

np.set_printoptions(threshold=np.inf)


def plot_fourier():
    # A = [-1.4007e+01, -3.4718e+00, 6.5743e-01, 8.8805e-02, -3.0080e-01,
    #      -3.5471e-01, -1.4764e-02, 2.6334e-01, 1.8092e-02, -3.2248e-03,
    #      -2.0254e-01, -1.4549e-01, -2.5778e-01, 7.1596e-02, 4.7056e-01,
    #      -2.2896e-01, -2.3136e-01, 3.5681e-01, 2.9199e-01, -3.1465e-01,
    #      1.3936e-01, 4.0030e-01, 2.6467e-01, 1.4517e-01, -3.2120e-01,
    #      -7.4065e-02, -2.1800e-01, 9.7215e-03, 3.6617e-01, -1.3458e-01]
    #
    # B = [1.0666e+02, 5.5523e-02, 1.5156e-01, 1.0732e+00, 1.3449e+00,
    #      4.3107e-01, -1.6028e-01, 9.6653e-02, 9.3628e-02, -2.9941e-01,
    #      -2.1513e-01, 1.6375e-01, 2.5644e-01, 5.6640e-01, 2.1762e-01,
    #      -5.5805e-01, 3.3145e-01, 4.4965e-01, 6.2410e-02, 4.6896e-01,
    #      1.1122e-01, -4.2424e-01, -2.3626e-01, 1.3257e-01, 1.6765e-01,
    #      1.1034e-01, -3.3865e-01, 4.1961e-01, 3.5860e-02, -3.5997e-01]
    #
    # a0 = [-0.6680]
    # w = [0.1082]

    A2 = [-1.2352e+01, -2.6777e+00, 1.1735e+00, 9.6817e-01, -3.8853e-01,
          -2.9416e-01, 6.2111e-01, 5.5016e-01, -4.0051e-01, -5.8255e-01,
          -6.7169e-01, -2.0782e-02, 1.2045e-01, -1.5635e-01, 3.5693e-01,
          -4.4047e-03, 8.0439e-02, -7.8501e-02, -3.4026e-01, 6.3158e-02,
          1.3354e-01, -7.9358e-02, -5.2533e-01, -3.2353e-04, -3.4361e-01,
          5.8855e-02, 2.0341e-01, 2.0815e-01, -3.5870e-01, 1.2187e-01,
          3.9006e-01, 2.8208e-01, 2.2837e-01, -5.6674e-03, -9.9744e-02,
          -3.8618e-02, -8.5518e-02, 2.4358e-01, -3.8433e-02, 2.4588e-02,
          -1.8911e-01, 5.3780e-02, -3.2673e-01, 2.6276e-02, 1.0357e-01,
          6.8024e-02, -1.4796e-01, -1.2333e-01, 1.1115e-01, -3.6225e-03,
          -5.5287e-02, -2.7419e-03, -2.5348e-01, 1.1222e-01, 3.2781e-01,
          -3.3503e-01, -7.2745e-02, -3.7198e-01, -2.9558e-01, -3.4114e-01]

    B2 = [1.5636e+02, -6.9833e+00, -2.9326e+00, 2.8808e+00, 1.0443e+00,
          2.1343e-01, 2.9940e-01, -2.2237e-01, -4.1574e-01, -1.0562e-01,
          2.2238e-01, -4.4288e-01, -2.2463e-01, -5.7661e-02, -2.2151e-01,
          -3.4247e-02, 6.8086e-02, 3.8578e-01, 6.4125e-01, 8.2399e-02,
          1.7905e-02, 1.8146e-02, 1.4927e-01, 3.1433e-02, -2.1360e-01,
          -8.1496e-02, 2.5444e-01, -2.8598e-01, -3.5381e-01, -2.8549e-01,
          -1.0203e-01, -1.2462e-01, -3.1015e-01, 4.4962e-01, 4.0865e-01,
          -3.2740e-01, 1.6421e-01, 2.9512e-01, -3.6827e-01, -1.8384e-01,
          3.5175e-01, 1.9272e-01, -4.6295e-02, 4.1105e-01, 2.7000e-01,
          -2.4768e-01, 3.9407e-01, 4.3073e-01, 3.0924e-02, 8.8034e-02,
          7.4031e-04, 5.9237e-01, 1.8244e-01, 2.4543e-01, 3.1397e-01,
          7.1408e-03, -3.6451e-02, 5.1323e-01, 2.0668e-01, -2.9739e-01]

    a02 = [-1.4152]
    w2 = [-0.1744]

    A = [-1.2352e+01]
    B = [1.5636e+02]
    a0 = [-1.4152]
    w = [-0.1744]

    x = np.arange(-200, 200, 0.001)
    y = []

    for i in x:
        ans = 0
        for j in range(len(A)):
            ans += A[j] * cos((j + 1) * w[0] * i) + B[j] * sin((j + 1) * w[0] * i)

        ans += a0[0]
        y.append(ans)

    # ymax=max(y)
    # for i in range(len(y)):
    #     y[i]=asin(y[i]/ymax)
    y = np.array(y)

    z = []

    for i in x:
        ans = 0
        for j in range(len(A2)):
            ans += A2[j] * cos((j + 1) * w2[0] * i) + B2[j] * sin((j + 1) * w2[0] * i)

        ans += a02[0]
        z.append(ans)

    # zmax=max(z)
    # for i in range(len(z)):
    #     z[i]=asin(z[i]/zmax)
    z = np.array(z)

    plt.plot(x, y, )
    plt.plot(x, z, )
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis()
    plt.show()


def plot_fourier_reslut_show():
    A = [-6.5930, -1.5981, -0.1886,  0.0171,  0.1118,  0.5507, -0.1946,  0.2597,
         0.3966, -0.2536]

    B = [11.2644,  3.4163,  1.8624,  0.6027,  0.3399,  0.1305,  0.1632, -0.0441,
         0.1768, -0.1061]

    a0 = [11.4677]
    w = [0.3129]

    x = np.arange(-160, 160, 0.001)
    y = []

    for i in x:
        ans = 0
        for j in range(len(A)):
            ans += A[j] * cos((j + 1) * w[0] * i) + B[j] * sin((j + 1) * w[0] * i)

        ans += a0[0]
        y.append(ans)

    # ymax = max(max(y), -min(y))
    # for i in range(len(y)):
    #     y[i] = asin(y[i] / ymax)
    y = np.array(y)

    plt.plot(x, y, )

    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis()
    plt.show()


def plot_Sin():
    A = [-3.4770]
    B = [0.5474]

    a0 = [0.9981]
    w = [0.7789]

    x = np.arange(-20, 20, 0.001)
    y = []

    for i in x:
        ans = A[0] * sin(B[0]) * cos(w[0] * i) + A[0] * cos(B[0]) * sin(w[0] * i)

        y.append(ans + a0[0])

    ymax = max(y)
    # for i in range(len(y)):
    #     y[i] = asin(y[i] / ymax)
    y = np.array(y)

    plt.plot(x, y, )

    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis()
    plt.show()


def plot_ReLU():
    x = np.arange(-2, 2, 0.001)
    y = []
    L = 1
    w = pi / L
    n = 60
    A = []
    B = []

    for j in range(1, n + 1):
        aj = (cos(j * pi) - 1.0) * L / (pi * pi * j * j)
        bj = -cos(pi * j) * L / (j * pi)
        A.append(aj)
        B.append(bj)
    for i in x:
        ans = 0
        for j in range(1, n + 1):
            ans += A[j - 1] * cos(j * w * i) + B[j - 1] * sin(j * w * i)

        ans += L / 2.0
        y.append(ans)

    y = np.array(y)

    print(A)
    print(B)
    print(w)
    print(L / 2.0)

    plt.plot(x, y, )
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis()
    plt.show()


if __name__ == '__main__':
    # plot_ReLU()
    # plot_fourier()
    plot_fourier_reslut_show()
    # plot_Sin()
