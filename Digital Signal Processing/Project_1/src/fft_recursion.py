import matplotlib.pyplot as plt
import random
import numpy as np
import math
import cmath
import time


def preprocess(a):
    def zero_padding(ori):
        lenth = len(ori)
        bin_len = math.ceil(math.log(lenth, 2))
        length = 2 ** bin_len
        z = np.zeros((1, int(length-lenth)))
        ori = np.hstack((ori, z))
        return ori

    a = np.reshape(a, (1, len(a)))
    a = zero_padding(a)
    return a


def dit_fft(a):
    def rot_factor(n, k):
        expo = -1j*(2*math.pi*k/n)
        return cmath.exp(expo)

    b = np.empty([1, a.shape[1]], dtype=complex)
    if b.shape[1] > 2:
        length = b.shape[1]
        rf = np.array([rot_factor(length, _) for _ in xrange(length/2)])
        b_1 = np.array(a[0][::2], ndmin=2)
        b_2 = np.array(a[0][1::2], ndmin=2)
        b[0][0:length/2] = dit_fft(b_1) + rf * dit_fft(b_2)
        b[0][length/2:length] = dit_fft(b_1) - rf * dit_fft(b_2)
    elif b.shape[1] == 2:
        b[0][0] = a[0][0] + a[0][1]
        b[0][1] = a[0][0] - a[0][1]

    return b


def dif_fft(a):
    def rot_factor(n, k):
        expo = -1j*(2*math.pi*k/n)
        return cmath.exp(expo)

    b = np.empty([1, a.shape[1]], dtype=complex)
    if b.shape[1] > 2:
        length = b.shape[1]
        rf = np.array([rot_factor(length, _) for _ in xrange(length / 2)])
        b_1 = np.array(a[0][:length/2] + a[0][length/2:length], ndmin=2)
        b_2 = np.array((a[0][:length/2] - a[0][length/2:length]) * rf, ndmin=2)
        b[0][::2] = dif_fft(b_1)
        b[0][1::2] = dif_fft(b_2)
    elif b.shape[1] == 2:
        b[0][0] = a[0][0] + a[0][1]
        b[0][1] = a[0][0] - a[0][1]

    return b


def direct_dft(ori):
    def rot_factor(n, k):
        expo = -1j*(2*math.pi*k/n)
        return cmath.exp(expo)

    dft_1 = time.clock()
    length = ori.shape[1]
    result = np.zeros((1, length))
    result = result.astype(complex)
    for k in xrange(length):
        for n in xrange(length):
            result[0][k] += ori[0][n] * rot_factor(length, n*k)

    return result

x = np.array([random.randint(0, 5) for _ in xrange(1024)])

x = preprocess(x)

dit_1 = time.clock()
dit = dit_fft(x)
dit_2 = time.clock()

dif_1 = time.clock()
dif = dif_fft(x)
dif_2 = time.clock()

dft_1 = time.clock()
dft = direct_dft(x)
dft_2 = time.clock()

print dit_2 - dit_1
print dif_2 - dif_1
print dft_2 - dft_1
