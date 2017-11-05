from __future__ import division
import matplotlib.pyplot as plt
import random
import numpy as np
import math
import cmath
import time


def dit_fft(ori):
    def bin_inv(x, bin_len):
        string = bin(x)
        string = string[2:]
        string = string.zfill(bin_len)
        return int(string[::-1], 2)

    def zero_padding(a):
        lenth = len(a)
        bin_len = math.ceil(math.log(lenth, 2))
        length = 2 ** bin_len
        z = np.zeros((int(length-lenth), 1))
        a = np.vstack((a, z))
        return a

    def rearrange(a):
        a = np.reshape(a, (len(a), 1))
        a = zero_padding(a)
        length = len(a)
        bin_len = int(math.log(length, 2))

        b = np.zeros((length, 1))
        for i in xrange(len(a)):
            j = bin_inv(i, bin_len)
            b[j] = a[i]
        b = b.astype(complex)
        return b

    def pick_input(a, level):
        n = len(a)
        pair = np.zeros((int(n/2), 2))
        unit_length = 2 ** level
        unit_num = int(n / unit_length)
        stride = int(n / 2 / unit_num)
        row = 0
        rot_dict = {}
        for i in xrange(unit_num):
            index = i * unit_length
            for j in xrange(stride):
                index1 = index + j
                index2 = index1 + stride
                pair[row, 0] = int(index1)
                pair[row, 1] = int(index2)
                row += 1
                rot_dict[index2] = [unit_length, j]
        return pair, rot_dict

    def rot_factor(n, k):
        expo = -1j*(2*math.pi*k/n)
        return cmath.exp(expo)

    def basic_btfly(x1, x2, rot):
        y1 = x1 + x2 * rot
        y2 = x1 - x2 * rot
        return y1, y2

    def butterfly(a, pair, rot_dict):
        basic_num = len(pair)
        for i in xrange(basic_num):
            index1, index2 = pair[i, :]
            index1 = int(index1)
            index2 = int(index2)
            n, k = rot_dict.get(index2)
            rot = rot_factor(n, k)
            a[index1], a[index2] = basic_btfly(a[index1], a[index2], rot)
        return a

    start = time.clock()
    result = rearrange(ori)
    n = len(result)
    level = int(math.log(n, 2))
    for l in xrange(level):
        pair, rot_dict = pick_input(result, l+1)
        result = butterfly(result, pair, rot_dict)
    end = time.clock()
    return result, end-start


def dif_fft(ori):
    def bin_inv(x, bin_len):
        string = bin(x)
        string = string[2:]
        string = string.zfill(bin_len)
        return int(string[::-1], 2)

    def zero_padding(a):
        lenth = len(a)
        bin_len = math.ceil(math.log(lenth, 2))
        length = 2 ** bin_len
        z = np.zeros((int(length-lenth), 1))
        a = np.vstack((a, z))
        return a

    def rearrange(a):
        length = len(a)
        bin_len = int(math.log(length, 2))

        b = np.zeros((length, 1))
        b = b.astype(complex)
        for i in xrange(len(a)):
            j = bin_inv(i, bin_len)
            b[j] = a[i]
        return b

    def pick_input(a, level, level_length):
        n = len(a)
        pair = np.zeros((int(n/2), 2))
        unit_length = 2 ** (level_length-level+1)
        unit_num = int(n / unit_length)
        stride = int(n / 2 / unit_num)
        row = 0
        rot_dict = {}
        for i in xrange(unit_num):
            index = i * unit_length
            for j in xrange(stride):
                index1 = index + j
                index2 = index1 + stride
                pair[row, 0] = int(index1)
                pair[row, 1] = int(index2)
                row += 1
                rot_dict[index2] = [unit_length, j]
        return pair, rot_dict

    def rot_factor(n, k):
        expo = -1j*(2*math.pi*k/n)
        return cmath.exp(expo)

    def basic_btfly(x1, x2, rot):
        y1 = x1 + x2
        y2 = (x1 - x2) * rot
        return y1, y2

    def butterfly(a, pair, rot_dict):
        basic_num = len(pair)
        for i in xrange(basic_num):
            index1, index2 = pair[i, :]
            index1 = int(index1)
            index2 = int(index2)
            n, k = rot_dict.get(index2)
            rot = rot_factor(n, k)
            a[index1], a[index2] = basic_btfly(a[index1], a[index2], rot)
        return a

    start = time.clock()
    ori = np.reshape(ori, (len(ori), 1))
    ori = zero_padding(ori)
    result = ori.astype(complex)
    n = len(result)
    level = int(math.log(n, 2))
    for l in xrange(level):
        pair, rot_dict = pick_input(result, l + 1, level)
        result = butterfly(result, pair, rot_dict)
    result = rearrange(result)
    end = time.clock()
    return result, end-start


def direct_dft(ori):
    def rot_factor(n, k):
        expo = -1j*(2*math.pi*k/n)
        return cmath.exp(expo)

    start = time.clock()
    length = len(ori)
    result = np.zeros((length, 1))
    result = result.astype(complex)
    for k in xrange(length):
        for n in xrange(length):
            result[k] += ori[n] * rot_factor(length, n*k)
    end = time.clock()
    return result, end-start


'''''''''''
N = input("Please input the number of point(radix 2):")
n = int(2 ** N)

x = np.arange(0, n)

y1, time1 = dif_fft(x)
y2, time2 = dit_fft(x)
y3, time3 = direct_dft(x)

print("The time of dif_fft is %f" % time1)
print("The time of dit_fft is %f" % time2)
print("The time of dft is %f" % time3)

'''''''''''
time_fft = np.zeros((6, 1))
time_dit = np.zeros((6, 1))
time_dif = np.zeros((6, 1))
time_dft = np.zeros((6, 1))
enhance = np.zeros((6, 1))
N = np.array([10, 11, 12, 13, 14, 15])
for i in xrange(6):
    print i     # indicate which step the program is taking
    n = int(2 ** N[i])

    x = np.array([random.randint(0, 100) for _ in xrange(n)])

    y1, time1 = dit_fft(x)
    y2, time2 = dif_fft(x)
    y3, time3 = direct_dft(x)

    time_dit[i] = time1
    time_dif[i] = time2
    time_fft[i] = 0.5 * (time1 + time2)
    time_dft[i] = time3
    enhance[i] = 2 * time3 / (time1 + time2)

np.savetxt("dit.txt", time_dit)
np.savetxt("dif.txt", time_dif)
np.savetxt("dft.txt", time_dft)
np.savetxt("fft.txt", time_fft)
# np.savetxt("enhance.txt", enhance)

plt.figure(1)
plt.plot(N, time_fft, label='$fft$')
plt.plot(N, time_dft, label='$dft$')
plt.title('Time comparison between Radix 2 FFT and DFT')
plt.xlabel('power of the point')
plt.ylabel('time/s')
plt.legend()

plt.figure(2)
plt.plot(N, enhance, label='$actual\ enhancement$')
plt.plot(N, np.array([2**i/i for i in N]), label='$N/log_2N$')
plt.title('Time enhancement of FFT over DFT')
plt.xlabel('power of the point')
plt.ylabel('enhancement ratio')
plt.legend()
plt.show()
