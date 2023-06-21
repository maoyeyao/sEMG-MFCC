from scipy import signal
import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt
from numpy import array
import pywt
from scipy.signal import filtfilt, iirnotch, freqz, butter
from scipy.fftpack import fft, fftshift, fftfreq


def calculate_rms(data):
    """
    计算给定数据的RMS
    参数：
    data: 包含数据的列表或数组
    返回值：
    RMS
    """
    n = len(data)
    square_sum = sum(x**2 for x in data)
    rms = math.sqrt(square_sum / n)
    return rms


def calculate_mav(data):
    """
    计算给定数据的MAV
    参数：
    data: 包含数据的列表或数组
    返回值：
    MAV
    """
    n = len(data)
    mav = sum(abs(x) for x in data) / n
    return mav


def calculate_wl(data):
    """
    计算给定数据的WL
    参数：
    data: 包含数据的列表或数组
    返回值：
    WL
    """
    n = len(data)
    wl = sum(abs(data[i] - data[i-1]) for i in range(1, n))
    return wl


def calculate_zc(data):
    """
    计算给定数据的ZC
    参数：
    data: 包含数据的列表或数组
    返回值：
    ZC
    """
    n = len(data)
    zc = 0
    for i in range(1, n):
        if data[i] * data[i-1] < 0:
            zc += 1
    return zc


def calculate_ssc(data):
    """
    计算给定数据的SSC
    参数：
    data: 包含数据的列表或数组
    返回值：
    SSC
    """
    n = len(data)
    ssc = 0
    for i in range(1, n-1):
        if (data[i] > data[i-1] and data[i] > data[i+1]) or \
           (data[i] < data[i-1] and data[i] < data[i+1]):
            ssc += 1
    return ssc


def median_frequency(signal, sample_rate):
    # 对信号进行傅里叶变换
    n = len(signal)
    freqs = fftfreq(n, d=1 / sample_rate)
    complex_spectrum = fft(signal)
    power_spectrum = np.abs(complex_spectrum) ** 2

    # 计算频率轴上的中位数
    sorted_power_spectrum = sorted(power_spectrum)
    cumsum = np.cumsum(sorted_power_spectrum)
    median_idx = np.searchsorted(cumsum, cumsum[-1] / 2)
    median_freq = freqs[power_spectrum.tolist().index(sorted_power_spectrum[median_idx])]

    return median_freq


def power_spectral_entropy(x, fs):
    # x: 输入信号
    # fs: 采样频率
    N = len(x)
    # 加窗和零填充
    x_p = x * signal.windows.hann(N)
    x_p = np.append(x_p, np.zeros(N))
    # 计算 PSD
    P = np.abs(np.fft.fft(x_p))**2 / (fs*N)
    P = P[:int(N/2)+1]
    P = P / np.sum(P)
    # 计算功率谱熵
    H = -np.sum(P*np.log2(P))
    return H

