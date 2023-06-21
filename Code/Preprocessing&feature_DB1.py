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


for ind in range(1, 28):
    for ex in range(1, 4):
        SEMG = scio.loadmat('D:/NinaproDB1/s' + str(ind) + '/S' + str(ind) + '_A1_E' + str(ex) + '.mat')
        semg = SEMG['emg']
        NUBER = SEMG['rerepetition']
        LABEL = SEMG['restimulus']
        label_all = np.argwhere(LABEL)[:, 0]
        number_all = np.argwhere(NUBER)[:, 0]
        Separation = []
        x = []
        Separationlabel = []

        # 判断，找出连续的动作（把单独的动作挑选出来）；数据格式：0000……001111111……111000……000
        for i in range(0, len(label_all)):
            if i + 1 < len(label_all):
                if label_all[i + 1] == label_all[i] + 1:
                    x.append(label_all[i])
                else:
                    x.append(label_all[i])
                    Separation.append(x)
                    Separationlabel.append(LABEL[label_all[i]])
                    x = []
            else:
                x.append(label_all[len(label_all) - 1])
                Separation.append(x)
                Separationlabel.append(LABEL[label_all[i]])

        emgSeparation = []  # 把连在一起的分开（这里17*6=102个动作）
        for i in range(0, len(Separation)):
            index = Separation[i]
            X_semg = semg[index[0]:(index[-1] + 1), :]
            for tongdao in range(0, 10):
                semg_tongdao = X_semg[:, tongdao]
                wavelet = pywt.Wavelet('db8')
                levels = pywt.dwt_max_level(len(semg_tongdao), wavelet)
                coeffs = pywt.wavedec(semg_tongdao, wavelet, level=levels)

                # 设置阈值
                sigma = np.median(np.abs(coeffs[-levels]))
                # threshold = sigma * np.sqrt(2 * np.log(len(semg_tongdao))) * 1.5
                threshold = sigma * np.sqrt(2 * np.log(len(semg_tongdao))) * 1

                # 阈值去噪
                coeffs_thresh = []
                for xiao in range(len(coeffs)):
                    coeffs_thresh.append(pywt.threshold(coeffs[xiao], threshold, mode='soft'))

                # 重构信号
                x_denoised = pywt.waverec(coeffs_thresh, wavelet)
                semg_tongdao = x_denoised

                RMS = calculate_rms(semg_tongdao)
                MAV = calculate_mav(semg_tongdao)
                # ZC = calculate_zc(semg_tongdao)
                MF = median_frequency(semg_tongdao, sample_rate=100)
                SSC = calculate_ssc(semg_tongdao)
                POW = power_spectral_entropy(semg_tongdao, fs=100)
                feature_tongdao = [RMS, WL, MAV, SSC, MF, POW]
                feature_tongdao = np.array(feature_tongdao)
                if tongdao == 0:
                    feature = feature_tongdao
                else:
                    feature = np.vstack((feature, feature_tongdao))
            feature = feature.flatten()
            label_index = LABEL[index[0]][0]
            number_index = NUBER[index[0]][0]
            if label_index in label_list:
                label_index2 = label_list.index(label_index)
                label = label_actual[label_index2]
                if number_index <= 8:
                    train_feature = np.vstack((train_feature, feature))
                    train_label.append(label)
                else:
                    test_feature = np.vstack((test_feature, feature))
                    test_label.append(label)









