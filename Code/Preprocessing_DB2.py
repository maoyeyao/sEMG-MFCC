from scipy import signal
import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt
from numpy import array
import pywt
from scipy.signal import filtfilt, iirnotch, freqz, butter
from scipy.fftpack import fft, fftshift, fftfreq


def list_txt(path, list=None):
    '''

    :param path: 储存list的位置
    :param list: list数据
    :return: None/relist 当仅有path参数输入时为读取模式将txt读取为list
             当path参数和list都有输入时为保存模式将list保存为txt
    '''
    if list != None:
        file = open(path, 'w')
        file.write(str(list))
        file.close()
        return None
    else:
        file = open(path, 'r')
        rdlist = eval(file.read())
        file.close()
        return rdlist

for ind in range(1, 41):
    sEMG = list_txt('D:/NinaproDB2/DB2_10/S' + str(ind) + '_E1_A1.txt', list=None)
    CC = []
    for i in range(0, 102):
        x1 = np.array(sEMG[i])
        XX = np.zeros((len(x1), 12))
        for j in range(0, 12):
            L1 = x1[:, j]
            # plt.figure()
            # plt.plot(L1)
            # plt.show()

            sos = signal.butter(10, 500, btype='low', output='sos', fs=2000)

            LF_x1 = signal.sosfilt(sos, L1)
            # plt.figure()
            # plt.plot(LF_x1)
            # plt.show()

            # Filtering using iirnotch
            fs = 2000
            f0 = 50
            w0 = f0 / (fs / 2)
            Q = 30
            b, a = iirnotch(w0, Q)
            # filter response
            w, h = freqz(b, a)
            filt_freq = w * fs / (2 * np.pi)
            y_50Hz = filtfilt(b, a, LF_x1)
            # plt.figure()
            # plt.plot(y_50Hz)
            # plt.show()

            # 选用db8小波
            threshold = 0.1
            w = pywt.Wavelet('db8')
            maxlev = pywt.dwt_max_level(len(y_50Hz), w.dec_len)
            coffs = pywt.wavedec(y_50Hz, 'db8', level=maxlev)
            for n in range(1, len(coffs)):
                coffs[n] = pywt.threshold(coffs[n], threshold * max(abs(coffs[n])))

            xiaobo = pywt.waverec(coffs, 'db8')
            # plt.figure()
            # plt.plot(xiaobo)
            # plt.show()
            chuli = xiaobo[1:]
            if len(chuli) == len(x1):
                cyy = chuli
            elif len(chuli) > len(x1):
                cyy = chuli[(len(chuli)-len(x1)):]
            elif len(chuli) < len(x1):
                cyy = np.zeros(len(x1))
                cyy[0:len(chuli)] = chuli

            XX[:, j] = cyy
        XX = XX.tolist()
        CC.append(XX)
    path = 'D:/lvbo/DB_2/S' + str(ind) + '_E1_A1.txt'
    list_txt(path, list=CC)