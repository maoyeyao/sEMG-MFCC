import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt



for ind in range(1, 41):
    filepath = 'D:/NinaproDB2/DB2_s'+str(ind)+'/S'+str(ind)+'_E1_A1.mat'
    test_data = scio.loadmat(filepath)
    emg = test_data['emg']
    label = test_data['restimulus']
    dandu = np.argwhere(label)[:, 0]
    Separation = []
    x = []
    Separationlabel = []

    # 判断，找出连续的动作（把单独的动作挑选出来）；数据格式：0000……001111111……111000……000
    for i in range(0, len(dandu)):
        if i + 1 < len(dandu):
            if dandu[i + 1] == dandu[i] + 1:
                x.append(dandu[i])
            else:
                x.append(dandu[i])
                Separation.append(x)
                Separationlabel.append(label[dandu[i]])
                x = []
        else:
            x.append(dandu[len(dandu) - 1])
            Separation.append(x)
            Separationlabel.append(label[dandu[i]])

    emgSeparation = []   # 把连在一起的分开（这里17*6=102个动作）
    X = []
    for i in range(0, len(Separation)):
        index = Separation[i]
        X = emg[index[0]:(index[-1]+1), :]
        X = X.tolist()
        emgSeparation.append(X)
        X = []

    # （写入）配合list_txt文件进行使用
    path1 = 'D:/NinaproDB2/DB2_10/S'+str(ind)+'_E1_A1.txt'
    list_txt(path1, list=emgSeparation)
    # （读入）配合list_txt文件进行使用
    # C = list_txt(path, list=None)

    Splabel = np.array(Separationlabel)   # 标签：把连在一起的分开（这里17*6=102个动作）

    path2 = 'D:/NinaproDB2/DB2_10/S'+str(ind)+'_E1_A1label.txt'
    list_txt(path2, list=Separationlabel)