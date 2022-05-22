import numpy as np
import pandas as pd
import scipy.io as sio




class_dic = [0,1,2,3,4,5,6,7,8,9,10,11,12,13]

dic = {'AM': 0, 'SSB': 1, 'PSK': 2, '2FSK': 3,  'CW': 4,
       'Saopin': 5,  'Interp': 6, 'SingleSound': 7, 'AM_expand_large': 8,
       'Interp_flash': 9, 'Unknow': 10, 'Saopin_large': 11, 'Noise': 12, 'CW_fast': 13}

if __name__ == '__main__':
    #读取保存结果，每64个代表整段1.25MHz频谱
    final_co = np.load('result.npz',allow_pickle = True)['final_co']
    final_sc = np.load('result.npz',allow_pickle = True)['final_sc']
    final_cl = np.load('result.npz', allow_pickle=True)['final_cl']
    #经过融合后结果
    total_co = []
    total_sc =[]
    total_cl =[]
    total_frequce =[]
    N_times = 40
    N_class = 14
    #初始化检查结果,40次，20类

    for i in range (0,N_times * 64):
        time_i = int(np.floor(i / 64))
        real_freq = np.fmod(i,64)
        co = final_co[i]
        sc = final_sc[i]
        cl = final_cl[i]

        for j in range(len(co)):
            #当前频率检测结果
            center_fre = (co[j][0]+co[j][1])/2 + real_freq * 1250/ 64

            # center_cls = cl[j]
            center_cls = class_dic[cl[j]]
            center_sc = sc[j]

            sub_flag = 0
            sub_ti =[]
            #判断是否可以融合
            for ti in range(len(total_frequce)):
                sub_fre =np.abs(center_fre-total_frequce[ti])
                #中心频率相差1KHz以内,检测为位置同一信号
                if sub_fre < 1 :
                    #融合位置sub_ti
                    sub_ti = ti
                    sub_flag = 1
                    continue

            #若可以融合
            if sub_flag:
                total_cl[sub_ti][time_i][center_cls] += 1

                total_co[sub_ti][time_i][0] = co[j][0] + real_freq * 1250 / 64
                total_co[sub_ti][time_i][1] = co[j][1] + real_freq * 1250 / 64

            else:
            #新建中心频率，BW，类别

                ini_fi = np.zeros((N_times, N_class))
                ini_fi[time_i][center_cls] = 1
                total_frequce.append(center_fre)
                total_cl.append(ini_fi)

                ini_co = np.zeros((N_times, 2))
                ini_co[time_i][0] = co[j][0] + real_freq * 1250 / 64
                ini_co[time_i][1] = co[j][1] + real_freq * 1250 / 64
                total_co.append(ini_co)



    print(len(total_frequce))
    total_cl = np.array(total_cl)
    total_co = np.array(total_co)
    total_frequce = np.array(total_frequce)
    sio.savemat('results.mat', {'total_cl': total_cl, 'total_co': total_co , 'total_frequce':total_frequce})





