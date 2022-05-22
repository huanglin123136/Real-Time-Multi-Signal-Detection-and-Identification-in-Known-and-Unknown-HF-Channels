import pandas as pd
import numpy as np
fft10 = np.fromfile(r'F:\vs2017_work\Array_Demo_orig_miao_80ms_debug\build\recvBuf.dat',dtype='float32')
fft10 = fft10.reshape((16,10,1760))

RC20 =np.fromfile(r'F:\vs2017_work\Array_Demo_orig_miao_80ms_debug\build\recvBuf2.dat',dtype='float32')
RC20 = RC20.reshape((16,1,3520))

RC40 = np.fromfile(r'F:\vs2017_work\Array_Demo_orig_miao_80ms_debug\build\recvBuf3.dat',dtype='float32')
RC40 = RC40.reshape((16,1,7040))
print(fft10)