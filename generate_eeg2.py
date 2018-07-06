import numpy as np
import matplotlib.pyplot as plt
import math
import pdb
from scipy import signal
import matplotlib.pyplot as plt
from math import *

def get_eeg():

    time = range(256*8)
    eeg = np.zeros((len(time)))

    for t in time:
        eeg[t] = 14*sin(2*pi*4*t/256) + 52*sin(2*pi*22*t/256) + 16*sin(2*pi*5*t/256) + 43*sin(2*pi*11*t/256)

    #plt.plot(eeg)
    #plt.show()


    numCoeffs = math.floor(256/4)+1
    fc = (256*0.8)/(2*8)
    wc = 2*(fc/256)
    h = signal.firwin(int(numCoeffs-1), wc, window = 'hann', scale = False)

    Fchp = 0.5
    alpha = (2*math.pi*Fchp)/256

    wc = 2*(Fchp/256)
    b, a = signal.butter(5, wc, 'high')

'''
    plt.plot(h)
    plt.plot(b)
    plt.plot(a)
    plt.show()
'''

    eeg1 = signal.lfilter(h,1, eeg)
    eeg2 = signal.lfilter(b, a, eeg1)
    eeg3 = signal.decimate(eeg2,8)

    #eeg3 = np.reshape(eeg3[:-1], (1,len(eeg3)-1,1))
    eeg3 = np.reshape(eeg3[:], (1,len(eeg3),1))
'''
    plt.plot(eeg)
    plt.figure()
    plt.plot(eeg1)
    plt.figure()
    plt.plot(eeg2)
    plt.figure()

    #plt.plot(eeg3)
    plt.show()
'''
    return(eeg3)

