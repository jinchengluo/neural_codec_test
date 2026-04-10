# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 12:39:44 2026

@author: Francois
"""
import torch
import torchaudio
import torchaudio.functional as F
import numpy as np
from scipy import signal
import scipy.io.wavfile as wav
from matplotlib import pyplot as plt

def synchro_signals (x, y, dmax_ms = 80.0, tau_ms=15.0, sf=8000.0):
    
    # search range
    P = int(sf * dmax_ms * 1.0e-03)
        
    # forgetting factor
    alpha = np.exp(-1.0 / (sf * tau_ms * 1.0e-03))

    # envelop filter coefficients
    b = np.array([1.0 - alpha])
    a = np.array([1.0, -alpha])
    
    # envelop extraction
    et = signal.lfilter(b, a, np.abs(x[::-1]))
    ex = signal.lfilter(b, a, et[::-1])

    et = signal.lfilter(b, a, np.abs(y[::-1]))
    ey = signal.lfilter(b, a, et[::-1])
    
    # zero-mean centering for cross-correlation
    ex = ex - np.mean(ex)
    ey = ey - np.mean(ey)
    
    # set common length
    L = min(len(ex),len(ey))
    
    # cross-correlation maximisation
    delay = 0
    rn_max = -1.1
    
    for k in range(-P,P+1):
        
        rxx = np.sum(ex[P:L-P] ** 2)
        ryy = np.sum(ey[P+k:L-P+k] ** 2)
        rxy = np.sum(ex[P:L-P] * ey[P+k:L-P+k])
        
        rn = rxy / np.sqrt(rxx * ryy)
        
        if rn > rn_max:
            rn_max = rn
            delay = k
            
    # ouput synchronised signals
    if delay == 0:
        xs = x
        ys = y
    elif delay > 0:
        xs = x[:-delay]
        ys = y[+delay:]
    else:
        xs = x[-delay:]
        ys = y[:+delay]
    
    return delay, rn_max, xs, ys, ex, ey

# # read signals
# sampleRate, x = wav.read("./data/x.wav")
# sampleRate, y = wav.read("./data/y.wav")

# # synchronization process
# delay, rn_max, xs, ys, ex, ey = synchro_signals (x, y, dmax_ms = 80.0, tau_ms=15.0, sf = sampleRate)

# #♦ display
# offset = 15500
# N = 1000

# plt.figure(figsize=(14,8))
# plt.subplot(2,1,1)
# e1=ex[offset:offset+N]
# e2=ey[offset:offset+N]
# plt.plot(e1-np.mean(e1),'b')
# plt.plot(e2-np.mean(e2),'r')
# plt.grid(True)
# plt.title('before synchronization')
# plt.subplot(2,1,2)
# if delay >= 0:
#     e1=ex[offset:offset+N-delay]
#     e2=ey[offset+delay:offset+N]
#     plt.plot(e1-np.mean(e1),'b')
#     plt.plot(e2-np.mean(e2),'r')
# else:
#     e1=ex[offset-delay:offset+N]
#     e2=ey[offset:offset+N+delay]
#     plt.plot(e1-np.mean(e1),'b')
#     plt.plot(e2-np.mean(e2),'r')
# plt.title('after synchronization')
# plt.grid(True)
# plt.show()
    
# plt.figure(figsize=(14,8))
# plt.subplot(2,1,1)
# plt.plot(x[offset:offset+N],'b')
# plt.plot(y[offset:offset+N],'r')
# plt.grid(True)
# plt.title('before synchronization')
# plt.subplot(2,1,2)
# plt.plot(xs[offset:offset+N],'b')
# plt.plot(ys[offset:offset+N],'r')
# plt.grid(True)
# plt.title('after synchronization')
# plt.show()