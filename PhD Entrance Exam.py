
# coding: utf-8


get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft
import random
from sklearn import preprocessing


#Exercice 1
y = [(2*np.sin(i)+2*np.cos(i)) for i in np.arange(0,30,0.1)]
x = [i for i in np.arange(0,30,0.1)]

#Exercice 2
plt.subplot()
plt.title('Original')
plt.plot(x,y)

#Exercice 3
y_ftt = np.fft.fft(y) # or y_ftt = fft(y)

plt.subplot()
plt.title('FFT')
plt.plot(y_ftt)

y_noise = y
for i in range(0, len(y)):
    y_noise[i] = y_noise[i] + random.random()

plt.subplot()
plt.title('Original Noised')
plt.plot(y_noise)

y_noise_ftt = fft(y_noise) # or y_noise_ftt = np.fft.fft(y_noise)

plt.subplot()
plt.title('FFT Noised')
plt.plot(y_noise_ftt)

#Exercice 4

scale_y = preprocessing.scale(y)

#Calculate MEAN for both signals
#Plot results
plt.subplot()
plt.title('Scale Original')
plt.plot(scale_y)


scale_y = preprocessing.scale(scale_y)
plt.subplot()
plt.title('scale Originals')
plt.plot(scale_y)


#Calculate STD for original signal
STD_scale_y = scale_y.std(axis=0)
STD_scale_y


#Calculate STD for noise FTT
STD_scale_y_noise_ftt = scale_y_noise_ftt.std(axis=0)
STD_scale_y_noise_ftt

#Calculate MEAN for original signal
MEAN_scale_y = scale_y.mean(axis=0)
MEAN_scale_y

#Calculate MEAN for noise FTT
MEAN_scale_y_noise_ftt = scale_y_noise_ftt.mean(axis=0)
MEAN_scale_y_noise_ftt

