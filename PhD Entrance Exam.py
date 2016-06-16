import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft
import random
from sklearn import preprocessing


#Exercice 1
y = [(2*np.sin(i)+2*np.cos(i)) for i in np.arange(0,30,0.1)]
x = [i for i in np.arange(0,30,0.1)]

#Exercice 2
plt.subplot(2,2,1)
plt.title('Original')
plt.plot(x,y)

#Exercice 3
y_ftt = np.fft.fft(y) # or y_ftt = fft(y)

plt.subplot(2,2,2)
plt.title('FFT')
plt.plot(y_ftt)

y_noise = y
for i in range(0, len(y)):
    y_noise[i] = y_noise[i] + random.random()

plt.subplot(2,2,3)
plt.title('Original Noised')
plt.plot(y_noise)

y_noise_ftt = fft(y_noise) # or y_noise_ftt = np.fft.fft(y_noise)

plt.subplot(2,2,4)
plt.title('FFT Noised')
plt.plot(y_noise_ftt)

#Exercice 4

scale_y_noise_ftt = preprocessing.scale(y_noise_ftt)
scale_y = preprocessing.scale(y)
#Calculate STD for both signals
STD_scale_y_noise_ftt = scale_y_noise_ftt.std(axis=0)
STD_scale_y = scale_y.std(axis=0)
#Calculate MEAN for both signals
MEAN_scale_y_noise_ftt = scale_y_noise_ftt.mean(axis=0)
MEAN_scale_y = scale_y.mean(axis=0)
#Plot results
plt.subplot(2,2,5)
plt.title('Scale Original')
plt.plot(scale_y)

plt.subplot(2,2,6)
plt.title('scale_y_noise_ftt')
plt.plot(scale_y_noise_ftt)

plt.subplot(2,2,7)
plt.title('scale_y_noise_ftt')
plt.plot(STD_scale_y)

plt.subplot(2,2,8)
plt.title('scale_y_noise_ftt')
plt.plot(STD_scale_y_noise_ftt)

plt.subplot(2,2,8)
plt.title('scale_y_noise_ftt')
plt.plot(MEAN_scale_y)

plt.subplot(2,2,8)
plt.title('scale_y_noise_ftt')
plt.plot(MEAN_scale_y_noise_ftt)

plt.show()
