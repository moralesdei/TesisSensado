#!/usr/bin/env python
# encoding: utf-8
# Random Demodulator Recovery Algorithm by Using Simulated Analog Data from LTSpice.
# Before using this script run rd_demo_ltsipce_gen script, run LTspice
# simulation and export vout.
# Based : Alexander López-Parrado (2017)
# Created : Juan camilo Montilla Orjuela, Deimer Andres Morales Herrera (2019)
# Contact : moralesdei@protonamil.com

# Librerias necesarias para el correcto funcionamiento de los algoritmos.
import sys
import pickle
from os.path import join, dirname, abspath
dir_abs = join(dirname(abspath(__file__)))
from numpy import pi, asarray, exp, arange, diag, sort, matmul
from numpy.fft import fft,fftshift
from matplotlib.pyplot import plot, show
from scipy.io import loadmat

#Cargamos todos nuestra area de trabajo.
dir_environment = join(dir_abs, 'matlab', 'rd')

# Cargamos el archivo con las variables de nuestro entorno de trabajo.
mat_fname = join(dir_environment,'environment.mat')
data = loadmat(mat_fname, squeeze_me=True)
locals().update(data)


n = arange(-N/2, N/2).reshape(1,-1)
f = open('signal_recovery.pckl', 'rb')
s = pickle.load(f)
f.close()

index = []
for count, element  in enumerate(s):
    if abs(element) > 0:
        index.append(count)
index = sort(index)

X_hat = N*s
tones = n.T
freq_hat = tones[index]
t = arange(0,Tx,1/(W)).reshape(1,-1)

x_hat = asarray((1/N)*sum(matmul(diag(X_hat[index,0]), exp(1j*(2*pi)/Tx*-freq_hat*t)).conj(), 1))
f = (arange(-len(x_hat)/2,len(x_hat)/2) * W)/len(x_hat)
plot(f/1e6,abs(fftshift(fft(x_hat))))
show()
