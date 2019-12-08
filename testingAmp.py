#!/usr/bin/env python
# encoding: utf-8
# Testing para el algoritmo AMP
# Based : Alexander López-Parrado (2017)
# Created : Juan camilo Montilla Orjuela, Deimer Andres Morales Herrera (2019)
# Contact : moralesdei@protonamil.com

# Librerias necesarias para el correcto funcionamiento de los algoritmos.

import sys
import pickle
from os.path import join, dirname, abspath
dir_abs = join(dirname(abspath(__file__)))
dir_modul = join(dir_abs, 'scripts')
sys.path.append(dir_modul)
from AmpAlgor import amp
from numpy import arange, sin, pi, eye, asarray, shape, sqrt
from numpy.linalg import inv
from numpy.random import randint
from numpy.fft import fft, fftshift
from matplotlib.pyplot import plot, show, figure, title

# Tamano de la senal
N = 256

# Numero de observaciones a tomar.
M = 32

# Frecuencia de dos senales sinosoidales.
k1 = 9
k2 = 30

n = arange(0,N)
x = sin(2*pi*(k1/N)*n)+sin(2*pi*(k2/N)*n)

f = (arange(-len(x)/2,len(x)/2))
title('Senal original')
plot(f,abs(fftshift(fft(x))))

# Creando dft matrix
B = fft(eye(N))
Binv=inv(B)*N/sqrt(M);

p = randint(0, N, size=M).reshape(-1,1)

y = x[p]

A = Binv[p,:].reshape(M,N)
fil = open('store_amp.pckl', 'wb')
pickle.dump([A,y], fil)
fil.close()
s = amp(A, y, 50)


figure()
title('Senal Recuperada')
plot(f,abs(fftshift(s)))
show()
