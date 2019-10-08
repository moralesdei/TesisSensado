#!/usr/bin/env python
# encoding: utf-8
# Testing para el algoritmo AMP
# Based : Alexander López-Parrado (2017)
# Created : Juan camilo Montilla Orjuela, Deimer Andres Morales Herrera (2019)
# Contact : moralesdei@protonamil.com

# Librerias necesarias para el correcto funcionamiento de los algoritmos.

import sys
from os.path import join, dirname, abspath
dir_abs = join(dirname(abspath(__file__)))
dir_modul = join(dir_abs, 'scripts')
sys.path.append(dir_modul)
from AmpAlgor import amp
from numpy import arange, sin, pi, eye, asarray, shape, sqrt
from numpy.linalg import inv
from numpy.fft import fft, fftshift
from matplotlib.pyplot import plot, show

# Tamano de la senal
N = 256

# Numero de observaciones a tomar.
M = 32

# Frecuencia de dos senales sinosoidales.
k1 = 9
k2 = 30

n = arange(0,N)
x = sin(2*pi*(k1/N)*n)+sin(2*pi*(k2/N)*n)

# f = (arange(-len(x)/2,len(x)/2))
# plot(f,abs(fftshift(fft(x))))
# show()

# Creando dft matrix
B = fft(eye(N))
Binv=inv(B)*N/sqrt(M);
# Utilizar esta funcion en un futuro
# np.random.randint(2, size=10)

p =[ 59,  28, 231,  79, 202, 249, 122,  79, 101,  20,  13, 244,  69,177,  97, 205,  37, 232,  42,  50,  46, 129,  59,  72,   7, 200,43, 117, 145,  40, 142, 171];

p = asarray(p).reshape(1,-1)
y = x[p].T

A = Binv[p,:].reshape(M,N)
s = amp(A, y, 50)

f = (arange(-len(s)/2,len(s)/2))
plot(f,abs(fftshift(fft(s))))
show()
