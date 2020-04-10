#!/usr/bin/env python
# encoding: utf-8
# Random Demodulator Recovery Algorithm by Using Simulated Analog Data from LTSpice.
# Before using this script run rd_demo_ltsipce_gen script, run LTspice
# simulation and export vout.
# Based : Alexander López-Parrado (2017)
# Created : Juan camilo Montilla Orjuela, Deimer Andres Morales Herrera (2019)
# Contact : moralesdei@protonamil.com

# Librerias necesarias para el correcto funcionamiento de los algoritmos.
import pickle
from numpy import arange, sin, pi
from numpy.fft import fftshift
from matplotlib.pyplot import plot, show, figure, title, xlabel, ylabel


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

fil = open('signal_recovery.pckl', 'rb')
s = pickle.load(fil)
fil.close()

figure()
title("Fourier Spectrum--Reconstructed signal")
xlabel("Hz")
ylabel("Magnitude")
plot(f,abs(fftshift(s)))
show()
