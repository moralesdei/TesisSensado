#!/usr/bin/env python

# Random Demodulator Recovery Algorithm by Using Simulated Analog Data from LTSpice.
# Before using this script run rd_demo_ltsipce_gen script, run LTspice
# simulation and export vout.
# Based : Alexander López-Parrado (2017)
# Created : Juan camilo Montilla Orjuela, Deimer Andres Morales Herrera (2019)
# Contact : moralesdei@protonamil.com

# Librerias necesarias para el correcto funcionamiento de los algoritmos.
from numpy import pi
from control import  matlab
from scipy import signal, zeros, io

# Cargamos el archivo con las variables de nuestro entorno de trabajo.
data = io.loadmat('matlab/rd/environment.mat')

# Extraemos las variables para poderlas utilizarla en nuestro script.
locals().update(data)

# there are L Nyquist periods in one period of p(t) (for Tropp's analysis
# Tx=Tp or L=N).
L=N

# 1-st order RC low-pass filter
# En este fragmento de codigo se hallo la respuesta al impulso del filtro RC
# Implementado analogamente.
tau = 1/(2*pi*float(fc))
B = 1
A = [tau,1]
lpf = matlab.tf(B,A)
lpf_d = matlab.c2d(lpf,(1/100E6),'tustin')
Bd,Ad = matlab.tfdata(lpf_d)
Bd = Bd[0][0]
Ad = Ad[0][0]
x = zeros(25)
x[0] = 1
h = signal.lfilter(Bd,Ad,x)
print(h)
