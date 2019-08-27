#!/usr/bin/env python

# Random Demodulator Recovery Algorithm by Using Simulated Analog Data from LTSpice.
# Before using this script run rd_demo_ltsipce_gen script, run LTspice
# simulation and export vout.
# Based : Alexander López-Parrado (2017)
# Created : Juan camilo Montilla Orjuela, Deimer Andres Morales Herrera (2019)
# Contact : moralesdei@protonamil.com

# Librerias necesarias para el correcto funcionamiento de los algoritmos.
import scipy.io as sio
import math

# Cargamos el archivo con las variables de nuestro entorno de trabajo.
data = sio.loadmat('matlab/rd/environment.mat')

# Extraemos las variables para poderlas utilizarla en nuestro script.
locals().update(data)

# there are L Nyquist periods in one period of p(t) (for Tropp's analysis
# Tx=Tp or L=N)
L=N

# 1-st order RC low-pass filter
tau = 1/(2*math.pi*float(fc))
B = 1
A = [tau,1]
