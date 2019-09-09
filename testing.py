#!/usr/bin/env python

# Random Demodulator Recovery Algorithm by Using Simulated Analog Data from LTSpice.
# Before using this script run rd_demo_ltsipce_gen script, run LTspice
# simulation and export vout.
# Based : Alexander López-Parrado (2017)
# Created : Juan camilo Montilla Orjuela, Deimer Andres Morales Herrera (2019)
# Contact : moralesdei@protonamil.com

# Librerias necesarias para el correcto funcionamiento de los algoritmos.

import sys
from os.path import join, dirname, abspath
dir_abs = join(dirname(abspath(__file__)))
dir_modul = join(dir_abs, 'scripts')
sys.path.append(dir_modul)
from OmpAlgor import omp
from os import name, system
from numpy import pi, absolute, array, mean, zeros, shape, matrix, exp, diag, flipud, roll
from control.matlab import tf, c2d, tfdata
from scipy import signal, special, linalg
from scipy.io import loadmat
from operator import itemgetter
from csv import reader

system('cls' if name == 'nt' else 'clear')
print('\n')
print(' (☉_☉) Pruebas de algoritmos de reconstruccion (☉_☉)')
print(' Este progama es totalmente libre para su uso y distribucion.')
print('\n')

#Cargamos todos nuestra area de trabajo.
dir_environment = join(dir_abs, 'matlab', 'rd')
dir_signal = join(dir_abs, 'ltspice')

# Cargamos el archivo con las variables de nuestro entorno de trabajo.
mat_fname = join(dir_environment,'environment.mat')
data = loadmat(mat_fname, squeeze_me=True)
locals().update(data)

tspice = []
lpfoutputI = []
lpfoutputQ = []

# Cargamos el archivo con la salida de la senal del filtro pasabajos.
signal_fname = join(dir_signal,'test-hfa3101-rc-lpf.txt')
with open(signal_fname) as fileData:
    ReadData = reader(fileData, delimiter='\t')
    next(ReadData)
    for row in ReadData:
        tspice.append(row[0])
        lpfoutputI.append(row[2])
        lpfoutputQ.append(row[1])

tspice = array(list(map(float, tspice)))
lpfoutputI = array(list(map(float, lpfoutputI)))
lpfoutputQ = array(list(map(float, lpfoutputQ)))

# Ganancia del mixer.
MixerGain=8

lpfoutput = (lpfoutputI+1j*lpfoutputQ)
# there are L Nyquist periods in one period of p(t) (for Tropp's analysis
# Tx=Tp or L=N).
L=N

# 1-st order RC low-pass filter
# En este fragmento de codigo se hallo la respuesta al impulso del filtro RC
# Implementado analogamente.
tau = 1/(2*pi*fc)
B = 1
A = [tau,1]
lpf = tf(B,A)
lpf_d = c2d(lpf,1/(float(W)),'tustin')
[[Bd]],[[Ad]] = tfdata(lpf_d)

### Simulando la funcion impz de matlab.
x = zeros(25)
x[0] = 1
h = signal.lfilter(Bd,Ad,x)

# Finds closest time value to perform sampling
i_tstart, dt_start = min(enumerate(absolute(tspice-Td)), key=itemgetter(1))
lpfoutput = lpfoutput[i_tstart:-1]
tspice = tspice[i_tstart:-1]
lpfoutput=lpfoutput-mean(lpfoutput)
lpfoutput = lpfoutput/MixerGain

# Time to start sampling after dummy training signal
tstart=1/fsn-1/(2*W)+Td

# ADC simulation
z=zeros(M, dtype=complex)
print(' Adquiriendo las muestras a la tasa sub-Nyquist, espera un momento...')

for k in range(M):
   i_tstart, dt_start = min(enumerate(absolute(tspice-tstart)), key=itemgetter(1))
   z[k] = lpfoutput[i_tstart]
   tstart = tstart+1/fsn
print(' Muestras aquiridas.')

# Creando la base de representacion Psi, se puede mejorar su rendimiento.
print(' Creando la base de representacion por favor espera ...')
l = matrix(range(int(N)))
n = matrix(range(int(-N/2), int(N/2)))
Psi =  exp(1j*(2*pi/N)*l.T * n)
print(' Base de representacion construida.')

D = diag(S)
H=zeros((M,N))
H[0,0:int((N/M))]=flipud(h[:])
for i in range(1,M):
    H[i,:] = roll(H[i-1,:],(0,int(N/M)))
H = matrix(H)

# Matriz necesaria para la recuperacion de la senal,tetha.
print(' Generando la matriz para la recuperacion ...')
A = H*D*Psi
print(' Matriz necesaria para la recuperacion creada correctamente')

# Linea reservada para la invocacion de los algoritmos de recuperacion
s = omp(A,z,k)


