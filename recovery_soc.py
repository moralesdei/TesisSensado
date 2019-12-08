#!/usr/bin/env python
# encoding: utf-8
# Archivo de recuperacion de senales.
# Created : Juan camilo Montilla Orjuela, Deimer Andres Morales Herrera (2019)
# Contact : moralesdei@protonamil.com

# Librerias necesarias para el correcto funcionamiento de los algoritmos.

import sys
import pickle
from os.path import join, dirname, abspath
from numpy import shape
dir_abs = join(dirname(abspath(__file__)))
dir_modul = join(dir_abs, 'scripts')
sys.path.append(dir_modul)
from OmpAlgor import omp
from CoSaMPAlgor import CoSaMP
from AmpAlgor import amp

# Cargando la matriz y el vector.
f = open('store.pckl', 'rb')
[A,z] = pickle.load(f)
f.close()

# Linea reservada para la invocacion de los algoritmos de recuperacion
s = CoSaMP(A, z, 50)

# Guardando la senal recuperada.
f = open('signal_recovery.pckl', 'wb')
pickle.dump(s, f)
f.close()

