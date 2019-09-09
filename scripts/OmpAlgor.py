#!/usr/bin/env python

# Algoritmo de reconstruccion de señales, utilizando sensado compresivo OMP
# Created : Juan camilo Montilla Orjuela, Deimer Andres Morales Herrera (2019)
# Contact : moralesdei@protonamil.com
from numpy import matrix, shape
from numpy.linalg import norm

def omp(A,b,k):

    # inicializamos las variables necesarias.
    A = matrix(A)
    b = matrix(b)
    At = lambda x: A.T * x
    r = b.H
    normR = norm(r)
    Ar = At(r)
    N = len(Ar)
    M = len(r)

