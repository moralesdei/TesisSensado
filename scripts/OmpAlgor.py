#!/usr/bin/env python
# encoding: utf-8
# Algoritmo de reconstruccion de señales, utilizando sensado compresivo OMP
# Created : Juan camilo Montilla Orjuela, Deimer Andres Morales Herrera (2019)
# Contact : moralesdei@protonamil.com
from numpy import argmax, shape, zeros, dot
from numpy.linalg import norm, qr
from scipy.linalg import solve_triangular

def omp(A,b,k):
    # inicializamos las variables necesarias.
    At = lambda x: (dot(A.T,x)).conj()
    r = b
    Ar = At(r)
    # Numero de atomos.
    N = len(Ar)
    # Tamano de cada atomo.
    M = len(r)
    x = zeros((N,1), dtype=complex)
    indx_set = zeros((k,1))
    A_T = zeros((M,k), dtype=complex)
    A_T_nonorth = zeros((M,k), dtype=complex)

    # Aca empieza el 'el ciclo'
    for kk in range(k):

        # Encomtramos el nuevo atomo
        ind_new = argmax(abs(Ar))
        indx_set[kk] = ind_new

        atom_new = A[:,ind_new]
        A_T_nonorth[:,kk] = atom_new

        # Primero, ortogonaliza 'atom_new' contra todos los átomos anteriores
        for j in range(kk-1):
            aux = A_T[:,j]
            atom_new = atom_new - (dot(aux.T, atom_new)).conj() * aux

        # Segundo, normalizamos
        atom_new = atom_new/norm(atom_new)
        A_T[:,kk] = atom_new

        # Tercero, Solucionar el problema de los minimos cuadrados.
        x_T = (dot(A_T[:,0:kk+1].T,b)).conj()
        x[indx_set[0:kk+1].astype(int).T] = x_T

        # Cuarto, actualizar el residuo.
        r = b - (dot(A_T[:,0:kk+1], x_T)).conj()

        # Preparandose para el proximo golpe
        if kk < k:
            Ar = At(r)

    A = A_T_nonorth[:,0:kk+1]
    Q, R = qr(A)
    x_T = solve_triangular(R, dot(Q.T,b).conj(), lower=True)

    # Esta linea unicamente se creo con fines comparativos, no descomentar.
    # x_T = asarray([[-0.0188 + 0.0028j],[-0.0056 + 0.0159j],[-0.0175 - 0.0038j],[-0.0012 - 0.0152j]])

    x[indx_set[0:kk+1].astype(int).T] = x_T
    return x
