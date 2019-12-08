#!/usr/bin/env python
# encoding: utf-8
# Algoritmo de reconstruccion de señales, utilizando sensado compresivo CoSaMP
# Created : Juan camilo Montilla Orjuela, Deimer Andres Morales Herrera (2019)
# Contact : moralesdei@protonamil.com
from numpy import argmax, argsort, shape, zeros, sort, dot, asarray
from numpy.linalg import lstsq, norm, qr
from scipy.linalg import solve_triangular

def CoSaMP(A,b,k):
    At = lambda x: (dot(A.T,x)).conj()
    r = b
    Ar = At(r)
    N = len(Ar) # Numero de atomos.
    M = len(r) # Tamano de los atomos.
    x = zeros((N,1), dtype=complex)
    ind_k = []

    for kk in range(k):
        y_sort = abs(Ar)
        y_sort = y_sort[argsort(y_sort[:,0])][::-1]
        cutoff = y_sort[k]
        cutoff = max(cutoff,1e-10)

        ind_new = []
        for count, element in enumerate(Ar):
            if abs(element) >= cutoff:
                ind_new.append(count)

        T = sort(asarray(list(set(ind_new).union(ind_k))))

        Q, R = qr(A[:,T])
        x_T = solve_triangular(R, dot(Q.T,b).conj(), lower=False)

        cutoff = findCutoff(x_T, k)
        Tk = []
        for count, element in enumerate(x_T):
            if abs(element) >= cutoff:
                Tk.append(count)
        Tk = sort(asarray(Tk))
        ind_k = T[Tk]
        x = 0*x;
        x[ind_k] = x_T[Tk]

        Q, R = qr(A[:,ind_k])
        x_T2 = solve_triangular(R, dot(Q.T,b).conj(), lower=False)
        x[ind_k] = x_T2
        r = b - (dot(A[:,ind_k], x_T[Tk])).conj()

        if kk < k:
            Ar = At(r)
    return x

def findCutoff(x,k):
    x = abs(x)
    x = x[argsort(x[:,0])][::-1]
    if k > len(x):
        return x[-1,0] * 0.999
    else:
        return x[k-1,0]
