#!/usr/bin/env python
# encoding: utf-8
# Algoritmo de reconstruccion de señales, utilizando sensado compresivo AMP
# Created : Juan camilo Montilla Orjuela, Deimer Andres Morales Herrera (2019)
# Contact : moralesdei@protonamil.com

from numpy import spacing, arange, size, empty, shape, zeros, dot, mean, ones, median, sqrt, log
from numpy.linalg import norm

def amp(A,b,k):
    Af = lambda A,x: dot(A,x).conj()
    At = lambda A,x: (dot(A.T,x)).conj()
    n = size(b)
    lengthN = At(A,zeros((n,1)))
    N = size(lengthN)
    b = b - mean(b)
    colnormA = ones((N,1))
    norm_past = 10
    xall = zeros((N,k+1))
    mx = zeros((N,1))
    mz = b - Af(A,mx/colnormA)

    for kk in range(k):
        temp_z = (At(A,mz)/colnormA + mx)
        sigma_hat = 1/sqrt(log(2))*median(abs(temp_z))
        mx = softhold(temp_z,sigma_hat)
        etaderR,etaderI = dersofthold(temp_z,sigma_hat)
        mz = b - Af(A,mx/colnormA) + mz*(sum(etaderR) + sum(etaderI))/(2*n)
        normMZ = norm(mz)
        if norm_past < 1:
             break
        norm_past = normMZ
        print(normMZ)

        xall = mx/colnormA
    return xall

def softhold(x, lam):
    eta=(abs(x)> lam)*(abs(x)-lam)*(x)/abs(x+spacing(1))
    return eta

def dersofthold(x, lam):
    xr = x.real
    xi = x.imag
    absx3over2 = (xr**2+xi**2)**(3/2)+spacing(1)
    indicatorabsx = (xr**2+xi**2>lam**2)

    dxR1 = indicatorabsx*(1- lam*xi**2/absx3over2)
    dxR2 = lam*indicatorabsx*xr*xi/absx3over2

    dxI1 = lam*indicatorabsx*xr*xi/absx3over2
    dxI2 = indicatorabsx*(1- lam*xr**2/absx3over2)

    d0 = dxR1
    d1 = dxI2

    return d0, d1
