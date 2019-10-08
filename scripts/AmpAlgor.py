#!/usr/bin/env python
# encoding: utf-8
# Algoritmo de reconstruccion de señales, utilizando sensado compresivo AMP
# Created : Juan camilo Montilla Orjuela, Deimer Andres Morales Herrera (2019)
# Contact : moralesdei@protonamil.com

from numpy import spacing, arange, size, empty, shape

def amp(A,b,k):

def softhold(x, lam):
    eta=(abs(x)> lam)*(abs(x)-lam)*(x)/abs(x+spacing(1))
    return eta

def dersofthold(x, lam):
    xr = x.real
    xi = x.imag
    absx3over2 = (xr**2+xi**2)**(3/2)+spacing(1)
    indicatorabsx = (xr**2+xi**2>lam**2)

    dxR = empty(shape=[size(indicatorabsx), 2])
    dxR[:,0] = indicatorabsx*(1- lam*xi**2/absx3over2)
    dxR[:,1] = lam*indicatorabsx*xr*xi/absx3over2

    dxI = empty(shape=[size(indicatorabsx), 2])
    dxI[:,0] = lam*indicatorabsx*xr*xi/absx3over2
    dxI[:,1] = indicatorabsx*(1- lam*xr**2/absx3over2)

    d0 = dxR[:,0].reshape(-1,1)
    d1 = dxI[:,1].reshape(-1,1)

    return d0, d1



