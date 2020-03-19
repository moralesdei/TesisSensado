#!usr/bin/env python
# encoding: utf-8
import sysv_ipc
from numpy import frombuffer, float32, dot, shape, squeeze, asarray, array, ravel
from subprocess import call
import array
import time

def mulhowking(m1, m2):


	shm_key_m1r = 0x1234;
	shm_key_m1i = 0x2345;
	shm_key_m2r = 0x3456;
	shm_key_m2i = 0x4567;
	shm_key_rer = 0x5678;
	shm_key_rei = 0x6789;

	mem_m1r = sysv_ipc.SharedMemory(shm_key_m1r, sysv_ipc.IPC_CREAT, size=4*shape(m1)[0]*shape(m1)[1])
	mem_m1i = sysv_ipc.SharedMemory(shm_key_m1i, sysv_ipc.IPC_CREAT, size=4*shape(m1)[0]*shape(m1)[1])
	mem_m2r = sysv_ipc.SharedMemory(shm_key_m2r, sysv_ipc.IPC_CREAT, size=4*shape(m2)[0]*shape(m2)[1])
	mem_m2i = sysv_ipc.SharedMemory(shm_key_m2i, sysv_ipc.IPC_CREAT, size=4*shape(m2)[0]*shape(m2)[1])
	mem_rer = sysv_ipc.SharedMemory(shm_key_rer, sysv_ipc.IPC_CREAT, size=4*shape(m1)[0]*shape(m2)[1])
	mem_rei = sysv_ipc.SharedMemory(shm_key_rei, sysv_ipc.IPC_CREAT, size=4*shape(m1)[0]*shape(m2)[1])


	m1r = m1.real.astype(float32).flatten('F')
	m1i = m1.imag.astype(float32).flatten('F')
	m2r = m2.real.astype(float32).flatten('F')
	m2i = m2.imag.astype(float32).flatten('F')

	m1r.tobytes()
	m1i.tobytes()
	m2r.tobytes()
	m2i.tobytes()

	mem_m1r.write(m1r)
	mem_m1i.write(m1i)
	mem_m2r.write(m2r)
	mem_m2i.write(m2i)

	mem_m1r.detach()
	mem_m1i.detach()
	mem_m2r.detach()
	mem_m2i.detach()

	call(["./main.out", str(shape(m1)[0]), str(shape(m2)[1]), str(shape(m1)[1])])

	resultado_real = mem_rer.read()
	resultado_imag = mem_rei.read()

	mem_rer.detach()
	mem_rei.detach()
	mem_rer.remove()
	mem_rei.remove()

	resr = frombuffer(resultado_real, dtype=float32)
	resi = frombuffer(resultado_imag, dtype=float32)

	resultado = asarray([(resr + 1j*resi)]).reshape([shape(m2)[1],shape(m1)[0]]).T

	return resultado
