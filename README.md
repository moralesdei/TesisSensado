# Recuperacion de señales muestreadas a una tasa sub-Nyquist

> Creado por :
> Juan Camilo Montilla Orjuela.
> Deimer Andres Morales Herrera.

El presente proyecto va a implementar la reconstruccion de señales muestreadas a una tasa sub-Nyquist, Para recuperar dichas señales nos debemos apoyar en el sensado compresivo **(CS)** por sus siglas en ingles, esta teoria nos presenta varios algoritmos utilizados para llevar a cabo esta reconstruccion, entre los cuales podemos encontrar el **OMP**, **AMP** y **CoSaMP**. Aca vamos a describirlos en lenguaje python y posteriormente se les realizara una mejora en hadware que se describira en opencl e implementara en una FPGA.

Los algortimos estan individualmente en archivos en el folder **scripts**.

## Para ejecutar los algoritmos se recomienda :

* Ubicarse en el folder principal del proyecto.

* Instalar los correspondientes modulos de python.

	- matplotlib
	- scipy
	- numpy
	- control

* Darle permisos de ejecucion al archivo **testing.py**.
```
sudo chmod +x testing.py
```
* Ejecutar el archivo.
```
./testing.py
```
* Disfrutar la reconstruccion de la señal !!!!

