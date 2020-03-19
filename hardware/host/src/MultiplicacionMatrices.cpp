// Multiplicacion de matrices, Este programa multiplica dos matrices complejas
// Copyright (C) 2020 moralesdei
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cstring>
#include <unistd.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/types.h>
#include <sys/time.h>
#include <string.h>
#include <time.h>
#include "CL/opencl.h"
#include "AOCL_Utils.h"
#include "matrixMult.h"

using namespace aocl_utils;

#define SHM_KEY_M1R 0x1234
#define SHM_KEY_M1I 0x2345
#define SHM_KEY_M2R 0x3456
#define SHM_KEY_M2I 0x4567
#define SHM_KEY_RER 0x5678
#define SHM_KEY_REI 0x6789

// OpenCL runtime configuration
static cl_platform_id platform = NULL;
static cl_device_id device = NULL;
static cl_context context = NULL;
static cl_command_queue queue = NULL;
static cl_kernel kernelM = NULL;
static cl_kernel kernelA = NULL;
static cl_kernel kernelR = NULL;
static cl_program program = NULL;
static cl_mem buff_m1r, buff_m1i, buff_m2r, buff_m2i, buff_rer, buff_rei;
static cl_mem buff_m1r_pad, buff_m1i_pad, buff_m2r_pad, buff_m2i_pad, buff_rer_pad, buff_rei_pad;
static float *m1rH, *m1iH, *m2rH, *m2iH, *rerH, *reiH;
static size_t local[2] = {BLOCK_SIZE, BLOCK_SIZE};

static cl_int status;

bool init();
void cleanup();
double get_wall_time();
void addpad(int, int, int, int, cl_mem, int, cl_mem);
void removepad(int, int, int, int, cl_mem, int, cl_mem);
void multiply(int, int, cl_mem, cl_mem, cl_mem, cl_mem, cl_mem, cl_mem, int, int);

int main(int argc, char *argv[]){


	size_t pad_c1f2, pad_col_m2, pad_fil_m1;
	size_t col_m2, fil_m1, c1f2;
	fil_m1 = atoi(argv[1]);
	col_m2 = atoi(argv[2]);
	c1f2 = atoi(argv[3]);

	int shmid_m1r,shmid_m2r, shmid_m1i, shmid_m2i, shmid_rer, shmid_rei;
  pad_fil_m1 = fil_m1;
  pad_col_m2 = col_m2;
	pad_c1f2 = c1f2;

	bool flag_m1 = false;
	bool flag_m2 = false;
	bool flag_m = false;

	for(int i=0;i<BLOCK_SIZE-1;i++){

			if(pad_fil_m1%20 != 0)
				{
					pad_fil_m1++;
					flag_m1 = true;
				}
			if(pad_col_m2%20 != 0)
				{
					pad_col_m2++;
					flag_m2 = true;
				}
			if(pad_c1f2%20 != 0)
				{
					pad_c1f2++;
					flag_m1 = true;
					flag_m2 = true;
				}
	}



	shmid_m1r = shmget(SHM_KEY_M1R, sizeof(float)*fil_m1*c1f2, 0644|IPC_CREAT);
	shmid_m1i = shmget(SHM_KEY_M1I, sizeof(float)*fil_m1*c1f2, 0644|IPC_CREAT);
	shmid_m2r = shmget(SHM_KEY_M2R, sizeof(float)*c1f2*col_m2, 0644|IPC_CREAT);
	shmid_m2i = shmget(SHM_KEY_M2I, sizeof(float)*c1f2*col_m2, 0644|IPC_CREAT);
	shmid_rer = shmget(SHM_KEY_RER, sizeof(float)*fil_m1*col_m2, 0644|IPC_CREAT);
	shmid_rei = shmget(SHM_KEY_REI, sizeof(float)*fil_m1*col_m2, 0644|IPC_CREAT);

	m1rH = (float *)shmat(shmid_m1r, NULL, 0);
	m1iH = (float *)shmat(shmid_m1i, NULL, 0);
	m2rH = (float *)shmat(shmid_m2r, NULL, 0);
	m2iH = (float *)shmat(shmid_m2i, NULL, 0);
	rerH = (float *)shmat(shmid_rer, NULL, 0);
	reiH = (float *)shmat(shmid_rei, NULL, 0);



	if(!init()){
			return -1;
	}

	// Create buffers
	buff_m1r = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR | CL_MEM_COPY_HOST_PTR, sizeof(float)*fil_m1*c1f2, (void*)m1rH, &status);
	checkError(status, "Failed to create buffer");
	buff_m1i = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR | CL_MEM_COPY_HOST_PTR, sizeof(float)*fil_m1*c1f2, (void*)m1iH, &status);
	checkError(status, "Failed to create buffer");
	buff_m2r = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR | CL_MEM_COPY_HOST_PTR, sizeof(float)*c1f2*col_m2, (void*)m2rH, &status);
	checkError(status, "Failed to create buffer");
	buff_m2i = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR | CL_MEM_COPY_HOST_PTR, sizeof(float)*c1f2*col_m2, (void*)m2iH, &status);
	checkError(status, "Failed to create buffer");
	buff_rer = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR, sizeof(float)*fil_m1*col_m2, NULL, &status);
	checkError(status, "Failed to create buffer");
	buff_rei = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR, sizeof(float)*fil_m1*col_m2, NULL, &status);
	checkError(status, "Failed to create buffer");


	if(flag_m1 && flag_m2)
		{
			buff_m1r_pad = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR, sizeof(float)*pad_fil_m1*pad_c1f2, NULL, &status);
			checkError(status, "Failed to create buffer");
			buff_m1i_pad = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR, sizeof(float)*pad_fil_m1*pad_c1f2, NULL, &status);
			checkError(status, "Failed to create buffer");
			buff_m2r_pad = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR, sizeof(float)*pad_c1f2*pad_col_m2, NULL, &status);
			checkError(status, "Failed to create buffer");
			buff_m2i_pad = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR, sizeof(float)*pad_c1f2*pad_col_m2, NULL, &status);
			checkError(status, "Failed to create buffer");
			buff_rer_pad = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR, sizeof(float)*pad_fil_m1*pad_col_m2, NULL, &status);
			checkError(status, "Failed to create buffer");
			buff_rei_pad = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR, sizeof(float)*pad_fil_m1*pad_col_m2, NULL, &status);
			checkError(status, "Failed to create buffer");

			addpad(pad_fil_m1, pad_c1f2, fil_m1, c1f2, buff_m1r, pad_fil_m1,  buff_m1r_pad);
			addpad(pad_fil_m1, pad_c1f2, fil_m1, c1f2, buff_m1i, pad_fil_m1,  buff_m1i_pad);
			addpad(pad_c1f2, pad_col_m2, c1f2, col_m2, buff_m2r, pad_c1f2,  buff_m2r_pad);
			addpad(pad_c1f2, pad_col_m2, c1f2, col_m2, buff_m2i, pad_c1f2,  buff_m2i_pad);
			multiply(pad_fil_m1, pad_col_m2, buff_m1r_pad, buff_m1i_pad, buff_m2r_pad, buff_m2i_pad, buff_rer_pad, buff_rei_pad, pad_fil_m1, pad_c1f2);
			removepad(pad_fil_m1, pad_col_m2, fil_m1, col_m2, buff_rer_pad, pad_fil_m1,  buff_rer);
			removepad(pad_fil_m1, pad_col_m2, fil_m1, col_m2, buff_rei_pad, pad_fil_m1,  buff_rei);

		}
	else if(flag_m1)
		{
			buff_m1r_pad = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR, sizeof(float)*pad_fil_m1*c1f2, NULL, &status);
			checkError(status, "Failed to create buffer");
			buff_m1i_pad = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR, sizeof(float)*pad_fil_m1*c1f2, NULL, &status);
			checkError(status, "Failed to create buffer");
			buff_rer_pad = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR, sizeof(float)*pad_fil_m1*col_m2, NULL, &status);
			checkError(status, "Failed to create buffer");
			buff_rei_pad = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR, sizeof(float)*pad_fil_m1*col_m2, NULL, &status);
			checkError(status, "Failed to create buffer");


			addpad(pad_fil_m1, c1f2, fil_m1, c1f2, buff_m1r, pad_fil_m1, buff_m1r_pad);
			addpad(pad_fil_m1, c1f2, fil_m1, c1f2, buff_m1i, pad_fil_m1, buff_m1i_pad);
			multiply(pad_fil_m1, col_m2, buff_m1r_pad, buff_m1i_pad, buff_m2r, buff_m2i, buff_rer_pad, buff_rei_pad, pad_fil_m1, c1f2);
			removepad(pad_fil_m1, col_m2, fil_m1, col_m2, buff_rer_pad, pad_fil_m1, buff_rer);
			removepad(pad_fil_m1, col_m2, fil_m1, col_m2, buff_rei_pad, pad_fil_m1, buff_rei);
		}
	else if(flag_m2)
		{
			buff_m2r_pad = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR, sizeof(float)*c1f2*pad_col_m2, NULL, &status);
			checkError(status, "Failed to create buffer");
			buff_m2i_pad = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR, sizeof(float)*c1f2*pad_col_m2, NULL, &status);
			checkError(status, "Failed to create buffer");
			buff_rer_pad = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR, sizeof(float)*fil_m1*pad_col_m2, NULL, &status);
			checkError(status, "Failed to create buffer");
			buff_rei_pad = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR, sizeof(float)*fil_m1*pad_col_m2, NULL, &status);
			checkError(status, "Failed to create buffer");


			addpad(c1f2, pad_col_m2, c1f2, col_m2, buff_m2r, c1f2, buff_m2r_pad);
			addpad(c1f2, pad_col_m2, c1f2, col_m2, buff_m2i, c1f2, buff_m2i_pad);
			multiply(fil_m1, pad_col_m2, buff_m1r, buff_m1i, buff_m2r_pad, buff_m2i_pad, buff_rer_pad, buff_rei_pad, fil_m1, pad_c1f2);
			removepad(fil_m1, pad_col_m2, fil_m1, col_m2, buff_rer_pad, fil_m1, buff_rer);
			removepad(fil_m1, pad_col_m2, fil_m1, col_m2, buff_rei_pad, fil_m1, buff_rei);

		}
	else
		{
			multiply(fil_m1, col_m2, buff_m1r, buff_m1i, buff_m2r, buff_m2i, buff_rer, buff_rei, fil_m1, c1f2);
		}

	// Read buffer output
	status = clEnqueueReadBuffer(queue, buff_rer, CL_TRUE, 0, sizeof(float) * fil_m1 * col_m2, (void*)rerH, 0, NULL, NULL);
	checkError(status, "Failed to read buffer");
	status = clEnqueueReadBuffer(queue, buff_rei, CL_TRUE, 0, sizeof(float) * fil_m1 * col_m2, (void*)reiH, 0, NULL, NULL);
	checkError(status, "Failed to read buffer");

	cleanup();



	shmdt(rerH);
	shmdt(reiH);
	shmdt(m1rH);
	shmdt(m1iH);
	shmdt(m2rH);
	shmdt(m2rH);

	shmctl(shmid_m1i, IPC_RMID, 0);
	shmctl(shmid_m1r, IPC_RMID, 0);
	shmctl(shmid_m2i, IPC_RMID, 0);
	shmctl(shmid_m2r, IPC_RMID, 0);



	return 0;

}

void cleanup() {

	if(buff_m1r) {
		clEnqueueUnmapMemObject(queue, buff_m1r, m1rH, 0, NULL, NULL);
		clReleaseMemObject(buff_m1r);
	}
	if(buff_m1i) {
		clEnqueueUnmapMemObject(queue, buff_m1i, m1iH, 0, NULL, NULL);
		clReleaseMemObject(buff_m1i);
	}
  if(buff_m2r) {
		clEnqueueUnmapMemObject(queue, buff_m2r, m2rH, 0, NULL, NULL);
		clReleaseMemObject(buff_m2r);
	}
if(buff_m2i) {
		clEnqueueUnmapMemObject(queue, buff_m2i, m2iH, 0, NULL, NULL);
		clReleaseMemObject(buff_m2i);
	}
	if(buff_rer) {
		clEnqueueUnmapMemObject(queue, buff_rer, rerH, 0, NULL, NULL);
		clReleaseMemObject(buff_rer);
	}
	if(buff_rei) {
		clEnqueueUnmapMemObject(queue, buff_rei, reiH, 0, NULL, NULL);
		clReleaseMemObject(buff_rei);
	}
	if(buff_m1r_pad) {
		clReleaseMemObject(buff_m1r_pad);
	}
	if(buff_m1i_pad) {
		clReleaseMemObject(buff_m1i_pad);
	}
	if(buff_m2r_pad) {
		clReleaseMemObject(buff_m2r_pad);
	}
	if(buff_m2i_pad) {
		clReleaseMemObject(buff_m2i_pad);
	}
	if(buff_rer_pad) {
		clReleaseMemObject(buff_rer_pad);
	}
	if(buff_rei_pad) {
		clReleaseMemObject(buff_rei_pad);
	}
	if(kernelM) {
		clReleaseKernel(kernelM);
	}
	if(kernelA) {
		clReleaseKernel(kernelA);
	}
	if(kernelR) {
		clReleaseKernel(kernelR);
	}
	if(program) {
		clReleaseProgram(program);
	}
	if(queue) {
		clReleaseCommandQueue(queue);
	}
	if(context) {
		clReleaseContext(context);
	}
}

bool init() {

	platform = findPlatform("Altera");

	if(platform == NULL) {
		printf("ERROR: Unable to find Altera OpenCL platform.\n");
		return false;
	}

	scoped_array<cl_device_id> devices;
	cl_uint num_devices;

	devices.reset(getDevices(platform, CL_DEVICE_TYPE_ALL, &num_devices));

	// We'll just use the first device.
	device = devices[0];

	// Create the context.
	context = clCreateContext(NULL, 1, &device, NULL, NULL, &status);
	checkError(status, "Failed to create context");

	// Create the command queue.
	queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
	checkError(status, "Failed to create command queue");

	// Create the program.
	std::string binary_file = getBoardBinaryFile("mult", device);
	//printf("Using AOCX: %s\n", binary_file.c_str());
	program = createProgramFromBinary(context, binary_file.c_str(), &device, 1);

	// Build the program that was just created.
	status = clBuildProgram(program, 0, NULL, "", NULL, NULL);
	checkError(status, "Failed to build program");

	// Create the kernel - name passed in here must match kernel name in the
	// original CL file, that was compiled into an AOCX file using the AOC tool
	const char *kernelM_name = "Matmul";  // Kernel name, as defined in the CL file
	kernelM = clCreateKernel(program, kernelM_name, &status);
	checkError(status, "Failed to create kernel");

	// Create the kernel - name passed in here must match kernel name in the
	// original CL file, that was compiled into an AOCX file using the AOC tool
	const char *kernelA_name = "paddingAddZeroes";  // Kernel name, as defined in the CL file
	kernelA = clCreateKernel(program, kernelA_name, &status);
	checkError(status, "Failed to create kernel");

	// Create the kernel - name passed in here must match kernel name in the
	// original CL file, that was compiled into an AOCX file using the AOC tool
	const char *kernelR_name = "paddingRemoveZeroes";  // Kernel name, as defined in the CL file
	kernelR = clCreateKernel(program, kernelR_name, &status);
	checkError(status, "Failed to create kernel");

	return true;
}

double get_wall_time(){
    struct timeval time;
    if (gettimeofday(&time,NULL)){
        //  Handle error
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}

void addpad(int f, int c, int arg0, int arg1, cl_mem arg2, int arg3, cl_mem arg4){

		size_t global[2] = {f, c};

		// Set kernel arguments add pad
		status = clSetKernelArg(kernelA, 0, sizeof(int), (void*)&arg0);
		checkError(status, "Failed to set kernel arg 0");
		status = clSetKernelArg(kernelA, 1, sizeof(int), (void*)&arg1);
		checkError(status, "Failed to set kernel arg 1");
		status = clSetKernelArg(kernelA, 2, sizeof(cl_mem), (void*)&arg2);
		checkError(status, "Failed to set kernel arg 2");
		status = clSetKernelArg(kernelA, 3, sizeof(int), (void*)&arg3);
		checkError(status, "Failed to set kernel arg 3");
		status = clSetKernelArg(kernelA, 4, sizeof(cl_mem), (void*)&arg4);
		checkError(status, "Failed to set kernel arg 5");

		// Launch the kernel add pad
		status = clEnqueueNDRangeKernel(queue, kernelA, 2, NULL, global, local, 0, NULL, NULL);
		checkError(status, "Failed to launch kernel");
	}

void removepad(int f, int c, int arg0, int arg1, cl_mem arg2, int arg3, cl_mem arg4){

		size_t global[2] = {f, c};
		// Set kernel arguments remove pad
		status = clSetKernelArg(kernelR, 0, sizeof(int), (void*)&arg0);
		checkError(status, "Failed to set kernel arg 0");
		status = clSetKernelArg(kernelR, 1, sizeof(int), (void*)&arg1);
		checkError(status, "Failed to set kernel arg 1");
		status = clSetKernelArg(kernelR, 2, sizeof(cl_mem), (void*)&arg2);
		checkError(status, "Failed to set kernel arg 2");
		status = clSetKernelArg(kernelR, 3, sizeof(int), (void*)&arg3);
		checkError(status, "Failed to set kernel arg 3");
		status = clSetKernelArg(kernelR, 4, sizeof(cl_mem), (void*)&arg4);
		checkError(status, "Failed to set kernel arg 5");

		// Launch the kernel remove pad
		status = clEnqueueNDRangeKernel(queue, kernelR, 2, NULL, global, local, 0, NULL, NULL);
		checkError(status, "Failed to launch kernel");
}
void multiply(int f, int c, cl_mem arg0, cl_mem arg1, cl_mem arg2, cl_mem arg3, cl_mem arg4, cl_mem arg5, int arg6, int arg7){

		size_t global[2] = {f, c};
		// Set kernel arguments multiply
		status = clSetKernelArg(kernelM, 0, sizeof(cl_mem), (void*)&arg0);
		checkError(status, "Failed to set kernel arg 0");
		status = clSetKernelArg(kernelM, 1, sizeof(cl_mem), (void*)&arg1);
		checkError(status, "Failed to set kernel arg 1");
		status = clSetKernelArg(kernelM, 2, sizeof(cl_mem), (void*)&arg2);
		checkError(status, "Failed to set kernel arg 2");
		status = clSetKernelArg(kernelM, 3, sizeof(cl_mem), (void*)&arg3);
		checkError(status, "Failed to set kernel arg 3");
		status = clSetKernelArg(kernelM, 4, sizeof(cl_mem), (void*)&arg4);
		checkError(status, "Failed to set kernel arg 4");
		status = clSetKernelArg(kernelM, 5, sizeof(cl_mem), (void*)&arg5);
		checkError(status, "Failed to set kernel arg 5");
		status = clSetKernelArg(kernelM, 6, sizeof(int), (void*)&arg6);
		checkError(status, "Failed to set kernel arg 6");
		status = clSetKernelArg(kernelM, 7, sizeof(int), (void*)&arg7);
		checkError(status, "Failed to set kernel arg 7");

		// Launch the kernel multiply
		status = clEnqueueNDRangeKernel(queue, kernelM, 2, NULL, global, local, 0, NULL, NULL);
		checkError(status, "Failed to launch kernel");
}
