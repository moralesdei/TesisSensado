
#include "../host/inc/matrixMult.h"

// kernel encargado de la multiplicacion de las matrices.
__kernel
__attribute((reqd_work_group_size(BLOCK_SIZE,BLOCK_SIZE,1)))
__attribute((num_simd_work_items(4)))
void Matmul(const __global float *m1r,
						const __global float *m1i,
						const __global float *m2r,
						const __global float *m2i,
						__global float *restrict rer,
						__global float *restrict rei,
						const int fil_m1,
						const int c1f2)
{
// Local ID index (offset within a block)
	int row = get_local_id(0);
	int col = get_local_id(1);

	// Global index
	int globalRow = BLOCK_SIZE * get_group_id(0) + row;
	int globalCol = BLOCK_SIZE * get_group_id(1) + col;

	__local float m1r_local[BLOCK_SIZE][BLOCK_SIZE];
	__local float m1i_local[BLOCK_SIZE][BLOCK_SIZE];
	__local float m2r_local[BLOCK_SIZE][BLOCK_SIZE];
	__local float m2i_local[BLOCK_SIZE][BLOCK_SIZE];


	float tmpr = 0.0f;
	float tmpi = 0.0f;

	const int numTiles = c1f2/BLOCK_SIZE;

	for (int t=0; t<numTiles; t++){

		const int tiledRow = BLOCK_SIZE*t + row;
		const int tiledCol = BLOCK_SIZE*t + col;

		m1r_local[col][row] = m1r[tiledCol*fil_m1 + globalRow];
		m1i_local[col][row] = m1i[tiledCol*fil_m1 + globalRow];
		m2r_local[col][row] = m2r[globalCol*c1f2  + tiledRow];
		m2i_local[col][row] = m2i[globalCol*c1f2  + tiledRow];

		barrier(CLK_LOCAL_MEM_FENCE);
		for(int k=0;k<BLOCK_SIZE;k++)
		{
			tmpr += m1r_local[k][row]*m2r_local[col][k] - m1i_local[k][row] * m2i_local[col][k];
			tmpi += m1r_local[k][row]*m2i_local[col][k] + m1i_local[k][row] * m2r_local[col][k];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	rer[globalCol * fil_m1 + globalRow] = tmpr;
	rei[globalCol * fil_m1 + globalRow] = tmpi * (-1);

}

// Kernel encargado de anadir el padding
__kernel
__attribute((reqd_work_group_size(BLOCK_SIZE,BLOCK_SIZE,1)))
__attribute((num_simd_work_items(4)))
void paddingAddZeroes(const int P,
											const int Q,
											const __global float* input,
											const int P_XL,
											__global float *restrict output) {

	// Thread identifiers
	const int tx = get_group_id(0)*BLOCK_SIZE + get_local_id(0); // 0..P_XL
	const int ty = get_group_id(1)*BLOCK_SIZE + get_local_id(1); // 0..Q_XL

		// Copy the input or pad a zero
		float value;
		if (tx < P && ty < Q) {
					value = input[ty*P + tx];
		}
		else{
			value = 0.0f;
		}

		// Store the result
		output[ty*P_XL + tx] = value;

}

// Kernel encargado de remover el padding.
__kernel
__attribute((reqd_work_group_size(BLOCK_SIZE,BLOCK_SIZE,1)))
__attribute((num_simd_work_items(4)))
void paddingRemoveZeroes(const int P,
													const int Q,
													const __global float* input,
													const int P_XL,
													__global float *restrict output) {

	// Thread identifiers
	const int tx = get_group_id(0)*BLOCK_SIZE + get_local_id(0); // 0..P_XL
	const int ty = get_group_id(1)*BLOCK_SIZE + get_local_id(1); // 0..Q_XL

	if (tx < P && ty < Q) {
		output[ty*P + tx] = input[ty*P_XL + tx];
	}

}
