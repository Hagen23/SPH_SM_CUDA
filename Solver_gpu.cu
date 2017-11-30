/*
This file contains the calls to the GPU kernels

@author Octavio Navarro
@version 1.0
*/
#include <iostream>
#include <string>

#include "thrust/sort.h"
#include "thrust/device_ptr.h"
#include "thrust/for_each.h"
#include "thrust/iterator/zip_iterator.h"

#include "Solver_gpu.cuh"

using namespace std;

extern "C"
{
	void computeGridSize(unsigned int n, unsigned int blockSize, unsigned int &numBlocks, unsigned int &numThreads)
	{
		numThreads = min(blockSize, n);
		numBlocks = (n + numThreads - 1) / numThreads;
	}

	void HandleError(cudaError_t err, string message)
	{
		if (cudaSuccess != err)
		{
			cout << message << endl;
			fprintf(stderr, "checkCudaErrors() Driver API error = %04d from file <%s>, line %i.\n", err, __FILE__, __LINE__);
			getLastCudaError(message.c_str());
        	exit(EXIT_FAILURE);
		}
	}

	void SetParameters(Parameters *p)
	{
		HandleError(cudaMemcpyToSymbol(para, p, sizeof(Parameters), 0, cudaMemcpyHostToDevice), "Failed to copy Symbol!");
		cudaDeviceSynchronize();
		getLastCudaError("SetParas execute failed!");
	}

	void CalHash(unsigned int* index, unsigned int* hash, float3* pos, unsigned int num_particles)
	{
		unsigned int numThreads, numBlocks;
		computeGridSize(num_particles, BLOCK_SIZE, numBlocks, numThreads);
		
		cudaCalHash <<<numBlocks, numThreads >>>(index, hash, pos, num_particles);
		
		cudaDeviceSynchronize();
		getLastCudaError("CalHash execute failed!");
	}
	void SortParticles(unsigned int *hash, unsigned int *index, unsigned int num_particles)
	{
		thrust::sort_by_key(thrust::device_ptr<unsigned int>(hash),
			thrust::device_ptr<unsigned int>(hash + num_particles),
			thrust::device_ptr<unsigned int>(index));
		
		cudaDeviceSynchronize();
		getLastCudaError("SortParticles execute failed!");
	}

	void ReorderDataAndFindCellStart(unsigned int* cellstart,
		unsigned int* cellend,
		float3* spos,
		float3* svel,
		float3* scorr_vel,
		unsigned int* hash,
		unsigned int* index,
		float3* pos,
		float3* vel,
		float3* corr_vel,
		unsigned int num_particles,
		unsigned int gridNum)
	{
		cudaMemset(cellstart, 0xffffffff, gridNum*sizeof(unsigned int));
		unsigned int memsize = sizeof(unsigned int)*(BLOCK_SIZE + 1);
		unsigned int numThreads, numBlocks;
		
		computeGridSize(num_particles, BLOCK_SIZE, numBlocks, numThreads);

		cudaReorderDataAndFindCellStart <<<numBlocks, numThreads, memsize>>>(cellstart, cellend, spos, svel, scorr_vel, hash, index, pos, vel, corr_vel, num_particles);

		cudaDeviceSynchronize();
		getLastCudaError("ReorderDataAndFindCellStart execute failed!");
	}

	void calcIntermediateVel(
		float3 *dintermediate_vel,
		float3 *scorrected_vel,
		float3* spos,
		float* dens,
		unsigned int* cellstart, 
		unsigned int* cellend, 
		unsigned int num_particles)
	{
		unsigned int numThreads, numBlocks;
		computeGridSize(num_particles, BLOCK_SIZE, numBlocks, numThreads);

		cudacalcIntermediateVel<<<numBlocks, numThreads>>>(dintermediate_vel, scorrected_vel, spos, dens, cellstart, cellend, num_particles);
		
		cudaDeviceSynchronize();
		getLastCudaError("calcIntermediateVel execute failed!");
	}

	void CalcDensityPressure(float* dens, float* press, unsigned int* cellstart, unsigned int* cellend, float3 *spos, unsigned int num_particles)
	{
		unsigned int numThreads, numBlocks;
		computeGridSize(num_particles, BLOCK_SIZE, numBlocks, numThreads);

		cudaCalcDensityPressure<<<numBlocks, numThreads>>>(dens, press, cellstart, cellend, spos, num_particles);

		cudaDeviceSynchronize();
		getLastCudaError("CalcDensity execute failed!");
	}

	void CalcForce(float3* force, float3* spos, float3* sintermediate_vel, float3* vel, float* press, 
		float* dens, unsigned int* index, unsigned int* cellstart, unsigned int* cellend, unsigned int num_particles)
	{
		unsigned int numThreads, numBlocks;
		computeGridSize(num_particles, BLOCK_SIZE, numBlocks, numThreads);

		cudaCalcForce <<<numBlocks, numThreads>>>(force, spos, sintermediate_vel, vel, press, dens, index, cellstart, cellend, num_particles);

		cudaDeviceSynchronize();
		getLastCudaError("CalcForce execute failed!");
	}

	void UpdateVelocityAndPosition(float3* pos, float3* vel, float3* force, unsigned int num_particles)
	{
		unsigned int numThreads, numBlocks;
		computeGridSize(num_particles, BLOCK_SIZE, numBlocks, numThreads);

		cudaUpdateVelocityAndPosition <<<numBlocks, numThreads >>>(pos, vel, force, num_particles);

		cudaDeviceSynchronize();
		getLastCudaError("UpdateVelocityAndPosition execute failed!");
	}

	void HandleBoundary(float3* pos, float3* vel, unsigned int num_particles)
	{
		unsigned int numThreads, numBlocks;
		computeGridSize(num_particles, BLOCK_SIZE, numBlocks, numThreads);

		cudaHandleBoundary <<<numBlocks, numThreads>>>(pos, vel, num_particles);

		cudaDeviceSynchronize();
		getLastCudaError("HandleBoundary execute failed!");
	}
}