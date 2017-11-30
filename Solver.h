/*
This class implements the SPH with CUDA.

@author Octavio Navarro
@version 1.0
*/
#pragma once
#ifndef Solver_h
#define Solver_h

#include "device_launch_parameters.h"
#include "cuda_runtime.h"
#include "Parameters.cuh"
#include "Math3D/m3Vector.h"
#include "m3Real.h"
#include "m3Matrix.h"
#include "m9Matrix.h"

#include <iostream>
#include <fstream>
#include <cstdlib>

#define GRID_SIZE 		64
#define NUM_PARTICLES	16384
#define BLOCK_SIZE		512

using namespace std;

class Solver
{
private:
	/// Host Data
	unsigned int 	num_particles;	/// Number of particles.
	size_t 			gridNum;		/// Number of cells.

	float3			*hpos;			/// Host position.
	float3			*hvel;			/// Host Velocity.		

	Parameters 		pa;				/// Parameters of the simulation.

	/// SM data
	m3Vector		*predicted_vel;		/// Predicted velocity: calculated with all forces but viscoelastic and pressure ones
	m3Vector		*mOriginalPos;		/// Original positions of the mesh points
	m3Vector		*mGoalPos;			/// Goal positions
	bool			*mFixed;			/// Whether de particle is fixed in place

	float3 			*hcorrected_vel;		/// Corrected velocity using SM
	float3			*dcorrected_vel;
	float3			*dscorrected_vel;
	float3			*dintermediate_vel;

	/// Device Data
	float3			*dpos;			/// Device position.
	float3			*dvel;			/// Device velocity.
	float3			*dspos;			/// Sorted positions.
	float3			*dsvel;			/// Sorted Velocity.
	float			*ddens;			/// Density.
	float3			*dforce;		/// Force to be applied on the particles. Acceleration.
	float			*dpress;		/// Pressure.

	unsigned int 	*dindex;		/// Array that stores the indices of particles in the grid.
	unsigned int 	*dhash;			/// Array that stores the hashes of particles in the grid.
	unsigned int 	*dcellStart;	/// Indicates where a hash starts, for neighbor search purposes.
	unsigned int 	*dcellEnd;		/// Indicates where a hash ends, for neighbor search purposes.

	/// Values for memory allocation
	size_t 			size1;			/// 1d float * num particles
	size_t 			size3;			/// 3d float * num particles

	void InitParticles();

	/// SM methods
	void calculate_corrected_velocity();
	void calculate_intermediate_velocity();	
	void calculate_velocity_correction();
	void projectPositions();
	void apply_external_forces();

public:
	
	Solver(unsigned int _num_particles);
	~Solver();

	void Update();

	inline float3* GetPos(){ return hpos; }
	inline m3Vector* GetGoalPos(){ return mGoalPos; }
};

#endif