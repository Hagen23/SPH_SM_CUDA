#ifndef Parameters_cuh
#define Parameters_cuh

#include "device_launch_parameters.h"
#include "cuda_runtime.h"

struct Parameters
{
	float dt;		/// Delta time for the simulation
	
	int3 gridSize;	/// Grid size
	int	 cellSize;	/// Size of each cell in the grid

	/// Boundaries for the simulation
	float xmin;		
	float xmax;
	float ymin;
	float ymax;
	float zmin;
	float zmax;

	float mass;		/// Mass of each particle
	float h;		/// Core radius h
	float restDens;	/// Stand density
	float k;		/// ideal pressure formulation k; Stiffness of the fluid. The lower the value, the stiffer the fluid.
	float mu;		/// Viscosity

	/// SM parameters
	float alpha;
	float beta;
	bool quadraticMatch;
	bool volumeConservation;
	bool allowFlip;

	/// SPH SM parameters
	float velocity_mixing;
};

#endif