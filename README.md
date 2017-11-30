# Implementation of Smoothed particle Hydrodynamics and Shape Matching using CUDA

A 3D implementation of "A velocity correcting method for volume preserving viscoelastic fluids." [1], using the Shape Matching [2] velocity correction scheme. For now, it is using the SPH algorithm presented by Müller et al.[5], instead of IISPH [3]. Will most likely implement Divergence-free SPH [4] since it requires less memory than IISPH, as well as being more stable. Using CUDA to speedup the SPH calculations.

[1] Takahashi, Tetsuya, Issei Fujishiro, and Tomoyuki Nishita. "A velocity correcting method for volume preserving viscoelastic fluids." Proceedings of the Computer Graphics International. 2014.

[2] Müller, Matthias, et al. "Meshless deformations based on shape matching." ACM transactions on graphics (TOG) 24.3 (2005): 471-478.

[3] Ihmsen, Markus, et al. "Implicit incompressible SPH." IEEE Transactions on Visualization and Computer Graphics 20.3 (2014): 426-435.

[4] Bender, Jan, and Dan Koschier. "Divergence-free SPH for incompressible and viscous fluids." IEEE transactions on visualization and computer graphics 23.3 (2017): 1193-1206.

[5] Müller, Matthias, David Charypar, and Markus Gross. "Particle-based fluid simulation for interactive applications." Proceedings of the 2003 ACM SIGGRAPH/Eurographics symposium on Computer animation. Eurographics Association, 2003.