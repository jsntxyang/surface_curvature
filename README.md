# surface_curvature fermicurve1.0
A small program to calculate curvature of fermi surface from band data from simulations.

This version contains an object called fermicurve. This object can read numpy .npy file (N* N * N matrix) and use marching cube method to fine the isoenergy surface.
frmsf file from quantum espresso is also accepted but the method is unstable(and the energy unit is wrong). 

The crystle structure can only be fcc. Other unit cell methods will be available later.

In developing...
