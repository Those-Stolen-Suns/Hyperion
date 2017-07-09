###### -- HYPERIO -- ######

### python code for reading in 3D spherical mesh density data (e.g. from FARGO3D) and
### outputting a set of hyperion scattering images

# imports

from hyperion.model import Model
from hyperion.util.constants import pc, lsun, au
from subprocess import call
import os
import matplotlib
matplotlib.use('Agg') ## Do not activate X window, important for Hyades not for Jupyter
import matplotlib.pyplot as plt
import numpy as np
from pylab import *
import math

# user needs to input all the important variables for the mesh

# AZIMUTHAL ZONES
nx = 320

# RADIAL ZONES
ny = 80 

# COLATERAL ZONES
nz = 40 

# ANGLE WIDTH
thickness = 0.3

# MINIMUM RADIUS
rmin = 0.5

# MAXIMUM RADIUS
rmax = 2.0 

# FRAME
p = 86


### CODE ###

# import fargo data

dat = 'fargo_data/gasdens'+str(p)+'.dat'

# create an array

data = np.fromfile(dat).reshape(nz,ny,nx)

# want nx,nz,ny

data = np.rollaxis(data, 2)

# Initialize model
m = Model()

# Set up spherical mesh
    
r = np.linspace(rmin*au, rmax*au, ny+1)
r = np.hstack([0., r])  # add cell wall at r=0 to allow the source NOTE you need the first radial density value to be zero
#print(r)
thetazones = np.round((pi/thickness)*nz)+1
theta = np.linspace(0, pi, thetazones+1)
phi = np.linspace(0., 2 * pi, nx+1)
m.set_spherical_polar_grid(r, theta, phi)
print(np.round((pi/thickness)*nz)+1)

# need to insert density grid into background data grid
# Find the minimum background value

data = np.asarray(data)
min = np.amin(data)
print("minimum = " + str(min))
max = np.amax(data)
print("maximum = " + str(max))
data_array = np.zeros((nx,int(thetazones),ny+1))
data_array.fill(min)
mean = np.mean(data)
print("mean = " + str(mean))
print("small = " + str(data.shape))
print("large = " + str(data_array.shape))
x = 0
y = 1
z = 190
data_array[x:x+data.shape[0], z:z+data.shape[1], y:y+data.shape[2]] = data
data_array = data_array*(1e-18)

# Add density grid 

m.add_density_grid(data_array, 'kmh94_3.1_full.hdf5')

# Add a point source in the center
s = m.add_point_source()
s.position = (0., 0., 0.)
s.luminosity = 1000 * lsun
s.temperature = 6000.

# Add multi-wavelength image for a single viewing angle
image = m.add_peeled_images(sed=False, image=True)
image.set_wavelength_range(20, 1., 1000.)
image.set_viewing_angles([40.], [40.])
image.set_image_size(400, 400)
image.set_image_limits(-2.5*au, 2.5*au, -2.5*au, 2.5*au)

# Set runtime parameters
m.set_n_initial_iterations(5)
m.set_raytracing(True)
m.set_n_photons(initial=4e6, imaging=4e7,
                raytracing_sources=1e7, raytracing_dust=1e7)

# Write out input file
m.write('input/hyperion2.rtin')
#m.run('fargo.rtout', mpi=True)


print("Done!")