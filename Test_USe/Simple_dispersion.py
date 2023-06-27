import sys
import requests
sys.path.insert(0,'/home/jcdn500/Documents/abTEM')
sys.path.insert(0,'/home/jcdn500/Documents/abTEM/abtem')
import numpy as np
from abtem import *
import matplotlib.pyplot as plt

from abtem.waves import PlaneWave
from ase.io import read, write
from abtem.potentials import Potential

import sympy as sym

from ase.dft.kpoints import get_special_points, get_bandpath
from ase.spacegroup import Spacegroup

from ase.visualize import view

import matplotlib.colors as colors
import matplotlib.cm as cm
from ase import lattice

def bcc_interaction_new(x,y,neib=1):
    interactions=[25.116584143611217,14.57904505050389,-0.708850521323485,-2.388932055445415,-0.518413067833594,
    0.16716176473001532,0.08252289651228567,0.05289929263607921,-0.6199797096948699,0.5184130678335925,-0.31527978411104274,
    -0.0655951228687412,0.31104784070015606,-0.19255342519533564,1.102421258535925,-0.15446593449735635,-0.2560325763586323,
    -0.01904374534898965,0.10791455697760481,0.02327568875987577,0.019043745348988443,-0.07829095310139955,
    0.0105798585272156,0.019043745348989047,-0.07405900969051282]  
    
    Js=np.zeros([x,y,neib])
    for i in range(x):
        for j in range(y):
            Js[i,j,:]=interactions[:neib] 
    
    return Js

atoms = read('./Fe_BCC_25ML.cif')




pi=sym.pi
magmoms=[[2.26,0,0]]*25
magmoms=[]
for i in range(25):
    magmoms.append([2.26,0.0,0.0])
atoms.set_initial_magnetic_moments(magmoms)

lat = lattice.BCC(4)
print(list(lat.get_special_points()))

path =  lat.bandpath('GHNGPN', npoints=400)

Input_mag=MagnonInput(atoms,
                      interaction=bcc_interaction_new(1,1,neib=1),
                      anisotropies=0,
                      qpts=path.kpts,
                      Temperature=300.0,
                      inelastic_layer=[0],
                      z_periodic = False)

H = Input_mag.Hamiltonian()

eVals_full,eVecs_fullL,eVecs_fullR = Input_mag.diagonalize_function(H)

plt.plot(eVals_full)

plt.show()