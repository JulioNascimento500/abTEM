import sys
import requests
#sys.path.insert(0,'/home/jcdn500/Documents/Apenunga_New/apenunga-master')
#sys.path.insert(0,'/home/jcdn500/Documents/abTEM')
import abtem
print(abtem.__file__)

#from abtem.potentials import PotentialArray
from abtem.waves import PlaneWave
from ase.io import read
from abtem.potentials import Potential

atoms = read('/home/jcdn500/Documents/abTEM/Test_USe/srtio3_110.cif')

wave = PlaneWave(energy=300e3, sampling=.05)

Potential(atoms, sampling=.05).project()

pw_exit_wave = wave.multislice(atoms,pbar=False)

#pw_exit_wave.write('data/srtio3_110_exit_wave.hdf5')