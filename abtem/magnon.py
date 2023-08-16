from abc import abstractmethod, ABCMeta
from tkinter import SEL
from typing import Mapping, Union, Sequence
from numbers import Number
import sys
import magnons

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.colors as colors


from sympy import pprint
from scipy.linalg import eig
import sympy as sym
import numpy as np

from ase import lattice, spacegroup
from ase.spacegroup import Spacegroup
from ase.dft.kpoints import monkhorst_pack

from collections.abc import Iterable

from numba import njit, prange, config

from ase import Atoms
from ase.data import covalent_radii
from ase.calculators.neighborlist import NeighborList

from copy import copy

import os

from abtem.structures import orthogonalize_cell
from ase.build import surface



@njit
def boson_dist(omegaN,T):
    Kb=8.617333e-2#8.617333e-5
    boson=1/(np.exp((omegaN)/(Kb*T))-1)

    if np.isnan(boson):
        return 0+0.0j
    else:
        return boson

@njit
def broadening(omega,omegaN,delta):
    #delta=0.5
    broadening = (1.0/np.sqrt(2.0*np.pi*(delta**2.0)))*np.exp(-(((omega-omegaN)*np.conjugate(omega-omegaN))/(2.0*(delta**2.0)))) 

    return broadening

@njit
def innerLoop(i,Estep,N,delta,exp_sum_j,XR_array,eVecsR,eVals_full3,omegaX,Vminus,Vplus,alpha,beta,TempFlag,T):
    vertical = np.empty((Estep), dtype=np.complex128)
    if TempFlag:
        for j in range(Estep):
            Total = 0
            for k in range(N):
                XR_array[:, k] = Vminus[alpha, k] * eVecsR[:N, k + N, i] + Vplus[beta, k] * eVecsR[N:, k + N, i]

                outer_product_XR = np.outer(np.conjugate(exp_sum_j[i, :] * XR_array[:, k]), (exp_sum_j[i, :] * XR_array[:, k]))

                Total += np.sum(outer_product_XR) * broadening(omegaX[j], eVals_full3[i, k + N], delta) * (boson_dist(omegaX[j],T)+1) 
            
            vertical[j] = Total
    else:   
        for j in range(Estep):
            Total = 0
            for k in range(N):
                XR_array[:, k] = Vminus[alpha, k] * eVecsR[:N, k + N, i] + Vplus[beta, k] * eVecsR[N:, k + N, i]

                outer_product_XR = np.outer(np.conjugate(exp_sum_j[i, :] * XR_array[:, k]), (exp_sum_j[i, :] * XR_array[:, k]))

                Total += np.sum(outer_product_XR) * broadening(omegaX[j], eVals_full3[i, k + N], delta) * 2
            
            vertical[j] = Total
    return vertical

#@njit(parallel=True)
def scattering_function_loop(Qgrid,Egrid,plotValues,N,T,eVecsL,eVecsR,eVals,exp_sum_j,omegaX,delta,Vplus,Vminus,alpha,beta,TempFlag=True):

    # print(TempFlag)
    # if TempFlag:
    #     for i in prange(Qgrid):
    #         for j in range(Egrid):
    #             Total=0
    #             for k in range(N):
    #                 XL = (Vminus[alpha,k]*eVecsL[:N,k+N,i] + Vplus[beta,k]*eVecsL[N:,k+N,i])
    #                 XR = (Vminus[alpha,k]*eVecsR[:N,k+N,i] + Vplus[beta,k]*eVecsR[N:,k+N,i])    

    #                 #if (boson_dist(eVals[i,k+N],T)+1)!=0.0+0.0j:

    #                 Total+=np.sum(np.outer(np.conjugate(exp_sum_j[i,:].T*XR),(exp_sum_j[i,:]*XR).T)) *  broadening(omegaX[j],eVals[i,k+N],delta) *(boson_dist(omegaX[j],T)+1) 

    #                 #if i==0 and j==0:
    #                 #    print('BD',(boson_dist(eVals[i,k+N],T)+1))
    #                 #    print('BDning',broadening(omegaX[j],eVals[i,k+N],delta))
    #             plotValues[j,i]=Total
    #else:
        # for i in prange(Qgrid):
        #     for j in range(Egrid):
        #         Total=0
        #         for k in range(N):
        #             XL = (Vminus[alpha,k]*eVecsL[:N,k+N,i] + Vplus[beta,k]*eVecsL[N:,k+N,i])
        #             XR = (Vminus[alpha,k]*eVecsR[:N,k+N,i] + Vplus[beta,k]*eVecsR[N:,k+N,i])    
                    
        #             Total+=np.sum(np.outer(np.conjugate(exp_sum_j[i,:].T*XR),(exp_sum_j[i,:]*XR).T)) *  broadening(omegaX[j],eVals[i,k+N],delta) * 2 
        #             Total+=np.sum(np.outer(np.conjugate(exp_sum_j[i,:].T*XL),(exp_sum_j[i,:]*XL).T)) *  broadening(omegaX[j],eVals[i,k+N],delta) * 2 

        #         plotValues[j,i]=Total
        #XL_array = np.empty((N, N), dtype=np.complex128)
        #XR_array = np.empty((N, N), dtype=np.complex128)
        #XR_array_1 = np.empty((N, N, Qgrid), dtype=np.complex128)
        #XR_array_2 = np.empty((N, N, Qgrid), dtype=np.complex128)
        

        #XR_array_1 = Vminus[alpha,:N]*eVecsR[:N, N:, :] +  Vplus[beta,:N]*eVecsR[N:, N:, :]
        
        
        #XR_array_2 = eVecsR[:N, :N, :] +  eVecsR[N:, :N, :]
    XR_array = np.empty((N, N), dtype=np.complex128)
    for i in range(Qgrid):
        plotValues[:, i] = innerLoop(i,Egrid,N,delta,exp_sum_j,XR_array,eVecsR,eVals,omegaX,Vminus,Vplus,alpha,beta,TempFlag,T)
        # for i in range(Qgrid):
        #     for j in range(Egrid):
        #         Total = 0

        #         for k in range(N):
        #             #XL_array[:, k] = Vminus[alpha, k] * eVecsL[:N, k + N, i] + Vplus[beta, k] * eVecsL[N:, k + N, i]
        #             #XR_array[:, k] = Vminus[alpha, k] * eVecsR[:N, k + N, i] + Vplus[beta, k] * eVecsR[N:, k + N, i]
        #             #XR_array_1[:, k] = eVecsR[:N, k + N, i] +  eVecsR[N:, k + N, i]
        #             #XR_array_2[:, k] = eVecsR[:N, k, i] +  eVecsR[N:, k, i]


        #             outer_product_XR = np.outer(np.conjugate(exp_sum_j[i, :] * XR_array_1[:, k,i]), (exp_sum_j[i, :] * XR_array_1[:, k,i]))
        #             #outer_product_XL = np.outer(np.conjugate(exp_sum_j[i, :].T * XL_array[:, k]), (exp_sum_j[i, :] * XL_array[:, k]))

        #             Total += np.sum(outer_product_XR) * broadening(omegaX[j], eVals[i, k + N], delta) * 2
        #             #Total += np.sum(np.real(outer_product_XL)) * broadening(omegaX[j], eVals[i, k + N], delta) * 2

        #         plotValues[j, i] = Total
    return plotValues


# def scattering_function_loop(Qgrid,Egrid,plotValues,N,T,eVecsL,eVecsR,eVals,exp_sum_j,omegaX,delta,Vplus,Vminus,alpha,beta):
        
#         exp_sum_j=exp_sum_j.T
#         for j in range(Egrid):
#             Total=np.zeros([49],dtype=complex)
#             for k in range(N):
#                 XL = (Vminus[alpha,k]*eVecsL[:N,k+N,:] + Vplus[beta,k]*eVecsL[N:,k+N,:])
#                 XR = (Vminus[alpha,k]*eVecsR[:N,k+N,:] + Vplus[beta,k]*eVecsR[N:,k+N,:])    

#                 #print('shape of XL:',np.shape(XL))
#                 #print('exp_sum_j:',np.shape(exp_sum_j))

#                 Total+=np.sum(np.outer(np.conjugate(exp_sum_j*XL),(exp_sum_j*XL).T)) *  broadening(omegaX[j],eVals[:,k+N],delta) #*(boson_dist(eVals[i,k+N],T)+1) 

#                 #if i==0 and j==0:
#                 #    print('BD',(boson_dist(eVals[i,k+N],T)+1))
#                 #    print('BDning',broadening(omegaX[j],eVals[i,k+N],delta))
#             plotValues[j,:]=Total
#         return plotValues




def cleanup(array):
    error_array=np.finfo(float).eps * np.sqrt(np.sum(array**2))
    #print(error_array)
    return(np.where(abs(array)<error_array,0,array ))

cmap = cm.get_cmap('jet')
rgba = cmap(0.0)
cmap.set_bad('white',1.)

def U(T,P):
    return np.stack(np.array([[np.cos(T)*np.cos(P),np.cos(T)*np.sin(P),-np.sin(T)],
                     [-np.sin(P),np.cos(P),np.zeros([len(P)])],
                     [np.sin(T)*np.cos(P),np.sin(T)*np.sin(P),np.cos(P)]]))

def V(magmoms):

    Vplus=U(magmoms[:,1],magmoms[:,2])[0,:] + 1.0j*U(magmoms[:,1],magmoms[:,2])[1,:]
    Vminus=U(magmoms[:,1],magmoms[:,2])[0,:] - 1.0j*U(magmoms[:,1],magmoms[:,2])[1,:]

    return Vplus,Vminus

def get_theta(mag):
    return sym.acos(mag[2]/np.linalg.norm(mag))

def get_phi(mag):
    if mag[0]>0:
        return sym.atan(mag[1]/mag[0])
    if mag[0]<0 and mag[1]>=0:
        return sym.atan(mag[1]/mag[0]) + sym.pi
    if mag[0]<0 and mag[1]<0:
        return sym.atan(mag[1]/mag[0]) - sym.pi
    if mag[0]==0 and mag[1]>0:
        return sym.pi/2
    if mag[0]==0 and mag[1]<0:
        return -sym.pi/2
    if mag[0]==0 and mag[1]==0:
        return 0

def phase_factor(q, neighList):

    Gamma = np.array([neighList[:,0]])*np.exp(-1.0j*np.matmul(q,neighList[:,1:].T))

    return np.sum(Gamma, axis=1)

def lorentzian_spectrum(energy_range, energies, gamma):
    spectrum = np.zeros_like(energy_range)
    
    for energy in energies:
        spectrum += 1 / (np.pi * gamma) * (gamma**2 / ((energy_range - energy)**2 + gamma**2))
    
    return spectrum

##########################################
#
# Function defining the tranformation for angles of diffent atoms in the magnetic unitcell.
#
#######################################

def scos(x): return sym.N(sym.cos(x))

def ssin(x): return sym.N(sym.sin(x))

def get_theta(mag):
    return sym.acos(mag[2]/np.linalg.norm(mag))

def get_phi(mag):
    if mag[0]>0:
        return sym.atan(mag[1]/mag[0])
    if mag[0]<0 and mag[1]>=0:
        return sym.atan(mag[1]/mag[0]) + sym.pi
    if mag[0]<0 and mag[1]<0:
        return sym.atan(mag[1]/mag[0]) - sym.pi
    if mag[0]==0 and mag[1]>0:
        return sym.pi/2
    if mag[0]==0 and mag[1]<0:
        return -sym.pi/2
    if mag[0]==0 and mag[1]==0:
        return 0


def Fzz(Atoms,r,s):
        
    theta_r = Atoms._atoms.get_initial_magnetic_moments()[r,1]
    phi_r = Atoms._atoms.get_initial_magnetic_moments()[r,2]
    
    theta_s = Atoms._atoms.get_initial_magnetic_moments()[s,1]
    phi_s = Atoms._atoms.get_initial_magnetic_moments()[s,2]
    
    SinProd=ssin(theta_r)*ssin(theta_s)
    CosProd=scos(theta_r)*scos(theta_s)
    
    CosDiff=scos(phi_r-phi_s)
    
    return float(SinProd*CosDiff + CosProd)


def G1(Atoms,r,s):
        
    theta_r = Atoms._atoms.get_initial_magnetic_moments()[r,1]
    phi_r = Atoms._atoms.get_initial_magnetic_moments()[r,2]
    
    theta_s = Atoms._atoms.get_initial_magnetic_moments()[s,1]
    phi_s = Atoms._atoms.get_initial_magnetic_moments()[s,2]
    
    SinProd=ssin(theta_r)*ssin(theta_s)
    CosProd=scos(theta_r)*scos(theta_s)
    
    SinDiff=ssin(phi_r-phi_s)
    CosDiff=scos(phi_r-phi_s)
    
    CosPlus=scos(theta_r) + scos(theta_s)
    
    return complex(((CosProd + 1)*CosDiff) + SinProd - (1.0j*SinDiff*CosPlus))


def G2(Atoms,r,s):
       
    theta_r = Atoms._atoms.get_initial_magnetic_moments()[r,1]
    phi_r = Atoms._atoms.get_initial_magnetic_moments()[r,2]
    
    theta_s = Atoms._atoms.get_initial_magnetic_moments()[s,1]
    phi_s = Atoms._atoms.get_initial_magnetic_moments()[s,2]

    SinProd=ssin(theta_r)*ssin(theta_s)
    CosProd=scos(theta_r)*scos(theta_s)
    
    SinDiff=ssin(phi_r-phi_s)
    CosDiff=scos(phi_r-phi_s)
    
    CosMinus=scos(theta_r) - scos(theta_s)
    
    return complex(((CosProd - 1)*CosDiff) + SinProd - (1.0j*SinDiff*CosMinus))


def Ax(Atoms,Anisotropy,r):
    eta=Anisotropy[r][1]
    delta=Anisotropy[r][2]
    theta = Atoms._atoms.get_initial_magnetic_moments()[r,1]
    phi = Atoms._atoms.get_initial_magnetic_moments()[r,2]
    return float(ssin(eta)*scos(theta)*scos(delta-phi) -  \
           scos(eta)*ssin(theta))

def Ay(Atoms,Anisotropy,r):
    eta=Anisotropy[r][1]
    delta=Anisotropy[r][2]
    theta = Atoms._atoms.get_initial_magnetic_moments()[r,1]
    phi = Atoms._atoms.get_initial_magnetic_moments()[r,2]
    return float(ssin(eta)*ssin(delta-phi))
           
def Az(Atoms,Anisotropy,r):
    eta=Anisotropy[r][1]
    delta=Anisotropy[r][2]
    theta = Atoms._atoms.get_initial_magnetic_moments()[r,1]
    phi = Atoms._atoms.get_initial_magnetic_moments()[r,2]
    return float(ssin(eta)*ssin(theta)*scos(delta-phi) +  \
           scos(eta)*scos(theta))


def v_matrix_plus(Atoms,r,alpha):

    direction=['x','y','z']    
    
    if alpha not in direction:
        raise ValueError("the directions must be either the strings 'x' 'y' or 'z'")
    
    alpha=direction.index(alpha)
    
    theta_r = Atoms._atoms.get_initial_magnetic_moments()[r,1]
    phi_r = Atoms._atoms.get_initial_magnetic_moments()[r,2]
    
    U=np.array([[scos(theta_r)*scos(phi_r),scos(theta_r)*ssin(phi_r),-ssin(theta_r)],
                [-ssin(phi_r),scos(phi_r),0],
                [ssin(theta_r)*scos(phi_r),ssin(theta_r)*ssin(phi_r),scos(theta_r)]])
    
    return complex(U[0,alpha] + 1.0j*U[1,alpha])

def v_matrix_minus(Atoms,r,alpha):
 
    direction=['x','y','z']

    if alpha not in direction:
        raise ValueError("the directions must be either the strings 'x' 'y' or 'z'")

    
    alpha=direction.index(alpha)
    
    theta_r = Atoms._atoms.get_initial_magnetic_moments()[r,1]
    phi_r = Atoms._atoms.get_initial_magnetic_moments()[r,2]
    
    U=np.array([[scos(theta_r)*scos(phi_r),scos(theta_r)*ssin(phi_r),-ssin(theta_r)],
                [-ssin(phi_r),scos(phi_r),0],
                [ssin(theta_r)*scos(phi_r),ssin(theta_r)*ssin(phi_r),scos(theta_r)]])
    
    return complex(U[0,alpha] - 1.0j*U[1,alpha])
@njit
def lorentzian_distribution(energy, energy_ref, thickness):
    """
    Compute the value of the Lorentzian distribution at a given energy.

    Parameters:
    energy (float): The energy at which to evaluate the Lorentzian distribution.
    energy_ref (float): The reference energy (center) of the distribution.
    thickness (float): The thickness (scale) parameter of the distribution.

    Returns:
    float: The value of the Lorentzian distribution at the given energy.
    """
    return 1.0 / (np.pi * thickness * (1.0 + ((energy - energy_ref) / thickness) ** 2))
@njit
def bose_einstein_distribution(energy, chemical_potential, temperature):
    k_ev = 8.617333262145e-2  # Boltzmann constant in mev/K

    # Evaluate the Bose-Einstein distribution
    difference = energy - chemical_potential
    # if np.abs(difference) < 1e-12:  # Handle the case of small energy differences
    #     distribution = 1 / (k_ev * temperature)
    # else:
    distribution = 1 / (np.exp(difference / (k_ev * temperature)) - 1)

    return distribution


class AbstractMagnonInput(metaclass=ABCMeta):
    """Abstract base class for Magnon Input objects."""

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def generate_atoms(self):
        """
        Generate frozen phonon configurations.
        """
        pass

    def __iter__(self):
        return self.generate_atoms()

    @abstractmethod
    def __copy__(self):
        pass

    def copy(self):
        """
        Make a copy.
        """
        return copy(self)


class DummyMagnonInput(AbstractMagnonInput):
    """
    Dummy Magnon Input object.

    Generates the input Atoms object. Used as a stand-in for simulations without frozen phonons.

    Parameters
    ----------
    atoms: ASE Atoms object
        Generated Atoms object.
    """

    def __init__(self, atoms: Atoms):
        self._atoms = atoms.copy()

    def __len__(self):
        return 1

    def generate_atoms(self):
        yield self._atoms

    def __copy__(self):
        return self.__class__(self._atoms.copy())


class MagnonInput(AbstractMagnonInput):
  '''
    Information for magnon calculation.

    Gets ASE atoms object, with magnetic moments, interaction between them and k-gridGenerates atomic configurations for thermal diffuse scattering.
    Randomly displaces the atomic positions of an ASE Atoms object to emulate thermal vibrations.

    Parameters
    ----------
    atoms: ASE Atoms object
        Atoms with the average atomic configuration.

    interaction: numpy array 1D or 3D 
        Array 
    kpoints: numpy array 
        List of kpoints to be calculated.

    
    Stores
    ----------

  '''
  def __init__(self,
               atoms: Atoms,
               interaction: Union[float, Sequence[Union[float,int]]],
               anisotropies: Union[float,int,Sequence[Union[float,int]]],
               qpts: Sequence[Union[float,int]],
               Temperature: Union[float,int],
               inelastic_layer: int=None,#Union[float, Sequence[int]],
               interaction_override: Union[None,Sequence[Union[float,int]]] = None,
               z_periodic: bool = True,
               debugg: bool = False,
               orientation: Sequence = (0,0,1),
               layers: int=1,
               scale = (1,1,1)
               #num_configs: int,
               #sigmas: Union[float, Mapping[Union[str, int], float], Sequence[float]],
               #directions: str = 'xyz',
               ):

    self._atoms_multi = orthogonalize_cell(atoms)
    self._atoms_multi = surface(self._atoms_multi*scale, orientation, layers=layers, periodic=z_periodic)

    indices_to_remove = np.where(atoms.get_initial_magnetic_moments()[:,0] == 0)[0]

    atoms = atoms[np.logical_not(np.isin(np.arange(len(atoms)), indices_to_remove))]

    self._atoms = atoms

    self.M=len(np.unique(atoms.get_initial_magnetic_moments(),axis=0))

    if isinstance(interaction, float):
      # In case interaction is a single number, all interaction are the same but only first neighbours
      self.Js = np.full((self.M, self.M, 1), interaction)
      self.num_nei = 1

    if isinstance(interaction, Iterable):
      if len(np.shape(interaction)) == 1:
        # In case interaction is a list or 1D array, all interactions are the same and each entry on the list is a n'th neighbour
        self.Js = np.repeat(np.expand_dims(interaction, axis=0), self.M, axis=0)
        self.Js = np.reshape(self.Js, (self.M, self.M, -1))
        self.num_nei = len(interaction)
      else:
        # In case it is a (M, M, Number of next neighbours) array.
        if np.shape(interaction)[:2]==(self.M,self.M):
          self.Js=interaction
          self.num_nei = np.shape(interaction)[-1]
        else:
          raise RuntimeError('Missing or over specified Js, with respect to the number of different magnetic moments.')

    

    self._atoms.set_pbc((True, True, z_periodic))
    self._interaction = interaction
    self._anisotropies = anisotropies
    self._qpts = qpts
    self._z_periodic = z_periodic
    self.Temperature = Temperature
    self.inelastic_layer = inelastic_layer
    self.Total_types=np.unique(self._atoms.get_initial_magnetic_moments(),axis=0)
    ## Override

    self._interaction_override = np.zeros([len(self._atoms.positions),len(self._atoms.positions),self.num_nei])

    if interaction_override==None:
        pass
    else:
        for term in interaction_override:
            self._interaction_override[term[0],term[1],:len(term-2)] = term[2:]

    self._interaction_override=np.where(self._interaction_override==0.0, None,self._interaction_override)

  @property
  def atoms(self) -> Atoms:
      return self._atoms

  def __len__(self):
      return 1

  def generate_atoms(self):
      yield self._atoms

  def get_distance_vector(self,bondpair):
      Pos_Atom_1 = self._atoms.positions[bondpair[0]]
      Pos_Atom_2 = self._atoms.positions[bondpair[1]]+np.dot(bondpair[2],self._atoms.cell)
      distanceVector = Pos_Atom_2 - Pos_Atom_1
      #distanceVector[-1] = 0
      return distanceVector

  def get_distance_vector_scaled(self,bondpair):
      Pos_Atom_1 = self._atoms.get_scaled_positions()[bondpair[0]]
      Pos_Atom_2 = self._atoms.get_scaled_positions()[bondpair[1]]+bondpair[2]
      distanceVector = Pos_Atom_2 - Pos_Atom_1
      #distanceVector[-1] = 0
      return distanceVector

  def get_distance(self,bondpair):
          Pos_Atom_1 = self._atoms.positions[bondpair[0]]
          Pos_Atom_2 = self._atoms.positions[bondpair[1]]+np.dot(bondpair[2],self._atoms.cell)
          distanceVector = Pos_Atom_2 - Pos_Atom_1
          return np.linalg.norm(distanceVector)

  def get_neigb(self,step_size):

      Lists_of_Neigbours=[]
      List_of_bondpairs=[]

      radius=0

      ## Get all neighbours with increasing radius until read get far enought to fufill
      ## the required furthest neighbour.

      while len(List_of_bondpairs)<self.num_nei:
          radius+=step_size
          
          cutoffs = radius * (covalent_radii[self._atoms.numbers]/covalent_radii[self._atoms.numbers])
          cutoffs_skin = radius * (covalent_radii[self._atoms.numbers]/covalent_radii[self._atoms.numbers])
          
          nl = NeighborList(cutoffs=cutoffs, self_interaction=False,bothways=True)
          nl_skin = NeighborList(cutoffs=cutoffs_skin, self_interaction=False,bothways=True)
          
          nl.update(self._atoms)
          nl_skin.update(self._atoms)
          
          bondpairs = []
          for a in range(len(self._atoms)):
              indices, offsets = nl.get_neighbors(a)

              bondpairs.extend([(a, a2, offset)
                              for a2, offset in zip(indices, offsets)])
          if len(bondpairs)!=0: 
              if len(bondpairs) not in List_of_bondpairs:
                  List_of_bondpairs.append(len(bondpairs))
                  Lists_of_Neigbours.append(bondpairs)

      ## Remove duplicates            
                  
      for i in range(len(List_of_bondpairs)-1,0,-1):
          t_delete=[]
          for num,bond1 in enumerate(Lists_of_Neigbours[i]): 
              for bond2 in Lists_of_Neigbours[i-1]:
                  if bond1[0]==bond2[0] and bond1[1]==bond2[1] and (bond1[2]==bond2[2]).all():
                      t_delete.append(num)
                      
          Lists_of_Neigbours[i]=list(np.delete(np.array(Lists_of_Neigbours[i],dtype=object), t_delete,axis=0))     

      return Lists_of_Neigbours

  def Hamiltonian_Test(self,step_size=0.01,hermitian=False,anisotropy=False,debugger=False,badflag=False):
      
    if not hasattr(self, 'Lists_of_Neigbours'):
        self.Lists_of_Neigbours=self.get_neigb(step_size)

    ML=len(self._atoms.positions)


    self.list_Distances=[]
    for num,j in enumerate(self.Lists_of_Neigbours): 
        for bondpair in j: 
            self.list_Distances.append(round(self.get_distance(bondpair),3)) 
   

    self.list_Distances=np.sort(np.array(list(set(self.list_Distances))))

    H_main=np.zeros([ML,ML,len(self._qpts)],dtype=complex)
    H_main1=np.zeros([ML,ML,len(self._qpts)],dtype=complex)
    H_off1=np.zeros([ML,ML,len(self._qpts)],dtype=complex)
    H_off2=np.zeros([ML,ML,len(self._qpts)],dtype=complex)
    H_final=np.zeros([2*ML,2*ML,len(self._qpts)],dtype=complex)

    Total_types=np.unique(self._atoms.get_initial_magnetic_moments(),axis=0)

    for numNeib,neibList in enumerate(self.Lists_of_Neigbours):
        for interaction in neibList:

            self.DistanceInd=np.where(self.list_Distances==round(self.get_distance(interaction),3))[0][0]            
            
            r=interaction[0]#
            s=interaction[1]

            Sr=self._atoms.get_initial_magnetic_moments()[r][0]
            Ss=self._atoms.get_initial_magnetic_moments()[s][0]

            FzzM=Fzz(self,r,s)
            G1M=G1(self,r,s)
            G2M=G2(self,r,s)

            rJ=np.where((Total_types==self._atoms.get_initial_magnetic_moments()[r]).all(axis=1))[0][0]
            sJ=np.where((Total_types==self._atoms.get_initial_magnetic_moments()[s]).all(axis=1))[0][0]          

            Gamma=np.exp(-1.0j*np.dot(self._qpts,2*np.pi*self.get_distance_vector_scaled(interaction)))

            if debugger:
                #print(r,s,Gamma[r,s,numNeib,0])
                print(numNeib,interaction,rJ,sJ,self.DistanceInd,self.list_Distances[self.DistanceInd],self._interaction[rJ,sJ,self.DistanceInd])

            H_main[r,r,:]+=Ss*self._interaction[rJ,sJ,self.DistanceInd]*FzzM
            H_main[s,s,:]+=Sr*self._interaction[rJ,sJ,self.DistanceInd]*FzzM

            H_main[r,s,:]-=(np.sqrt(Sr*Ss)/2)*self._interaction[rJ,sJ,self.DistanceInd]*(Gamma*np.conj(G1M) + np.conj(Gamma)*G1M)

            H_main1[r,r,:]+=Ss*self._interaction[rJ,sJ,self.DistanceInd]*FzzM
            H_main1[s,s,:]+=Sr*self._interaction[rJ,sJ,self.DistanceInd]*FzzM

            H_main1[r,s,:]-=(np.sqrt(Sr*Ss)/2)*self._interaction[rJ,sJ,self.DistanceInd]*(Gamma*G1M + np.conj(Gamma)*np.conj(G1M))

            H_off1[r,s,:]-=(np.sqrt(Sr*Ss)/2)*self._interaction[rJ,sJ,self.DistanceInd]*(np.conj(Gamma)*np.conj(G2M) + Gamma*np.conj(G2M))
            H_off2[r,s,:]-=(np.sqrt(Sr*Ss)/2)*self._interaction[rJ,sJ,self.DistanceInd]*(np.conj(Gamma)*G2M + Gamma*G2M)

    if hermitian:
        for i in range(len(self._qpts)):
            H_final[:,:,i]=np.block([[H_main[:,:,i],H_off1[:,:,i]],
                                    [H_off2[:,:,i],H_main1[:,:,i]]])
    else:
        for i in range(len(self._qpts)):
            H_final[:,:,i]=np.block([[H_main[:,:,i],-H_off1[:,:,i]],
                                    [H_off2[:,:,i],-(H_main1[:,:,i])]])

    # if badflag:
    #     return H_final, self.orientationEach
    # else:
    #     return H_final
    return H_final/4


  def Hamiltonian(self,step_size=0.01,hermitian=False,anisotropy=False,debugger=False,badflag=False):

    
    if not hasattr(self, 'Lists_of_Neigbours'):
        self.Lists_of_Neigbours=self.get_neigb(step_size)

    # self.zNum=self._atoms.get_scaled_positions()[:,2]
    # self.zNum=[float(i) for i in self.zNum]
    # self.N_dict = {i:list(self.zNum).count(i) for i in self.zNum}
    # self.zNum=np.array([*sorted(set(self.zNum))])

    # LayersDictionary={}
    # for i in range(len(self.zNum)):
    #     LayersDictionary[i]=np.array([])

    # for num,i in enumerate(self._atoms.get_scaled_positions()):
    #     Temp=np.where((self.zNum==float(i[-1])))[0][0]
    #     LayersDictionary[Temp]=np.append(LayersDictionary[Temp],self._atoms.get_initial_magnetic_moments()[num])

    # for i in range(len(self.zNum)):
    #     Term1=LayersDictionary[i]
    #     LayersDictionary[i]=np.reshape(Term1,(len(Term1)//3,3))
    #     LayersDictionary[i]=np.unique(LayersDictionary[i], axis=0)


    # self.orientationEach = np.array([LayersDictionary[i][j] for i in LayersDictionary.keys() for j,value in enumerate(LayersDictionary[i])])


    ML=len(self._atoms.positions)

    Gamma=np.zeros([ML,ML,len(self.Lists_of_Neigbours),len(self._qpts)],dtype=complex) #Gamma[r,s,numNeib,0]=GammaValue    Gamma[r,s,numNeib,1]+=z
    z=np.zeros([ML,ML,len(self.Lists_of_Neigbours)])
    for numNeib,neibList in enumerate(self.Lists_of_Neigbours):
        for interaction in neibList:
            
            r=interaction[0]#
            s=interaction[1]

            Gamma[r,s,numNeib,:]+=np.exp(-1.0j*np.dot(self._qpts,2*np.pi*self.get_distance_vector_scaled(interaction)))
            #Gamma[s,r,numNeib,:]+=np.exp(-1.0j*np.dot(self._qpts,2*np.pi*self.get_distance_vector_scaled(interaction)))

            print(r,s,numNeib,self.get_distance_vector_scaled(interaction))#interaction,Gamma[r,s,numNeib,:])

            #Gamma[r,s,numNeib,:]+=(np.exp(-1.0j*np.dot(Input_mag.get_distance_vector(interaction),np.transpose(2*np.pi*(np.matmul(Input_mag._qpts,np.linalg.inv(Input_mag._atoms.cell[:])))))))
            
            z[r,s,numNeib]+=1
            #z[s,r,numNeib]+=1

    
    #print(Gamma)     
    loop=np.shape(z)
    for i in range(loop[0]):
        for j in range(loop[1]):
            for k in range(loop[2]):
                if z[i,j,k]!=0:
                    Gamma[i,j,k,:]/=z[i,j,k]
                else:
                    continue
    self.z=z
    self.Gamma=Gamma


    H_main=np.zeros([loop[0],loop[1],len(self._qpts)],dtype=complex)
    H_main1=np.zeros([loop[0],loop[1],len(self._qpts)],dtype=complex)
    H_off1=np.zeros([loop[0],loop[1],len(self._qpts)],dtype=complex)
    H_off2=np.zeros([loop[0],loop[1],len(self._qpts)],dtype=complex)
    H_final=np.zeros([2*loop[0],2*loop[1],len(self._qpts)],dtype=complex)


    Total_types=np.unique(self._atoms.get_initial_magnetic_moments(),axis=0)

    for r in range(loop[0]):
        for s in range(loop[1]):
            for numNeib in range(loop[2]):

                Sr=self._atoms.get_initial_magnetic_moments()[r][0]
                Ss=self._atoms.get_initial_magnetic_moments()[s][0]
                
                FzzM=Fzz(self,r,s)
                G1M=G1(self,r,s)
                G2M=G2(self,r,s)

                rJ=np.where((Total_types==self._atoms.get_initial_magnetic_moments()[r]).all(axis=1))[0][0]
                sJ=np.where((Total_types==self._atoms.get_initial_magnetic_moments()[s]).all(axis=1))[0][0]          

                if debugger:
                    #print(r,s,Gamma[r,s,numNeib,0])
                    print(r,s,numNeib,z[r,s,numNeib],Sr,Ss,self._interaction[rJ,sJ,numNeib],(1-Gamma[r,r,numNeib,0]),FzzM,G1M,G2M)

                # if r==s:
                #     H_main[r,r,:]+=(1/2)*Sr*z[r,r,numNeib]*self._interaction[rJ,rJ,numNeib]*(1-Gamma[r,r,numNeib,:]) 
                #     H_main1[r,r,:]+=(1/2)*Sr*z[r,r,numNeib]*self._interaction[rJ,rJ,numNeib]*(1-Gamma[r,r,numNeib,:]) 
                # else:
                #     H_main[r,r,:]+=(1/2)*Ss*z[r,s,numNeib]*self._interaction[rJ,sJ,numNeib]*FzzM  
                #     H_main1[s,s,:]+=(1/2)*Sr*z[r,s,numNeib]*self._interaction[rJ,sJ,numNeib]*FzzM
                    
                    
                #     H_main[r,s,:]-=(1/4)*(np.sqrt(Sr*Ss))*z[r,s,numNeib]*self._interaction[rJ,sJ,numNeib]*(np.conj(Gamma[r,s,numNeib,:]))*G1M
                #     H_main1[r,s,:]-=(1/4)*(np.sqrt(Sr*Ss))*z[r,s,numNeib]*self._interaction[rJ,sJ,numNeib]*(np.conj(Gamma[r,s,numNeib,:]))*np.conj(G1M)
                    
                #     H_off2[r,s,:]-=(1/4)*(np.sqrt(Sr*Ss))*z[r,s,numNeib]*self._interaction[rJ,sJ,numNeib]*(np.conj(Gamma[r,s,numNeib,:]))*np.conj(G2M)
                #     H_off1[r,s,:]-=(1/4)*(np.sqrt(Sr*Ss))*z[r,s,numNeib]*self._interaction[rJ,sJ,numNeib]*(np.conj(Gamma[r,s,numNeib,:]))*G2M

                H_main[r,r,:]+=Ss*z[r,s,numNeib]*self._interaction[rJ,sJ,numNeib]*FzzM
                H_main[s,s,:]+=Sr*z[r,s,numNeib]*self._interaction[rJ,sJ,numNeib]*FzzM

                H_main[r,s,:]-=(np.sqrt(Sr*Ss)/2)*z[r,s,numNeib]*self._interaction[rJ,sJ,numNeib]*(Gamma[r,s,numNeib,:]*np.conj(G1M) + np.conj(Gamma[r,s,numNeib,:])*G1M)

                H_main1[r,r,:]+=Ss*z[r,s,numNeib]*self._interaction[rJ,sJ,numNeib]*FzzM
                H_main1[s,s,:]+=Sr*z[r,s,numNeib]*self._interaction[rJ,sJ,numNeib]*FzzM

                H_main1[r,s,:]-=(np.sqrt(Sr*Ss)/2)*z[r,s,numNeib]*self._interaction[rJ,sJ,numNeib]*(Gamma[r,s,numNeib,:]*G1M + np.conj(Gamma[r,s,numNeib,:])*np.conj(G1M))

                H_off1[r,s,:]-=(np.sqrt(Sr*Ss)/2)*z[r,s,numNeib]*self._interaction[rJ,sJ,numNeib]*(np.conj(Gamma[r,s,numNeib,:])*np.conj(G2M) + Gamma[r,s,numNeib,:]*np.conj(G2M))
                H_off2[r,s,:]-=(np.sqrt(Sr*Ss)/2)*z[r,s,numNeib]*self._interaction[rJ,sJ,numNeib]*(np.conj(Gamma[r,s,numNeib,:])*G2M + Gamma[r,s,numNeib,:]*G2M)
               
                # H_main[r,r,:]+=Sr*z[r,r,numNeib]*self._interaction[rJ,rJ,numNeib]*FzzM
                # H_main[s,s,:]+=Ss*z[s,s,numNeib]*self._interaction[sJ,sJ,numNeib]*FzzM
                
                # H_main[r,s,:]-=(np.sqrt(Sr*Ss)/2)*z[r,s,numNeib]*self._interaction[rJ,sJ,numNeib]*(Gamma[r,s,numNeib,:]*G1M + np.conj(Gamma[r,s,numNeib,:])*np.conj(G1M))
                
                # H_main1[r,r,:]+=Sr*z[r,r,numNeib]*self._interaction[rJ,rJ,numNeib]*FzzM
                # H_main1[s,s,:]+=Ss*z[s,s,numNeib]*self._interaction[sJ,sJ,numNeib]*FzzM
                
                # H_main1[r,s,:]-=(np.sqrt(Sr*Ss)/2)*z[r,s,numNeib]*self._interaction[rJ,sJ,numNeib]*(Gamma[r,s,numNeib,:]*G1M + np.conj(Gamma[r,s,numNeib,:])*np.conj(G1M))
                
                # H_off1[r,s,:]-=(np.sqrt(Sr*Ss)/2)*z[r,s,numNeib]*self._interaction[rJ,sJ,numNeib]*(Gamma[r,s,numNeib,:]*np.conj(G2M) + np.conj(Gamma[r,s,numNeib,:])*G2M)
                # H_off2[r,s,:]-=(np.sqrt(Sr*Ss)/2)*z[r,s,numNeib]*self._interaction[rJ,sJ,numNeib]*(np.conj(Gamma[r,s,numNeib,:])*G2M + Gamma[r,s,numNeib,:]*np.conj(G2M))
                
                pprint(H_main[:,:,0])
    if hermitian:
        for i in range(len(self._qpts)):
            H_final[:,:,i]=np.block([[H_main[:,:,i],H_off1[:,:,i]],
                                    [H_off2[:,:,i],H_main1[:,:,i]]])
    else:
        for i in range(len(self._qpts)):
            H_final[:,:,i]=np.block([[H_main[:,:,i],-H_off1[:,:,i]],
                                    [H_off2[:,:,i],-(H_main1[:,:,i])]])

    # if badflag:
    #     return H_final, self.orientationEach
    # else:
    #     return H_final
    return H_final/4

  def Hamiltonian_film(self,Lists_of_Neigbours=None,step_size=0.01,hermitian=False,anisotropy=False,debugger=False,badflag=False):

    if not hasattr(self, 'Lists_of_Neigbours'):
        self.Lists_of_Neigbours=self.get_neigb(step_size)

    self.zNum=self._atoms.get_scaled_positions()[:,2]
    self.zNum=[float(i) for i in self.zNum]
    self.N_dict = {i:list(self.zNum).count(i) for i in self.zNum}
    self.zNum=np.array([*sorted(set(self.zNum))])

    LayersDictionary={}
    for i in range(len(self.zNum)):
        LayersDictionary[i]=np.array([])

    for num,i in enumerate(self._atoms.get_scaled_positions()):
        Temp=np.where((self.zNum==float(i[-1])))[0][0]
        LayersDictionary[Temp]=np.append(LayersDictionary[Temp],self._atoms.get_initial_magnetic_moments()[num])

    for i in range(len(self.zNum)):
        Term1=LayersDictionary[i]
        LayersDictionary[i]=np.reshape(Term1,(len(Term1)//3,3))
        LayersDictionary[i]=np.unique(LayersDictionary[i], axis=0)

    self.orientationEach = np.array([LayersDictionary[i][j] for i in LayersDictionary.keys() for j,value in enumerate(LayersDictionary[i])])

    self.Total_types=np.unique(self._atoms.get_initial_magnetic_moments(),axis=0)


    self.M_list=[len(LayersDictionary[key]) for key in LayersDictionary.keys()]

    M_types=[]

    for key in LayersDictionary.keys():
        M_types.append([np.where((self.Total_types==item).all(axis=1))[0][0] for item in LayersDictionary[key]])


    N_list=[self.N_dict[key] for key in self.N_dict.keys()]

    self.list_Distances=[]
    for num,j in enumerate(self.Lists_of_Neigbours): 
        for bondpair in j: 
            self.list_Distances.append(round(self.get_distance(bondpair),2)) 
   

    self.list_Distances=np.array(list(set(self.list_Distances)))

    #M=len(self.Total_types)

    self.M_list=list(np.array(self.M_list))

    #ML=len(self.M_list)

    H_main=np.zeros([sum(self.M_list),sum(self.M_list),len(self._qpts)],dtype=complex)
    H_main1=np.zeros([sum(self.M_list),sum(self.M_list),len(self._qpts)],dtype=complex)
    H_off1=np.zeros([sum(self.M_list),sum(self.M_list),len(self._qpts)],dtype=complex)
    H_off2=np.zeros([sum(self.M_list),sum(self.M_list),len(self._qpts)],dtype=complex)
    H_final=np.zeros([2*sum(self.M_list),2*sum(self.M_list),len(self._qpts)],dtype=complex)

    nw_length=[len(term) for term in M_types]


    ML=len(self._atoms.positions)

    Gamma_matrix=np.zeros([ML,ML,len(self.Lists_of_Neigbours),len(self._qpts)],dtype=complex)

    S=self._atoms.get_initial_magnetic_moments()[:,0]

    for num,j in enumerate(self.Lists_of_Neigbours):
        #print(j)
        for i in j:
            
            self.DistanceInd=np.where(self.list_Distances==round(self.get_distance(i),2))[0][0]#np.where(self.list_Distances==round(get_distance(self,i),5))[0][0]
            #print(self.DistanceInd)
            ## Layer of atoms 
            Layer1=np.where((self.zNum==float(self._atoms.get_scaled_positions()[i[0],-1])))[0][0]
            Layer2=np.where((self.zNum==float(self._atoms.get_scaled_positions()[i[1],-1])))[0][0]

  
            r=np.where((self.Total_types==self._atoms.get_initial_magnetic_moments()[i[0]]).all(axis=1))[0][0]
            s=np.where((self.Total_types==self._atoms.get_initial_magnetic_moments()[i[1]]).all(axis=1))[0][0]           


            sumnw_length_i=sum(nw_length[:Layer1])
            sumnw_length_j=sum(nw_length[:Layer2])

            Mi=M_types[Layer1]
            Mj=M_types[Layer2]

            rn=Mi.index(r)
            sn=Mj.index(s)

            Sr=(S[(sumnw_length_i)+rn])
            Ss=(S[(sumnw_length_j)+sn])




            if self._interaction_override[(sumnw_length_i)+rn,(sumnw_length_j)+sn,num]==None:
                JMatrixValue=self.Js[r,s,self.DistanceInd]
            else:
                JMatrixValue=self._interaction_override[(sumnw_length_i)+rn,(sumnw_length_j)+sn,num]

            ######################

            z=1*(self.M_list[Layer1]/N_list[Layer1])

            Gamma=(np.exp(-1.0j*np.dot(self._qpts,2*np.pi*self.get_distance_vector_scaled(i))))*(self.M_list[Layer1]/N_list[Layer1])

            Gamma_matrix[i[0],i[1],num,:]+=Gamma
            if debugger:
                #print(i,r,s,Gamma,z,round(self.get_distance(i),5),self.DistanceInd)
                print(i[0],i[1],num,self.get_distance_vector_scaled(i))#interaction,Gamma[r,s,numNeib,:])
                #print((sumnw_length_i)+rn,(sumnw_length_j)+sn,z)
            #Gamma=(np.exp(-1.0j*np.dot(self.get_distance_vector(i),np.transpose(2*np.pi*(q/np.array([a,a,a]))))))*(self.M_list[Layer1]/N_list[Layer1])
            
            FzzM=Fzz(self,i[0],i[1])
            G1M=G1(self,i[0],i[1])
            G2M=G2(self,i[0],i[1])


            H_main[(sumnw_length_i)+rn,(sumnw_length_i)+rn,:]+=z*JMatrixValue*(Sr)*FzzM

            H_main[(sumnw_length_j)+sn,(sumnw_length_j)+sn,:]+=z*JMatrixValue*(Ss)*FzzM                       


            H_main[(sumnw_length_i)+rn,(sumnw_length_j)+sn,:]-=JMatrixValue*    \
            + (((np.sqrt((Sr*Ss))/2)*(Gamma*G1M))  \
            +  ((np.sqrt((Sr*Ss))/2)*(np.conj(Gamma)*np.conj(G1M))))

            ################################
            H_main1[(sumnw_length_i)+rn,(sumnw_length_i)+rn,:]+=z*JMatrixValue*(Sr)*FzzM

            H_main1[(sumnw_length_j)+sn,(sumnw_length_j)+sn,:]+=z*JMatrixValue*(Ss)*FzzM                           


            H_main1[(sumnw_length_i)+rn,(sumnw_length_j)+sn,:]-=JMatrixValue*    \
            + (((np.sqrt((Sr*Ss))/2)*(np.conj(Gamma)*np.conj(G1M)))  \
            +  ((np.sqrt((Sr*Ss))/2)*(Gamma*G1M)))

            
            #################################
            H_off1[(sumnw_length_i)+rn,(sumnw_length_j)+sn,:]-=JMatrixValue*    \
            + (((np.sqrt((Sr*Ss))/2)*(Gamma*np.conj(G2M)))  \
            +  ((np.sqrt((Sr*Ss))/2)*(np.conj(Gamma)*G2M))) 

            
            ################################
            H_off2[(sumnw_length_i)+rn,(sumnw_length_j)+sn,:]-=JMatrixValue*    \
            + (((np.sqrt((Sr*Ss))/2)*(np.conj(Gamma)*G2M))  \
            +  ((np.sqrt((Sr*Ss))/2)*(Gamma*np.conj(G2M)))) 


            pprint(H_main[:,:,0])

    self.Gamma=Gamma_matrix   
    if hermitian:
        for i in range(len(self._qpts)):
            H_final[:,:,i]=np.block([[H_main[:,:,i],H_off1[:,:,i]],
                                     [H_off2[:,:,i],H_main1[:,:,i]]])
    else:
        for i in range(len(self._qpts)):
            H_final[:,:,i]=np.block([[H_main[:,:,i],-H_off1[:,:,i]],
                                     [H_off2[:,:,i],-(H_main1[:,:,i])]])
            
    H_final=H_final/4

    if badflag:
        return H_final, self.orientationEach
    else:
        return H_final

####### Anisotropy ###########

  def HamiltonianK(self,step_size=0.01,hermitian=False):

    MML=len(self.atoms.get_initial_magnetic_moments())

    H_main=np.zeros([MML,MML,len(self._qpts)],dtype=complex)
    H_off1=np.zeros([MML,MML,len(self._qpts)],dtype=complex)
    H_off2=np.zeros([MML,MML,len(self._qpts)],dtype=complex)
    Hk=np.zeros([2*MML,2*MML,len(self._qpts)],dtype=complex)

    for num,magMon in enumerate(self.atoms.get_initial_magnetic_moments()):
        
        Sr=magMon[0]
        
        AxTerm=Ax(self,self._anisotropies,num)
        AyTerm=Ay(self,self._anisotropies,num)
        AzTerm=Az(self,self._anisotropies,num)    
        
        #H_main[num,num,:]-=(self._anisotropies[num][0]/2) * ( (Sr*(AxTerm**2)) + (Sr*(AyTerm**2)) - (2*(Sr**2)*(AzTerm**2)) )
        #H_off1[num,num,:]-=(self._anisotropies[num][0]/2) * ((Sr/2)*(AxTerm**2) - (Sr/2)*(AyTerm**2) +1.0j*(Sr/2)*(AxTerm*AyTerm) +1.0j*(Sr/2)*(AyTerm*AxTerm))
        H_main[num,num,:]-=(self._anisotropies[num][0]/2) * Sr * ( (AxTerm**2) + (AyTerm**2) - (2*(AzTerm**2)) )
        H_off1[num,num,:]-=(self._anisotropies[num][0]/2) * Sr * (AxTerm + 1.0j*AyTerm)**2
        H_off2[num,num,:]-=(self._anisotropies[num][0]/2) * Sr * (AxTerm - 1.0j*AyTerm)**2
        
        if hermitian:
            for i in range(len(self._qpts)):
                Hk[:,:,i]=np.block([[H_main[:,:,i],H_off1[:,:,i]],
                                    [H_off2[:,:,i],H_main[:,:,i]]])
        else:
            for i in range(len(self._qpts)):
                Hk[:,:,i]=np.block([[H_main[:,:,i],-H_off1[:,:,i]],
                                    [H_off2[:,:,i],-(H_main[:,:,i])]])  

    return Hk

####### Anisotropy film ###########

  def HamiltonianK_film(self,step_size=0.01,hermitian=False):

    MML=len(self.atoms.get_initial_magnetic_moments())

    H_main=np.zeros([MML,MML,len(self._qpts)],dtype=complex)
    H_off1=np.zeros([MML,MML,len(self._qpts)],dtype=complex)
    Hk=np.zeros([2*MML,2*MML,len(self._qpts)],dtype=complex)

    for num,magMon in enumerate(self.atoms.get_initial_magnetic_moments()):
        
        Sr=magMon[0]
        
        AxTerm=Ax(self,self._anisotropies,num)
        AyTerm=Ay(self,self._anisotropies,num)
        AzTerm=Az(self,self._anisotropies,num)    
        
        H_main[num,num,:]-=(self._anisotropies[num][0]/2) * ( (Sr*(AxTerm**2)) + (Sr*(AyTerm**2)) - (2*(Sr**2)*(AzTerm**2)) )
        H_off1[num,num,:]-=(self._anisotropies[num][0]/2) * ((Sr/2)*(AxTerm**2) - (Sr/2)*(AyTerm**2) +1.0j*(Sr/2)*(AxTerm*AyTerm) +1.0j*(Sr/2)*(AyTerm*AxTerm))
        
        if hermitian:
            for i in range(len(self._qpts)):
                Hk[:,:,i]=np.block([[H_main[:,:,i],np.conj(H_off1[:,:,i])],
                                    [H_off1[:,:,i],H_main[:,:,i]]])
        else:
            for i in range(len(self._qpts)):
                Hk[:,:,i]=np.block([[H_main[:,:,i],-np.conj(H_off1[:,:,i])],
                                    [H_off1[:,:,i],-(H_main[:,:,i])]])  

    return Hk


  def diagonalize_function(self,hamiltonian):
  
      n=len(hamiltonian[:,:,0])//2
  
      nx=np.shape(hamiltonian)[0]
      ny=np.shape(hamiltonian)[1]
      nz=np.shape(hamiltonian)[2]
      
      eVals_full=np.zeros((nz,nx),dtype=complex, order='F')
      eVecs_fullL=np.zeros((nx,ny,nz),dtype=complex, order='F')
      eVecs_fullR=np.zeros((nx,ny,nz),dtype=complex, order='F')
  
  
      for i in range(len(hamiltonian[0,0,:])):
          
          eVals,eVecsL,eVecsR =  eig(hamiltonian[:,:,i], left=True, right=True)
          idx = eVals.argsort()[::1] 
          
          
          eVals = eVals[idx]
          eVecsL = eVecsL[:,idx]
          eVecsR = eVecsR[:,idx]
      
          eVals_full[i,:]=eVals
          eVecs_fullL[:,:,i]=eVecsL
          eVecs_fullR[:,:,i]=eVecsR      
  
      
      return eVals_full,eVecs_fullL,eVecs_fullR    

  
  def spin_scattering_function(self,H,Emax=100,Emin=-100,Estep=1000,Direction='xx',broadening=0.1,num_processors=1):

    # Set the number of processors
    os.environ["NUMBA_NUM_THREADS"] = str(num_processors)
    config.NUMBA_NUM_THREADS = num_processors

    eVals_full,eVecs_fullL,eVecs_fullR = self.diagonalize_function(H)
    omega=np.linspace(Emin,Emax,Estep)

    X1=np.zeros(np.shape(eVecs_fullL),dtype=complex)
    # X2=np.zeros(np.shape(eVecs_fullL),dtype=complex)    

    for i in prange(np.shape(eVecs_fullL)[-1]):
        X1[:,:,i] = np.linalg.inv(eVecs_fullL[:,:,i])

    # for i in prange(np.shape(eVecs_fullL)[-1]):
    #     X2[:,:,i] = np.linalg.inv(eVecs_fullR[:,:,i])        

    SpinSpinPlus=np.zeros([Estep,len(self._qpts)],dtype=complex)
    SpinSpinMinus=np.zeros([Estep,len(self._qpts)],dtype=complex)

    print(Direction[0])

    shape_H = np.shape(H)[0]
    M = shape_H // 2

    print(M)
    # for Enum, En in enumerate(omega):
    #     Smatrix = np.zeros_like(SpinSpin[0, :])  # Initialize Smatrix for each Enum
    #     for r in range(shape_H // 2):
    #         for s in range(shape_H // 2):
    #             Wval1 = v_matrix_minus(self, r, Direction[0]) * X1[:, r, :] + v_matrix_plus(self, r, Direction[1]) * X1[:, r + M, :]
    #             Wval2 = v_matrix_minus(self, s, Direction[0]) * X1[:, s, :] + v_matrix_plus(self, s, Direction[1]) * X1[:, s + M, :]
    #             #Wconj = np.conj(v_matrix_minus(self, s, Direction[0]) * X1[:, s, :] + v_matrix_plus(self, s, Direction[1]) * X1[:, s + M, :])

    #             Smatrix += np.sum(Wval1*np.conj(Wval2),axis=0)    
    #             #Smatrix += np.sum(Wval*Wconj,axis=0) 
        
    #     SpinSpin[Enum, :] =  Smatrix * np.sum(lorentzian_distribution(En, eVals_full[:, :], broadening), axis=1) #*(boson_dist(abs(En),self.Temperature)+1)

    for Enum, En in enumerate(omega):
        SmatrixPlus = np.zeros_like(SpinSpinPlus[0, :])  # Initialize Smatrix for each Enum
        SmatrixMinus = np.zeros_like(SpinSpinMinus[0, :])  # Initialize Smatrix for each Enum
        for r in range(shape_H // 2):
            for s in range(shape_H // 2):
                # Wval1Plus =         v_matrix_minus(self, r, Direction[0]) * X1[:M, r    , :] + \
                #                     v_matrix_plus(self, r, Direction[0])  * X1[:M, r + M, :]
                # Wval2Plus = np.conj(v_matrix_minus(self, s, Direction[1]) * X1[:M, s    , :] + \
                #                     v_matrix_plus(self, s, Direction[1])  * X1[:M, s + M, :])

                # SmatrixPlus += np.sum(Wval1Plus*Wval2Plus,axis=0)    
                
                # Wval1Minus =         v_matrix_minus(self, r, Direction[0]) * np.conj(X1[M:, r + M, :]) + \
                #                      v_matrix_plus(self, r, Direction[0])  * np.conj(X1[M:, r    , :])
                # Wval2Minus = np.conj(v_matrix_minus(self, s, Direction[1])) * X1[M:, s + M, :] + \
                #              np.conj(v_matrix_plus(self, s, Direction[1]))  * X1[M:, s    , :]

                # SmatrixMinus += np.sum(Wval1Minus*Wval2Minus,axis=0)    

                Wval1Plus =         v_matrix_minus(self, r, Direction[0]) * X1[r, :M    , :] + \
                                    v_matrix_plus(self, r, Direction[0])  * X1[r + M,:M, :]
                Wval2Plus = np.conj(v_matrix_minus(self, s, Direction[1]) * X1[s, :M    , :] + \
                                    v_matrix_plus(self, s, Direction[1])  * X1[s + M, :M, :])

                SmatrixPlus += np.sum(Wval1Plus*Wval2Plus,axis=0)    
                
                Wval1Minus =         v_matrix_minus(self, r, Direction[0])  * np.conj(X1[r + M, M:, :]) + \
                                     v_matrix_plus(self, r, Direction[0])   * np.conj(X1[r, M:    , :])
                Wval2Minus = np.conj(v_matrix_minus(self, s, Direction[1])) * X1[s + M, M:, :] + \
                             np.conj(v_matrix_plus(self, s, Direction[1]))  * X1[s, M:    , :]

                SmatrixMinus += np.sum(Wval1Minus*Wval2Minus,axis=0)         
        SpinSpinPlus[Enum, :] =  SmatrixPlus * np.sum(lorentzian_distribution(En, eVals_full[:, M:], broadening), axis=1) *(boson_dist(abs(En),self.Temperature)+1)
        SpinSpinMinus[Enum, :] =  SmatrixMinus * np.sum(lorentzian_distribution(En, eVals_full[:, :M], broadening), axis=1) *(boson_dist(abs(En),self.Temperature)+1)

    SpinSpin=(SpinSpinPlus-SpinSpinMinus)*(bose_einstein_distribution(abs(omega), 0, self.Temperature)+1)[:, np.newaxis]#*(boson_dist(abs(omega),self.Temperature)+1)[:, np.newaxis]
    # for Enum,En in enumerate(omega):
    #     for n in range(np.shape(H)[0]):
    #         Smatrix=0
    #         for r in range(np.shape(H)[0]//2):
    #             for s in range(np.shape(H)[0]//2):
    #                 Wval=v_matrix_minus(self,r,Direction[0])*X2[n,r,:] + v_matrix_plus(self,r,Direction[1])*X2[n,r+self.M,:]
    #                 Wconj=np.conj(v_matrix_minus(self,s,Direction[0])*X1[n,s,:] + v_matrix_plus(self,s,Direction[1])*X1[n,s+self.M,:])

    #                 Smatrix+=(Wval*Wconj)

    #         SpinSpin[Enum,:] = SpinSpin[Enum,:] + (bose_einstein_distribution(abs(En), 0,self.Temperature)+1)*(Smatrix*lorentzian_distribution(En, eVals_full[:,n], broadening))

    return SpinSpin
  
  def spin_scattering_function_film(self,H,Emax=100,Emin=-100,Estep=1000,Direction='xx',broadening=0.1,TempFlag=True,num_processors=1):
    

    self.zNum=self._atoms.get_scaled_positions()[:,2]
    self.zNum=[float(i) for i in self.zNum]
    self.N_dict = {i:list(self.zNum).count(i) for i in self.zNum}
    self.zNum=np.array([*sorted(set(self.zNum))])

    LayersDictionary={}
    for i in range(len(self.zNum)):
        LayersDictionary[i]=np.array([])

    for num,i in enumerate(self._atoms.get_scaled_positions()):
        Temp=np.where((self.zNum==float(i[-1])))[0][0]
        LayersDictionary[Temp]=np.append(LayersDictionary[Temp],self._atoms.get_initial_magnetic_moments()[num])

    for i in range(len(self.zNum)):
        Term1=LayersDictionary[i]
        LayersDictionary[i]=np.reshape(Term1,(len(Term1)//3,3))
        LayersDictionary[i]=np.unique(LayersDictionary[i], axis=0)

    self.orientationEach = np.array([LayersDictionary[i][j] for i in LayersDictionary.keys() for j,value in enumerate(LayersDictionary[i])])

    Directions=['x','y','z']

    print('Diagonalizing H')
    self.eVals_full,self.eVecs_fullL,self.eVecs_fullR = self.diagonalize_function(H)
    print('Done Diagonalizing H')
    N = len(self.eVals_full[0,:])//2
    N1 = len(self.zNum)
    Qgrid = len(self._qpts)

    # Set the number of processors
    os.environ["NUMBA_NUM_THREADS"] = str(num_processors)
    config.NUMBA_NUM_THREADS = num_processors


    omegaX=np.linspace(Emin,Emax,Estep,dtype=complex)

    Vplus,Vminus=V(self._atoms.get_initial_magnetic_moments())#V(self.orientationEach)

    # a=self._atoms.cell[0,0]
    # b=self._atoms.cell[1,1]
    # c=self._atoms.cell[2,2]

    # Total=[]

    # for numi,i in enumerate(self.M_list):
    #     for j in range(i):
    #         Total.append(self.zNum[numi]*c)

    # Total=np.array(Total)

    # A = (np.zeros(len(Total)))
    # A = np.vstack((A,np.zeros(len(Total))))
    # A = np.vstack((A,Total))
    # A = np.transpose(A)

    A = np.array(self._atoms.get_scaled_positions())

    #A = A[A[:, -1].argsort()]
    #A[:,0] = 0
    #A[:,1] = 0
#np.dot(self._qpts,2*np.pi*self.get_distance_vector_scaled(i))
    exp_sum_j = np.exp(-1.0j*np.dot(self._qpts@np.linalg.inv(self._atoms.cell[:]),np.transpose(A)))
    #exp_sum_j = np.exp(-1.0j*np.dot(2*np.pi*self._qpts/np.array([a,b,c]),np.transpose(A)))

    #exp_sum_j = cleanup(exp_sum_j)

    alpha=Directions.index(Direction[0])
    beta=Directions.index(Direction[1])

    plotValues= np.zeros((Estep,Qgrid),dtype=complex,order='F')

    print('Calling Fortran')

    plotValues= scattering_function_loop(Qgrid,Estep,plotValues,N,self.Temperature,self.eVecs_fullL,self.eVecs_fullR,self.eVals_full,exp_sum_j,omegaX,broadening,Vplus,Vminus,Directions.index(Direction[0]),Directions.index(Direction[1]),TempFlag=TempFlag)

    #magnons.magnons_function(Qgrid,Estep,plotValues,N,2*N,self.Temperature,eVecs_fullL,eVals_full,exp_sum_j,omegaX,broadening,Vplus,Vminus,Directions.index(Direction[0]),Directions.index(Direction[1]))  #Directions.index(Direction[0]),Directions.index(Direction[1])

    print('Fortran done')

    #plotValues=(1/(2*len(self._atoms.positions)))*plotValues

    plotValues=(1/(2))*plotValues

    return plotValues

  def DOS(self,Emin=0,Emax=1000,Esize=1000,delta=1.0,qGrid: Union[int, Sequence[int]] = None):
    
    energy_range = np.linspace(Emin,Emax,Esize)

    #sg = Spacegroup(spacegroup.get_spacegroup(self._atoms, symprec=1e-12))
    #symmetryOP=sg.get_op()[0]

    if qGrid is not None:
        save_qpts=self._qpts

        if isinstance(qGrid, int):
            if qGrid%2==0:
                qs=monkhorst_pack((qGrid,qGrid,qGrid))[:((qGrid**3)//2)]
            else:
                qs=monkhorst_pack((qGrid,qGrid,qGrid))[:((qGrid**3)//2)+1]

        if isinstance(qGrid, Sequence):
            qs=monkhorst_pack(qGrid)
            if len(qs)%2==0:
                qs=qs[:len(qs)//2]
            else:
                qs=qs[:(len(qs)//2)+1]

        self._qpts=qs

        H=self.Hamiltonian(hermitian=False)

        eVals_full,eVecs_fullL,eVecs_fullR = self.diagonalize_function(H)

        spectrum=0
        for i in range(len(eVals_full[:,0])):
            energies = eVals_full[i,:].real
            # delta is the Width parameter of the Lorentzian peaks
            spectrum += 2*lorentzian_spectrum(energy_range, energies, delta)


        self._qpts=save_qpts

    else:
        H=self.Hamiltonian(hermitian=False)

        eVals_full,eVecs_fullL,eVecs_fullR = self.diagonalize_function(H)

        spectrum=0
        for i in range(len(eVals_full[:,0])):
            energies = eVals_full[i,:].real
            # delta is the Width parameter of the Lorentzian peaks
            spectrum += 2*lorentzian_spectrum(energy_range, energies, delta)

    return energy_range,spectrum,eVals_full

  def __copy__(self):
    return self.__class__(atoms=self.atoms.copy(),interaction=self._interaction,
                            anisotropies=self._anisotropies, qpts=self._qpts, z_periodic=self._z_periodic)
