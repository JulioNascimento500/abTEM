from abc import abstractmethod, ABCMeta
from tkinter import SEL
from typing import Mapping, Union, Sequence
from numbers import Number
import sys
import magnons

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from scipy.linalg import eig
import sympy as sym
import numpy as np

from ase import lattice, spacegroup
from ase.spacegroup import Spacegroup
from ase.dft.kpoints import monkhorst_pack

from collections.abc import Iterable

from numba import njit

from ase import Atoms
from ase.data import covalent_radii
from ase.calculators.neighborlist import NeighborList

from copy import copy

def boson_dist(omegaN,T):
    Kb=8.617333e-2#8.617333e-5
    boson=1/(np.exp((omegaN)/(Kb*T))-1)

    if np.isnan(boson):
        return 0+0.0j
    # elif boson>maxval:
    #     return np.inf
    else:
        return boson


def broadening(omega,omegaN,delta):
    #delta=0.5
    broadening = (1.0/np.sqrt(2.0*np.pi*(delta**2.0)))*np.exp(-(((omega-omegaN)*np.conjugate(omega-omegaN))/(2.0*(delta**2.0)))) 

    return broadening


# def scattering_function_loop(Qgrid, Egrid, plotValues, N, T, eVecsL, eVecsR, eVals, exp_sum_j, omegaX, delta, Vplus, Vminus, alpha, beta):
#     for i in range(Qgrid):
#         for j in range(Egrid):
#             Total = 0
#             BD = boson_dist(eVals[i, N:], T) + 1  # Precompute boson_dist values outside the loop
#             broad = broadening(omegaX[j], eVals[i, N:], delta)  # Precompute broadening values outside the loop
            
#             for k in range(N):
#                 XL = (Vminus[alpha, k] * eVecsL[:N, k + N, i] + Vplus[beta, k] * eVecsL[N:, k + N, i])
#                 XR = (Vminus[alpha, k] * eVecsR[:N, k + N, i] + Vplus[beta, k] * eVecsR[N:, k + N, i])
                
#                 if BD[k] != 0.0 + 0.0j:
#                     Total += np.sum(np.outer(np.conjugate(exp_sum_j[i, :] * XL), (exp_sum_j[i, :] * XL).T)) * broad[k]
                
#                 if i == 0 and j == 0:
#                     print('BD', BD[k])
#                     print('BDning', broad[k])
                    
#             plotValues[j, i] = Total
            
#     return plotValues
def scattering_function_loop(Qgrid,Egrid,plotValues,N,T,eVecsL,eVecsR,eVals,exp_sum_j,omegaX,delta,Vplus,Vminus,alpha,beta):
    for i in range(Qgrid):
        for j in range(Egrid):
            Total=0
            for k in range(N):
                XL = (Vminus[alpha,k]*eVecsL[:N,k+N,i] + Vplus[beta,k]*eVecsL[N:,k+N,i])
                XR = (Vminus[alpha,k]*eVecsR[:N,k+N,i] + Vplus[beta,k]*eVecsR[N:,k+N,i])    

                if (boson_dist(eVals[i,k+N],T)+1)!=0.0+0.0j:

                    Total+=np.sum(np.outer(np.conjugate(exp_sum_j[i,:]*XL),(exp_sum_j[i,:]*XL).T)) *  broadening(omegaX[j],eVals[i,k+N],delta) #*(boson_dist(eVals[i,k+N],T)+1) 

                if i==0 and j==0:
                    print('BD',(boson_dist(eVals[i,k+N],T)+1))
                    print('BDning',broadening(omegaX[j],eVals[i,k+N],delta))
            plotValues[j,i]=Total
    return plotValues

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
    eta=Anisotropy[r,1]
    delta=Anisotropy[r,2]
    theta = Atoms._atoms.get_initial_magnetic_moments()[r,1]
    phi = Atoms._atoms.get_initial_magnetic_moments()[r,2]
    return float(ssin(eta)*scos(theta)*scos(delta-phi) -  \
           scos(eta)*ssin(theta))

def Ay(Atoms,Anisotropy,r):
    eta=Anisotropy[r,1]
    delta=Anisotropy[r,2]
    theta = Atoms._atoms.get_initial_magnetic_moments()[r,1]
    phi = Atoms._atoms.get_initial_magnetic_moments()[r,2]
    return float(ssin(eta)*ssin(delta-phi))
           
def Az(Atoms,Anisotropy,r):
    eta=Anisotropy[r,1]
    delta=Anisotropy[r,2]
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

def bose_einstein_distribution(energy, chemical_potential, temperature):
    k_ev = 8.617333262145e-5  # Boltzmann constant in ev/K

    # Evaluate the Bose-Einstein distribution
    difference = energy - chemical_potential
    # if np.abs(difference) < 1e-12:  # Handle the case of small energy differences
    #     distribution = 1 / (k_ev * temperature)
    #else:
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
               debugg: bool = False
               #num_configs: int,
               #sigmas: Union[float, Mapping[Union[str, int], float], Sequence[float]],
               #directions: str = 'xyz',
               ):

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

    self._atoms = atoms
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

  def Hamiltonian(self,Lists_of_Neigbours=None,step_size=0.01,hermitian=True,anisotropy=False,debugger=False):
    if Lists_of_Neigbours==None:
        Lists_of_Neigbours=self.get_neigb(step_size)

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

    a=self._atoms.cell[0,0]
    b=self._atoms.cell[1,1]
    c=self._atoms.cell[2,2]


    self.orientationEach = np.array([LayersDictionary[i][j] for i in LayersDictionary.keys() for j,value in enumerate(LayersDictionary[i])])

    self.Total_types=np.unique(self._atoms.get_initial_magnetic_moments(),axis=0)


    self.M_list=[len(LayersDictionary[key]) for key in LayersDictionary.keys()]

    M_types=[]

    for key in LayersDictionary.keys():
        M_types.append([np.where((self.Total_types==item).all(axis=1))[0][0] for item in LayersDictionary[key]])


    N_list=[self.N_dict[key] for key in self.N_dict.keys()]

    self.list_Distances=[]
    for num,j in enumerate(Lists_of_Neigbours): 
        for bondpair in j: 
            self.list_Distances.append(round(self.get_distance(bondpair),5)) 
   

    self.list_Distances=np.array(list(set(self.list_Distances)))

    #M=len(self.Total_types)

    self.M_list=list(np.array(self.M_list))

    H_main=np.zeros([self.M,self.M,len(self._qpts)],dtype=complex)
    H_main1=np.zeros([self.M,self.M,len(self._qpts)],dtype=complex)
    H_off1=np.zeros([self.M,self.M,len(self._qpts)],dtype=complex)
    H_off2=np.zeros([self.M,self.M,len(self._qpts)],dtype=complex)
    H_final=np.zeros([2*self.M,2*self.M,len(self._qpts)],dtype=complex)

    for num,j in enumerate(Lists_of_Neigbours):
        for i in j:
            
            Layer1=np.where((self.zNum==float(self._atoms.get_scaled_positions()[i[0],-1])))[0][0]
            Layer2=np.where((self.zNum==float(self._atoms.get_scaled_positions()[i[1],-1])))[0][0]

            r=np.where((self.Total_types==self._atoms.get_initial_magnetic_moments()[i[0]]).all(axis=1))[0][0]
            s=np.where((self.Total_types==self._atoms.get_initial_magnetic_moments()[i[1]]).all(axis=1))[0][0]
            
            Sr=self.Total_types[r,0]
            Ss=self.Total_types[s,0]
            
            FzzM=Fzz(self,i[0],i[1])
            G1M=G1(self,i[0],i[1])
            G2M=G2(self,i[0],i[1])

            z=1*(self.M_list[Layer1]/N_list[Layer1])
            
            Gamma=(np.exp(-1.0j*np.dot(self.get_distance_vector(i),np.transpose(2*np.pi*(self._qpts/self._atoms.cell.diagonal())))))*(self.M_list[Layer1]/N_list[Layer1])
            #print(G1M)

            if debugger:
                print(i,r,s,round(self.get_distance(i),5),self._interaction[r,s,num])
            
            H_main[r,r,:]+=z*self._interaction[r,s,num]*Ss*FzzM
            H_main[s,s,:]+=z*self._interaction[r,s,num]*Sr*FzzM
            H_main[r,s,:]-=self._interaction[r,s,num]*(np.sqrt(Sr*Ss)/2)*(np.conj(Gamma)*G1M + Gamma*np.conj(G1M))

            H_main1[r,r,:]+=z*self._interaction[r,s,num]*Ss*FzzM
            H_main1[s,s,:]+=z*self._interaction[r,s,num]*Sr*FzzM
            H_main1[r,s,:]-=self._interaction[r,s,num]*(np.sqrt(Sr*Ss)/2)*(np.conj(Gamma)*np.conj(G1M) + Gamma*G1M)

            H_off1[r,s,:]-=self._interaction[r,s,num]*(np.sqrt(Sr*Ss)/2)*(np.conj(Gamma)*np.conj(G2M) + Gamma*np.conj(G2M))

            H_off2[r,s,:]-=self._interaction[r,s,num]*(np.sqrt(Sr*Ss)/2)*(np.conj(Gamma)*G2M + Gamma*G2M)
        
            
    if hermitian:
        for i in range(len(self._qpts)):
            H_final[:,:,i]=np.block([[H_main[:,:,i],H_off1[:,:,i]],
                                     [H_off2[:,:,i],H_main1[:,:,i]]])
    else:
        for i in range(len(self._qpts)):
            H_final[:,:,i]=np.block([[H_main[:,:,i],-H_off1[:,:,i]],
                                     [H_off2[:,:,i],-(H_main1[:,:,i])]])
            
    H_final=H_final/4

    return H_final

  def Hamiltonian_film(self,Lists_of_Neigbours=None,step_size=0.01,hermitian=True,anisotropy=False,debugger=False):

    if Lists_of_Neigbours==None:
        Lists_of_Neigbours=self.get_neigb(step_size)

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

    a=self._atoms.cell[0,0]
    b=self._atoms.cell[1,1]
    c=self._atoms.cell[2,2]


    self.orientationEach = np.array([LayersDictionary[i][j] for i in LayersDictionary.keys() for j,value in enumerate(LayersDictionary[i])])

    self.Total_types=np.unique(self._atoms.get_initial_magnetic_moments(),axis=0)


    self.M_list=[len(LayersDictionary[key]) for key in LayersDictionary.keys()]

    M_types=[]

    for key in LayersDictionary.keys():
        M_types.append([np.where((self.Total_types==item).all(axis=1))[0][0] for item in LayersDictionary[key]])


    N_list=[self.N_dict[key] for key in self.N_dict.keys()]

    self.list_Distances=[]
    for num,j in enumerate(Lists_of_Neigbours): 
        for bondpair in j: 
            self.list_Distances.append(round(self.get_distance(bondpair),5)) 
   

    self.list_Distances=np.array(list(set(self.list_Distances)))

    #M=len(self.Total_types)

    self.M_list=list(np.array(self.M_list))

    #ML=len(self.M_list)

    H_main=np.zeros([sum(self.M_list),sum(self.M_list),len(self._qpts)],dtype=complex)
    H_main1=np.zeros([sum(self.M_list),sum(self.M_list),len(self._qpts)],dtype=complex)
    H_off1=np.zeros([sum(self.M_list),sum(self.M_list),len(self._qpts)],dtype=complex)
    H_off2=np.zeros([sum(self.M_list),sum(self.M_list),len(self._qpts)],dtype=complex)
    H_final=np.zeros([2*sum(self.M_list),2*sum(self.M_list),len(self._qpts)],dtype=complex)


    q = np.copy(self._qpts)
    q[:,2]=0

    nw_length=[len(term) for term in M_types]

    S=self._atoms.get_initial_magnetic_moments()[:,0]

    for num,j in enumerate(Lists_of_Neigbours):
        #print(j)
        for i in j:
            
            self.DistanceInd=np.where(self.list_Distances==round(self.get_distance(i),5))[0][0]#np.where(self.list_Distances==round(get_distance(self,i),5))[0][0]
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

            if debugger:
                print(i,r,s,round(self.get_distance(i),5),JMatrixValue)

            ######################
            z=1*(self.M_list[Layer1]/N_list[Layer1])
            Gamma=(np.exp(-1.0j*np.dot(self.get_distance_vector(i),np.transpose(2*np.pi*(q/np.array([a,a,a]))))))*(self.M_list[Layer1]/N_list[Layer1])
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
        
                                  
                
    if hermitian:
        for i in range(len(self._qpts)):
            H_final[:,:,i]=np.block([[H_main[:,:,i],H_off1[:,:,i]],
                                     [H_off2[:,:,i],H_main1[:,:,i]]])
    else:
        for i in range(len(self._qpts)):
            H_final[:,:,i]=np.block([[H_main[:,:,i],-H_off1[:,:,i]],
                                     [H_off2[:,:,i],-(H_main1[:,:,i])]])
            
    H_final=H_final/4

    return H_final, self.orientationEach
  
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

  def spin_scattering_function(self,H,Emax=100,Emin=-100,Estep=1000,Direction='xx',broadening=0.1):
    
    eVals_full,eVecs_fullL,eVecs_fullR = self.diagonalize_function(H)
    omega=np.linspace(Emin,Emax,Estep)

    X=np.zeros(np.shape(eVecs_fullL),dtype=complex)

    for i in range(np.shape(eVecs_fullL)[-1]):
        X[:,:,i] = np.linalg.inv(eVecs_fullL[:,:,i])

    SpinSpin=np.zeros([Estep,len(self._qpts)],dtype=complex)

    print(Direction[0])

    for Enum,En in enumerate(omega):
        for n in range(2*self.M):
            Smatrix=0
            for r in range(self.M):
                for s in range(self.M):

                    Wval=v_matrix_minus(self,r,Direction[0])*X[n,r,:] + v_matrix_plus(self,r,Direction[1])*X[n,r+self.M,:]
                    Wconj=np.conj(v_matrix_minus(self,s,Direction[0])*X[n,s,:] + v_matrix_plus(self,s,Direction[1])*X[n,s+self.M,:])

                    Smatrix+=(Wval*Wconj)

            SpinSpin[Enum,:] = SpinSpin[Enum,:] + (bose_einstein_distribution(abs(En), 0,self.Temperature)+1)*(Smatrix*lorentzian_distribution(En, eVals_full[:,n], broadening))

    return SpinSpin

  def spin_scattering_function_film(self,H,Emax=100,Emin=-100,Estep=1000,Direction='xx',broadening=0.1):
    
    Directions=['x','y','z']

    eVals_full,eVecs_fullL,eVecs_fullR = self.diagonalize_function(H)
    
    N = len(eVals_full[0,:])//2
    N1 = len(self.zNum)
    Qgrid = len(self._qpts)

    omegaX=np.linspace(Emin,Emax,Estep,dtype=complex)

    Vplus,Vminus=V(self.orientationEach)

    a=self._atoms.cell[0,0]
    b=self._atoms.cell[1,1]
    c=self._atoms.cell[2,2]

    Total=[]

    for numi,i in enumerate(self.M_list):
        for j in range(i):
            Total.append(self.zNum[numi]*c)

    Total=np.array(Total)

    A = (np.zeros(len(Total)))
    A = np.vstack((A,np.zeros(len(Total))))
    A = np.vstack((A,Total))
    A = np.transpose(A)

    exp_sum_j = np.exp(-1.0j*np.dot(2*np.pi*self._qpts/np.array([a,a,a]),np.transpose(A)))
    exp_sum_j = cleanup(exp_sum_j)

    alpha=Directions.index(Direction[0])
    beta=Directions.index(Direction[1])

    plotValues= np.zeros((Estep,Qgrid),dtype=complex,order='F')

    plotValues= scattering_function_loop(Qgrid,Estep,plotValues,N,self.Temperature,eVecs_fullL,eVecs_fullR,eVals_full,exp_sum_j,omegaX,broadening,Vplus,Vminus,Directions.index(Direction[0]),Directions.index(Direction[1]))



    #magnons.magnons_function(Qgrid,Estep,plotValues,N,2*N,self.Temperature,eVecs_fullL,eVals_full,exp_sum_j,omegaX,broadening,Vplus,Vminus,Directions.index(Direction[0]),Directions.index(Direction[1]))  #Directions.index(Direction[0]),Directions.index(Direction[1])

    plotValues=(1/(2*len(self.zNum)))*plotValues

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
