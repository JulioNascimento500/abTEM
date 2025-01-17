"""Module to calculate potentials using the independent atom model."""
from abc import ABCMeta, abstractmethod
from copy import copy
from typing import Union, Sequence, Callable, Generator, Tuple, Type
from collections import namedtuple

import h5py
import numpy as np
from ase import Atoms
from ase import units
from scipy.optimize import brentq

from abtem.base_classes import Grid, HasGridMixin, Cache, cached_method, HasAcceleratorMixin, Accelerator, Event, \
    watched_property, AntialiasFilter, cache_clear_callback, HasEventMixin
from abtem.device import get_device_function, get_array_module, get_array_module_from_device, copy_to_device, \
    HasDeviceMixin, asnumpy, get_available_memory
from abtem.measure import calibrations_from_grid, Measurement
from abtem.parametrizations import kirkland, dvdr_kirkland, load_kirkland_parameters, kirkland_projected_fourier
from abtem.parametrizations import lobato, dvdr_lobato, load_lobato_parameters
from abtem.structures import is_cell_orthogonal, SlicedAtoms, pad_atoms, rotate_atoms_to_plane, orthogonalize_cell
from abtem.tanh_sinh import integrate, tanh_sinh_nodes_and_weights
from abtem.temperature import AbstractFrozenPhonons, DummyFrozenPhonons
from abtem.magnon import AbstractMagnonInput, DummyMagnonInput
from abtem.utils import energy2sigma, ProgressBar, generate_batches, _disc_meshgrid
import warnings

# Vacuum permitivity in ASE units
eps0 = units._eps0 * units.A ** 2 * units.s ** 4 / (units.kg * units.m ** 3)

# Conversion factor from unitless potential parametrizations to ASE potential units
kappa = 4 * np.pi * eps0 / (2 * np.pi * units.Bohr * units._e * units.C)

# Permeability of free space   1.26e-6  (N/A^2)
mu0=units._mu0

# Bohr magneton (Am^2)
muB=9.27e-24                       

# Electron charge (Coulomb)
electronCharge=1.6e-19                   

class AbstractPotential(HasGridMixin, metaclass=ABCMeta):
    """
    Potential abstract base class

    Base class common for all potentials.
    """

    def __init__(self, precalculate):
        self._precalculate = precalculate

    def __len__(self):
        return self.num_slices

    @property
    @abstractmethod
    def num_slices(self):
        """The number of projected potential slices."""
        pass

    @property
    @abstractmethod
    def num_frozen_phonon_configs(self):
        pass

    @property
    @abstractmethod
    def generate_frozen_phonon_potentials(self, pbar=False):
        pass

    @property
    def thickness(self):
        return sum([self.get_slice_thickness(i) for i in range(len(self))])

    def check_slice_idx(self, i):
        """Raises an error if i is greater than the number of slices."""
        if i >= self.num_slices:
            raise RuntimeError('Slice index {} too large for potential with {} slices'.format(i, self.num_slices))

    def generate_transmission_functions(self, energy, first_slice=0, last_slice=None, max_batch=1):
        """
        Generate the transmission functions one slice at a time.

        Parameters
        ----------
        energy: float
            Electron energy [eV].
        first_slice: int
            First potential slice to generate.
        last_slice: int, optional
            Last potential slice generate.
        max_batch: int
            Maximum number of potential slices calculated in parallel.

        Returns
        -------
        generator of PotentialArray objects
        """
        antialias_filter = AntialiasFilter()

        for start, end, potential_slice in self.generate_slices(first_slice, last_slice, max_batch=max_batch):
            yield start, end, potential_slice.as_transmission_function(energy,
                                                                       in_place=True,
                                                                       max_batch=max_batch,
                                                                       antialias_filter=antialias_filter) ,potential_slice
## jcdn500 added to the yield the potential slice

    @abstractmethod
    def generate_slices(self, first_slice=0, last_slice=None, max_batch=1):
        """
        Generate the potential slices.

        Parameters
        ----------
        first_slice: int
            First potential slice to generate.
        last_slice: int, optional
            Last potential slice generate.
        max_batch: int
            Maximum number of potential slices calculated in parallel.

        Returns
        -------
        generator of PotentialArray objects
        """

        pass

    @abstractmethod
    def get_slice_thickness(self, i):
        """
        Get the slice thickness [Å].

        Parameters
        ----------
        i: int
            Slice index.
        """

        pass

    def __iter__(self):
        for _, __, p in self.generate_slices(max_batch=1):
            yield p

    # def show(self, **kwargs):
    #     """
    #     Show the potential projection. This requires building all potential slices.
    #
    #     Parameters
    #     ----------
    #     kwargs:
    #         Additional keyword arguments for abtem.plot.show_image.
    #     """
    #
    #     self[:].show(**kwargs)

    def copy(self):
        """Make a copy."""
        return copy(self)


class AbstractPotentialBuilder(AbstractPotential):
    """Potential builder abstract class."""

    def __init__(self, precalculate=True, device='cpu', storage='cpu'):
        self._precalculate = precalculate
        self._storage = storage
        self._device = device
        super().__init__(precalculate)

    @property
    def precalculate(self):
        return self._precalculate

    @property
    def storage(self):
        return self._storage

    @property
    def device(self):
        return self._device

    def _estimate_max_batch(self):
        memory_per_wave = 2 * 4 * self.gpts[0] * self.gpts[1]
        available_memory = .2 * get_available_memory(self._device)
        return min(int(available_memory / memory_per_wave), len(self))

    def __getitem__(self, items):
        if isinstance(items, int):
            if items >= self.num_slices:
                raise StopIteration
            return next(self.generate_slices(items, items + 1, max_batch=1))[2]

        elif isinstance(items, slice):
            if items.start is None:
                first_slice = 0
            else:
                first_slice = items.start

            if items.stop is None:
                last_slice = len(self)
            else:
                last_slice = items.stop

            if items.step is None:
                step = 1
            else:
                step = items.step

            return self.build(first_slice, last_slice, pbar=False)[::step]
        else:
            raise TypeError('Potential must indexed with integers or slices, not {}'.format(type(items)))

    def build(self,
              first_slice: int = 0,
              last_slice: int = None,
              energy: float = None,
              max_batch: int = None,
              pbar: Union[bool, ProgressBar] = False,
              ) -> 'PotentialArray':
        """
        Precalcaulate the potential as a potential array.

        Parameters
        ----------
        first_slice: int
            First potential slice to generate.
        last_slice: int, optional
            Last potential slice generate.
        energy: float
            Electron energy [eV]. If given, the transmission functions will be returned.
        max_batch: int
            Maximum number of potential slices calculated in parallel.
        pbar: bool
            If true, show progress bar.

        Returns
        -------
        PotentialArray object
        """

        

        self.grid.check_is_defined()

        if last_slice is None:
            last_slice = len(self)

        if max_batch is None:
            max_batch = self._estimate_max_batch()

        storage_xp = get_array_module_from_device(self._storage)

        if energy is None:
            array = storage_xp.zeros((last_slice - first_slice,) + (self.gpts[0], self.gpts[1]), dtype=np.float32)
            generator = self.generate_slices(max_batch=max_batch, first_slice=first_slice, last_slice=last_slice)
        else:
            array = storage_xp.zeros((last_slice - first_slice,) + (self.gpts[0], self.gpts[1]), dtype=np.complex64)
            generator = self.generate_transmission_functions(energy=energy,
                                                             max_batch=max_batch,
                                                             first_slice=first_slice,
                                                             last_slice=last_slice)

        slice_thicknesses = np.zeros(last_slice - first_slice)

        if isinstance(pbar, bool):
            pbar = ProgressBar(total=len(self), desc='Potential', disable=not pbar)
            close_pbar = True
        else:
            close_pbar = False

        pbar.reset()
        N = 0
        for start, end, potential_slice in generator:
            n = end - start
            array[N:N + n] = copy_to_device(potential_slice.array, self._storage)
            slice_thicknesses[N:N + n] = potential_slice.slice_thicknesses
            pbar.update(n)
            N += n

        pbar.refresh()

        if close_pbar:
            pbar.close()

        if energy is None:
            return PotentialArray(array, slice_thicknesses=slice_thicknesses, mag_variables=self.magnon_package, extent=self.extent)
        else:
            return TransmissionFunction(array, slice_thicknesses=slice_thicknesses, extent=self.extent, energy=energy)

    def project(self):
        projected = self[0]
        max_batch = self._estimate_max_batch()
        for _, _, projected_chunk in self.generate_slices(max_batch=max_batch):
            projected._array += projected_chunk.array.sum(0)
        return projected.project()


class PotentialIntegrator:
    """
    Perform finite integrals of a radial function along a straight line.

    Parameters
    ----------
    function: callable
        Radial function to integrate.
    r: array of float
        The evaluation points of the integrals.
    cutoff: float, optional
        The radial function is assumed to be zero outside this threshold.
    cache_size: int, optional
        The maximum number of integrals that will be cached.
    cache_key_decimals: int, optional
        The number of decimals used in the cache keys.
    tolerance: float, optional
        The absolute error tolerance of the integrals.
    """

    def __init__(self,
                 function: Callable,
                 r: np.ndarray,
                 max_interval,
                 cutoff: float = None,
                 cache_size: int = 4096,
                 cache_key_decimals: int = 2,
                 tolerance: float = 1e-6):

        self._function = function
        self._r = r

        if cutoff is None:
            self._cutoff = r[-1]
        else:
            self._cutoff = cutoff

        self._cache = Cache(cache_size)
        self._cache_key_decimals = cache_key_decimals
        self._tolerance = tolerance

        def f(z):
            return self._function(np.sqrt(self.r[0] ** 2 + (z * max_interval / 2 + max_interval / 2) ** 2))

        value, error_estimate, step_size, order = integrate(f, -1, 1, self._tolerance)

        self._xk, self._wk = tanh_sinh_nodes_and_weights(step_size, order)

    @property
    def r(self):
        return self._r

    @property
    def cutoff(self):
        return self._cutoff

    def integrate(self, z: Union[float, Sequence[float]], a: Union[float, Sequence[float]],
                  b: Union[float, Sequence[float]]):
        """
        Evaulate the integrals of the radial function at the evaluation points.

        Parameters
        ----------
        a: float
            Lower limit of integrals.
        b: float
            Upper limit of integrals.

        Returns
        -------
        1d array
            The evaulated integrals.
        """

        vr = np.zeros((len(z), self.r.shape[0]), np.float32)
        dvdr = np.zeros((len(z), self.r.shape[0]), np.float32)
        a = np.round(a - z, self._cache_key_decimals)
        b = np.round(b - z, self._cache_key_decimals)

        split = a * b < 0

        a, b = np.abs(a), np.abs(b)
        a, b = np.minimum(a, b), np.minimum(np.maximum(a, b), self.cutoff)

        for i, (ai, bi) in enumerate(zip(a, b)):
            if split[i]:  # split the integral
                values1, derivatives1 = self._do_integrate(0, ai)
                values2, derivatives2 = self._do_integrate(0, bi)
                result = (values1 + values2, derivatives1 + derivatives2)
            else:
                result = self._do_integrate(ai, bi)

            vr[i] = result[0]
            dvdr[i, :-1] = result[1]

        return vr, dvdr

    @cached_method('_cache')
    def _do_integrate(self, a, b):
        zm = (b - a) / 2.
        zp = (a + b) / 2.

        def f(z):
            return self._function(np.sqrt(self.r[:, None] ** 2 + (z * zm + zp) ** 2))

        values = np.sum(f(self._xk[None]) * self._wk[None], axis=1) * zm
        derivatives = np.diff(values) / np.diff(self.r)

        return values, derivatives


def superpose_deltas(positions, z, array):
    shape = array.shape[-2:]
    xp = get_array_module(array)
    rounded = xp.floor(positions).astype(xp.int32)
    rows, cols = rounded[:, 0], rounded[:, 1]

    array[z, rows, cols] += (1 - (positions[:, 0] - rows)) * (1 - (positions[:, 1] - cols))
    array[z, (rows + 1) % shape[0], cols] += (positions[:, 0] - rows) * (1 - (positions[:, 1] - cols))
    array[z, rows, (cols + 1) % shape[1]] += (1 - (positions[:, 0] - rows)) * (positions[:, 1] - cols)
    array[z, (rows + 1) % shape[0], (cols + 1) % shape[1]] += (rows - positions[:, 0]) * (cols - positions[:, 1])


class CrystalPotential(AbstractPotential, HasEventMixin):
    """
    Crystal potential object

    The crystal potential may be used to represent a potential consisting of a repeating unit. This may allow
    calculations to be performed with lower memory and computational cost.

    The crystal potential has an additional function in conjunction with frozen phonon calculations. The number of
    frozen phonon configurations are not given by the FrozenPhonon objects, rather the ensemble of frozen phonon
    potentials represented by a potential with frozen phonons represent a collection of units, which will be assembled
    randomly to represent a random potential. The number of frozen phonon configurations should be given explicitely.
    This may save computational cost since a smaller number of units can be combined to a larger frozen phonon ensemble.

    Parameters
    ----------
    potential_unit : AbstractPotential
        The potential unit that repeated will create the full potential.
    repetitions : three int
        The repetitions of the potential in x, y and z.
    num_frozen_phonon_configs : int
        Number of frozen phonon configurations.
    """

    def __init__(self,
                 potential_unit: AbstractPotential,
                 repetitions: Tuple[int, int, int],
                 num_frozen_phonon_configs: int = 1):

        self._potential_unit = potential_unit
        self.repetitions = repetitions
        self._num_frozen_phonon_configs = num_frozen_phonon_configs

        if (potential_unit.num_frozen_phonon_configs == 1) & (num_frozen_phonon_configs > 1):
            warnings.warn('"num_frozen_phonon_configs" is greater than one, but the potential unit does not have'
                          'frozen phonons')

        if (potential_unit.num_frozen_phonon_configs > 1) & (num_frozen_phonon_configs == 1):
            warnings.warn('the potential unit has frozen phonons, but "num_frozen_phonon_configs" is set to 1')

        self._cache = Cache(1)
        self._event = Event()

        gpts = (self._potential_unit.gpts[0] * self.repetitions[0],
                self._potential_unit.gpts[1] * self.repetitions[1])
        extent = (self._potential_unit.extent[0] * self.repetitions[0],
                  self._potential_unit.extent[1] * self.repetitions[1])

        self._grid = Grid(extent=extent, gpts=gpts, sampling=self._potential_unit.sampling, lock_extent=True)
        self._grid.observe(self._event.notify)
        self._event.observe(cache_clear_callback(self._cache))

        super().__init__(precalculate=False)

    @HasGridMixin.gpts.setter
    def gpts(self, gpts):
        if not ((gpts[0] % self.repetitions[0] == 0) and (gpts[1] % self.repetitions[0] == 0)):
            raise ValueError('gpts must be divisible by the number of potential repetitions')
        self.grid.gpts = gpts
        self._potential_unit.gpts = (gpts[0] // self._repetitions[0], gpts[1] // self._repetitions[1])

    @HasGridMixin.sampling.setter
    def sampling(self, sampling):
        self.sampling = sampling
        self._potential_unit.sampling = sampling

    @property
    def num_frozen_phonon_configs(self):
        return self._num_frozen_phonon_configs

    def generate_frozen_phonon_potentials(self, pbar=False):
        for i in range(self.num_frozen_phonon_configs):
            yield self

    @property
    def repetitions(self) -> Tuple[int, int, int]:
        return self._repetitions

    @repetitions.setter
    def repetitions(self, repetitions: Tuple[int, int, int]):
        repetitions = tuple(repetitions)

        if len(repetitions) != 3:
            raise ValueError('repetitions must be sequence of length 3')

        self._repetitions = repetitions

    @property
    def num_slices(self) -> int:
        return self._potential_unit.num_slices * self.repetitions[2]

    def get_slice_thickness(self, i) -> float:
        return self._potential_unit.get_slice_thickness(i)

    @cached_method('_cache')
    def _calculate_configs(self, energy, max_batch=1):
        potential_generators = self._potential_unit.generate_frozen_phonon_potentials(pbar=False)

        potential_configs = []
        for potential in potential_generators:

            if isinstance(potential, AbstractPotentialBuilder):
                potential = potential.build(max_batch=max_batch)
            elif not isinstance(potential, PotentialArray):
                raise RuntimeError()

            if energy is not None:
                potential = potential.as_transmission_function(energy=energy, max_batch=max_batch, in_place=False)

            potential = potential.tile(self.repetitions[:2])
            potential_configs.append(potential)

        return potential_configs

    def _generate_slices_base(self, first_slice=0, last_slice=None, max_batch=1, energy=None):

        first_layer = first_slice // self._potential_unit.num_slices
        if last_slice is None:
            last_layer = self.repetitions[2]
        else:
            last_layer = last_slice // self._potential_unit.num_slices

        first_slice = first_slice % self._potential_unit.num_slices
        last_slice = None

        configs = self._calculate_configs(energy, max_batch)

        if len(configs) == 1:
            layers = configs * self.repetitions[2]
        else:
            layers = [configs[np.random.randint(len(configs))] for _ in range(self.repetitions[2])]

        for layer_num, layer in enumerate(layers[first_layer:last_layer]):

            if layer_num == last_layer:
                last_slice = last_slice % self._potential_unit.num_slices

            for start, end, potential_slice in layer.generate_slices(first_slice=first_slice,
                                                                     last_slice=last_slice,
                                                                     max_batch=max_batch):
                yield layer_num + start, layer_num + end, potential_slice

                first_slice = 0

    def generate_slices(self, first_slice=0, last_slice=None, max_batch=1):
        return self._generate_slices_base(first_slice=first_slice, last_slice=last_slice, max_batch=max_batch)

    def generate_transmission_functions(self, energy, first_slice=0, last_slice=None, max_batch=1):
        return self._generate_slices_base(first_slice=first_slice, last_slice=last_slice, max_batch=max_batch,
                                          energy=energy)


class Potential(AbstractPotentialBuilder, HasDeviceMixin, HasEventMixin):
    """
    Potential object.

    The potential object is used to calculate the electrostatic potential of a set of atoms represented by an ASE atoms
    object. The potential is calculated with the Independent Atom Model (IAM) using a user-defined parametrization
    of the atomic potentials.

    Parameters
    ----------
    atoms : Atoms or FrozenPhonons object
        Atoms or FrozenPhonons defining the atomic configuration(s) used in the IAM of the electrostatic potential(s).
    gpts : one or two int, optional
        Number of grid points describing each slice of the potential.
    sampling : one or two float, optional
        Lateral sampling of the potential [1 / Å].
    slice_thickness : float, optional
        Thickness of the potential slices in Å for calculating the number of slices used by the multislice algorithm.
        Default is 0.5 Å.
    parametrization : 'lobato' or 'kirkland', optional
        The potential parametrization describes the radial dependence of the potential for each element. Two of the
        most accurate parametrizations are available by Lobato et. al. and Kirkland. The abTEM default is 'lobato'.
        See the citation guide for references.
    projection : 'finite' or 'infinite'
        If 'finite' the 3d potential is numerically integrated between the slice boundaries. If 'infinite' the infinite
        potential projection of each atom will be assigned to a single slice.
    cutoff_tolerance : float, optional
        The error tolerance used for deciding the radial cutoff distance of the potential [eV / e]. The cutoff is only
        relevant for potentials using the 'finite' projection scheme.
    device : str, optional
        The device used for calculating the potential. The default is 'cpu'.
    precalculate : bool
        If True, precalculate the potential else the potential will be calculated on-the-fly and immediately discarded.
        Default is True.
    storage : str, optional
        The device on which to store the created potential. The default is 'None', defaulting to the chosen device.
    """

    def __init__(self,
                 atoms: Union[Atoms, AbstractFrozenPhonons, AbstractMagnonInput] = None,
                 gpts: Union[int, Sequence[int]] = None,
                 sampling: Union[float, Sequence[float]] = None,
                 slice_thickness: float = .5,
                 parametrization: str = 'lobato',
                 projection: str = 'finite',
                 cutoff_tolerance: float = 1e-3,
                 device: str = 'cpu',
                 precalculate: bool = True,
                 z_periodic: bool = True,
                 plane: str = 'xy',
                 storage: str = None):

        self._cutoff_tolerance = cutoff_tolerance
        self._parametrization = parametrization
        self._slice_thickness = slice_thickness

        self._storage = storage

        if parametrization.lower() == 'lobato':
            self._parameters = load_lobato_parameters()
            self._function = lobato
            self._derivative = dvdr_lobato

        elif parametrization.lower() == 'kirkland':
            self._parameters = load_kirkland_parameters()
            self._function = kirkland
            self._derivative = dvdr_kirkland
        else:
            raise RuntimeError('Parametrization {} not recognized'.format(parametrization))

        if projection == 'infinite':
            if parametrization.lower() != 'kirkland':
                raise RuntimeError('Infinite projections are only implemented for the Kirkland parametrization')
        elif (projection != 'finite'):
            raise RuntimeError('Projection must be "finite" or "infinite"')

        self._projection = projection

        if isinstance(atoms, AbstractFrozenPhonons):
            self._frozen_phonons = atoms
        elif isinstance(atoms, AbstractMagnonInput):
            '''If the calculation is of magnons, it is necessary to obtain the eigenvalues and eigenvectors
            '''
            #self._frozen_phonons = DummyMagnonInput(atoms._atoms_multi)

            self._frozen_phonons = DummyMagnonInput(atoms._atoms_multi)

            self.unitatoms=atoms._atoms

            self.H, self.orientationEach = atoms.Hamiltonian_Test(badflag=True)
        
            print(np.shape(self.H))

            if atoms._anisotropies==0:
                print('No anisotropy')
                self.eVals_full,self.eVecs_fullL,self.eVecs_fullR = atoms.diagonalize_function(self.H)
            else:
                self.Hk=atoms.HamiltonianK()
                self.eVals_full,self.eVecs_fullL,self.eVecs_fullR = atoms.diagonalize_function(self.H+self.Hk)

            # print(atoms._atoms_multi.positions)
            # print(atoms._atoms.positions)

            self.qpts=atoms._qpts

            self.Temperature=atoms.Temperature

            self.inelastic_layer=atoms.inelastic_layer
            
        else:
            self._frozen_phonons = DummyFrozenPhonons(atoms)

        atoms = next(iter(self._frozen_phonons))

        if not is_cell_orthogonal(atoms):
            atoms, transformations = orthogonalize_cell(atoms, max_repetitions=2, return_transform=True)
            for transformation in transformations:
                if not np.allclose(transformation, 0.):
                    raise RuntimeError('The unit cell of the atoms is not orthogonal '
                                       'and could not be made orthogonal without ambiguity. '
                                       'See our tutorial on making orthogonal cells '
                                       'https://abtem.readthedocs.io/en/latest/tutorials/orthogonal_cells.html')

        print(atoms)
        if np.abs(atoms.cell[2, 2]) < 1e-12:
            raise RuntimeError('Atoms cell has no thickness')

        self._atoms = atoms
        self._plane = plane

        self._grid = Grid(extent=np.diag(atoms.cell)[:2], gpts=gpts, sampling=sampling, lock_extent=True)

        self._cutoffs = {}
        self._integrators = {}
        self._disc_indices = {}

        def grid_changed_callback(*args, **kwargs):
            self._integrators = {}
            self._disc_indices = {}

        self.grid.observe(grid_changed_callback)
        self._event = Event()

        if storage is None:
            storage = device

        self._z_periodic = z_periodic

        super().__init__(precalculate=precalculate, device=device, storage=storage)

    
    @property
    def magnon_package(self):

        #if isinstance(self.atoms, AbstractMagnonInput):
        """Returns all variables required for the inelastic magnons calculation"""

        package = namedtuple('package', ['atoms', 'eVals_full','eVecs_fullL','eVecs_fullR','qpts','orientationEach','Temperature','inelastic_layer'])

        if hasattr(self, 'eVals_full'):
            mag_var = package(self.unitatoms,self.eVals_full,self.eVecs_fullL,self.eVecs_fullR,self.qpts,self.orientationEach,self.Temperature,self.inelastic_layer)

        else:

            mag_var = package(self.unitatoms,None,None,None,None,None,None,None)
        return mag_var

    @property
    def parametrization(self):
        """The potential parametrization."""
        return self._parameters

    @property
    def projection(self):
        """The projection method."""
        return self._projection

    @property
    def parameters(self):
        """The parameters of the potential parametrization."""
        return self._parameters

    @property
    def function(self):
        """The potential function of the parametrization."""
        return self._function

    @property
    def atoms(self):
        """Atoms object defining the atomic configuration."""
        return self._atoms

    @property
    def frozen_phonons(self):
        """FrozenPhonons object defining the atomic configuration(s)."""
        return self._frozen_phonons

    @property
    def num_frozen_phonon_configs(self):
        return len(self.frozen_phonons)

    @property
    def cutoff_tolerance(self):
        """The error tolerance used for deciding the radial cutoff distance of the potential [eV / e]."""
        return self._cutoff_tolerance

    @property
    def num_slices(self):
        """The number of projected potential slices."""
        return int(np.ceil(self._atoms.cell[2, 2] / self._slice_thickness))

    @property
    def slice_thickness(self):
        """The thickness of the projected potential slices."""
        return self._slice_thickness

    @slice_thickness.setter
    @watched_property('_event')
    def slice_thickness(self, value):
        self._slice_thickness = value

    def get_slice_thickness(self, i) -> float:
        return self._atoms.cell[2, 2] / self.num_slices

    def get_parameterized_function(self, number) -> Callable:
        return lambda r: self._function(r, self.parameters[number])

    def get_cutoff(self, number: int) -> float:
        """
        Cutoff distance for atomic number given an error tolerance.

        Parameters
        ----------
        number: int
            Atomic number.

        Returns
        -------
        cutoff: float
            The potential cutoff.
        """

        try:
            return self._cutoffs[number]
        except KeyError:
            def f(r):
                return self.function(r, self.parameters[number]) - self.cutoff_tolerance

            self._cutoffs[number] = brentq(f, 1e-7, 1000)
            return self._cutoffs[number]

    def get_tapered_function(self, number: int) -> Callable:
        """
        Tapered potential function for atomic number.

        Parameters
        ----------
        number: int
            Atomic number.

        Returns
        -------
        callable
        """

        cutoff = self.get_cutoff(number)
        rolloff = .85 * cutoff

        def soft_function(r):
            result = np.zeros_like(r)
            valid = r < cutoff
            transition = valid * (r > rolloff)
            result[valid] = self._function(r[valid], self.parameters[number])
            result[transition] *= (np.cos(np.pi * (r[transition] - rolloff) / (cutoff - rolloff)) + 1.) / 2
            return result

        return soft_function

    def get_integrator(self, number: int) -> PotentialIntegrator:
        """
        Potential integrator for atomic number.

        Parameters
        ----------
        number: int
            Atomic number.

        Returns
        -------
        PotentialIntegrator object
        """

        try:
            return self._integrators[number]
        except KeyError:
            cutoff = self.get_cutoff(number)
            soft_function = self.get_tapered_function(number)
            inner_cutoff = np.min(self.sampling) / 2.

            num_points = int(np.ceil(cutoff / np.min(self.sampling) * 10.))
            r = np.geomspace(inner_cutoff, cutoff, num_points)
            max_interval = self.slice_thickness
            self._integrators[number] = PotentialIntegrator(soft_function, r, max_interval, cutoff)
            return self._integrators[number]

    def _get_radial_interpolation_points(self, number):
        """Internal function for the indices of the radial interpolation points."""
        try:
            return self._disc_indices[number]
        except KeyError:
            cutoff = self.get_cutoff(number)
            margin = np.int(np.ceil(cutoff / np.min(self.sampling)))
            rows, cols = _disc_meshgrid(margin)
            self._disc_indices[number] = np.hstack((rows[:, None], cols[:, None]))
            return self._disc_indices[number]

    def generate_slices(self, first_slice=0, last_slice=None, max_batch=1) -> Generator:
        self.grid.check_is_defined()

        if last_slice is None:
            last_slice = len(self)

        if self.projection == 'finite':
            return self._generate_slices_finite(first_slice=first_slice, last_slice=last_slice, max_batch=max_batch)
        else:
            return self._generate_slices_infinite(first_slice=first_slice, last_slice=last_slice, max_batch=max_batch)

    def _generate_slices_infinite(self, first_slice=0, last_slice=None, max_batch=1) -> Generator:
        # TODO : simplify method using the sliced atoms object

        xp = get_array_module_from_device(self._device)

        fft2_convolve = get_device_function(xp, 'fft2_convolve')

        atoms = self.atoms.copy()
        atoms = rotate_atoms_to_plane(atoms, self._plane)

        atoms.pbc = True
        atoms.wrap()

        positions = atoms.get_positions().astype(np.float32)
        numbers = atoms.get_atomic_numbers()
        unique = np.unique(numbers)
        order = np.argsort(positions[:, 2])

        positions = positions[order]
        numbers = numbers[order]

        kx = xp.fft.fftfreq(self.gpts[0], self.sampling[0])
        ky = xp.fft.fftfreq(self.gpts[1], self.sampling[1])
        kx, ky = xp.meshgrid(kx, ky, indexing='ij')
        k = xp.sqrt(kx ** 2 + ky ** 2)

        sinc = xp.sinc(xp.sqrt((kx * self.sampling[0]) ** 2 + (ky * self.sampling[1]) ** 2))

        scattering_factors = {}
        for atomic_number in unique:
            f = kirkland_projected_fourier(k, self.parameters[atomic_number])
            scattering_factors[atomic_number] = (f / (sinc * self.sampling[0] * self.sampling[1] * kappa)).astype(
                xp.complex64)

        slice_idx = np.floor(positions[:, 2] / atoms.cell[2, 2] * self.num_slices).astype(np.int)

        start, end = next(generate_batches(last_slice - first_slice, max_batch=max_batch, start=first_slice))

        array = xp.zeros((end - start,) + self.gpts, dtype=xp.complex64)
        temp = xp.zeros((end - start,) + self.gpts, dtype=xp.complex64)

        for start, end in generate_batches(last_slice - first_slice, max_batch=max_batch, start=first_slice):
            array[:] = 0.
            start_idx = np.searchsorted(slice_idx, start)
            end_idx = np.searchsorted(slice_idx, end)

            if start_idx != end_idx:
                for j, number in enumerate(unique):
                    temp[:] = 0.
                    chunk_positions = positions[start_idx:end_idx]
                    chunk_slice_idx = slice_idx[start_idx:end_idx] - start

                    if len(unique) > 1:
                        chunk_positions = chunk_positions[numbers[start_idx:end_idx] == number]
                        chunk_slice_idx = chunk_slice_idx[numbers[start_idx:end_idx] == number]

                    chunk_positions = xp.asarray(chunk_positions[:, :2] / self.sampling)

                    superpose_deltas(chunk_positions, chunk_slice_idx, temp)
                    temp = fft2_convolve(temp, scattering_factors[number])

                    array += temp

            slice_thicknesses = [self.get_slice_thickness(i) for i in range(start, end)]
            yield start, end, PotentialArray(array.real[:end - start], slice_thicknesses,mag_variables=self.magnon_package, extent=self.extent)

    def _generate_slices_finite(self, first_slice=0, last_slice=None, max_batch=1) -> Generator:
        xp = get_array_module_from_device(self._device)
        interpolate_radial_functions = get_device_function(xp, 'interpolate_radial_functions')

        array = None
        unique = np.unique(self.atoms.numbers)

        atoms = rotate_atoms_to_plane(self.atoms, self._plane)

        if (self._z_periodic) & (len(self.atoms) > 0):
            max_cutoff = max(self.get_integrator(number).cutoff for number in unique)
            atoms = pad_atoms(atoms, margin=max_cutoff, directions='z', in_place=False)

        sliced_atoms = SlicedAtoms(atoms, self.slice_thickness)

        for start, end in generate_batches(last_slice - first_slice, max_batch=max_batch, start=first_slice):
            if array is None:
                array = xp.zeros((end - start,) + self.gpts, dtype=xp.float32)
            else:
                array[:] = 0.

            for number in unique:
                integrator = self.get_integrator(number)
                disc_indices = xp.asarray(self._get_radial_interpolation_points(number))
                chunk_atoms = sliced_atoms.get_subsliced_atoms(start, end, number, z_margin=integrator.cutoff)

                if len(chunk_atoms) == 0:
                    continue

                positions = np.zeros((0, 3), dtype=xp.float32)
                slice_entrances = np.zeros((0,), dtype=xp.float32)
                slice_exits = np.zeros((0,), dtype=xp.float32)
                run_length_enconding = np.zeros((end - start + 1,), dtype=xp.int32)

                for i, slice_idx in enumerate(range(start, end)):
                    slice_atoms = chunk_atoms.get_subsliced_atoms(slice_idx,
                                                                  padding=integrator.cutoff,
                                                                  z_margin=integrator.cutoff)

                    slice_positions = slice_atoms.positions
                    slice_entrance = slice_atoms.get_slice_entrance(slice_idx)
                    slice_exit = slice_atoms.get_slice_exit(slice_idx)

                    positions = np.vstack((positions, slice_positions))
                    slice_entrances = np.concatenate((slice_entrances, [slice_entrance] * len(slice_positions)))
                    slice_exits = np.concatenate((slice_exits, [slice_exit] * len(slice_positions)))

                    run_length_enconding[i + 1] = run_length_enconding[i] + len(slice_positions)

                vr, dvdr = integrator.integrate(positions[:, 2], slice_entrances, slice_exits)

                vr = xp.asarray(vr, dtype=xp.float32)
                dvdr = xp.asarray(dvdr, dtype=xp.float32)
                r = xp.asarray(integrator.r, dtype=xp.float32)
                sampling = xp.asarray(self.sampling, dtype=xp.float32)

                interpolate_radial_functions(array,
                                             run_length_enconding,
                                             disc_indices,
                                             positions,
                                             vr,
                                             r,
                                             dvdr,
                                             sampling)

            slice_thicknesses = [self.get_slice_thickness(i) for i in range(start, end)]

            yield start, end, PotentialArray(array[:end - start] / kappa,
                                             slice_thicknesses, mag_variables=self.magnon_package,
                                             extent=self.extent)

    def generate_frozen_phonon_potentials(self, pbar: Union[ProgressBar, bool] = True):
        """
        Function to generate scattering potentials for a set of frozen phonon configurations.

        Parameters
        ----------
        pbar: bool, optional
            Display a progress bar. Default is True.

        Returns
        -------
        generator
            Generator of potentials.
        """

        if isinstance(pbar, bool):
            pbar = ProgressBar(total=len(self), desc='Potential', disable=(not pbar) or (not self._precalculate))

        for atoms in self.frozen_phonons:
            self.atoms.positions[:] = atoms.positions
            pbar.reset()

            if self._precalculate:
                yield self.build(pbar=pbar)
            else:
                yield self

        pbar.refresh()
        pbar.close()

    def magnon_inelastic(self,wave,numpot=1,T=300,flag_theta1=True,flag_theta2=True):
        
        #self.slice_thicknesses
        
        import sympy as sym
        if  self.z_periodic==True:
            omega= self._atoms.cell.volume * 1e-30# Was in A^3, converted to m^3
        else:
            omega=np.linalg.norm(np.cross(self._atoms.cell[:][0], self._atoms.cell[:][1])) * 1e-20
        h= 6.6256E-34                    #planks constant  J*s
        C = 299792458                    #light speed m/s
             
        M0 = 9.1091E-31                  #electron mass
        E0 = M0 * C * C                  # Rest energy (Einstein)
        E = wave.energy * electronCharge # Electron energy in J

        lambd  = (h * C) / np.sqrt((2 * E * E0) + E**2) * 1e10 # Electron Wave lenght in A

        kT=T*8.6173e-5

        ##### Calculate Gradients


        theta1_final=np.zeros_like(wave._array)
        theta2_final=np.zeros_like(wave._array)

        gradientX=np.gradient(wave._array,self.extent[0]* 1e-10/self.gpts[0],axis=0)  # Gradient with pixel size m
        gradientY=np.gradient(wave._array,self.extent[1]* 1e-10/self.gpts[1],axis=1)  # Gradient with pixel size m

        gradientX2=np.gradient(gradientX,self.extent[0]* 1e-10/self.gpts[0],axis=0)  # Gradient with pixel size m
        gradientY2=np.gradient(gradientY,self.extent[1]* 1e-10/self.gpts[1],axis=1)  # Gradient with pixel size m

        ##### Calculate pre-factors
        Nt=1/omega # Already in A

        N=len(self.eVals_full[0,:])//2#len(Input.atoms)

        def scos(x): return sym.N(sym.cos(x))  
        def ssin(x): return sym.N(sym.sin(x))

        A=np.zeros([3,3,len(self.orientationEach)])

        for numS,S in enumerate(self.orientationEach):
            A[0,0,numS]= scos(S[1])*scos(S[2]); A[0,1,numS]=-ssin(S[2]); A[0,2,numS]=ssin(S[1])*scos(S[2]) 
            A[1,0,numS]= scos(S[1])*ssin(S[2]); A[1,1,numS]= scos(S[2]); A[1,2,numS]=ssin(S[1])*ssin(S[2])
            A[2,0,numS]=-ssin(S[1])           ; A[2,1,numS]= 0         ; A[2,2,numS]=scos(S[1])

        potential=self._array

        deltaz =self.thickness  ## in A 
        zperp  =(deltaz*numpot) + deltaz/2  ## Already in A
        sigmai = (np.pi/(lambd*wave.energy))  ## in 1/[A][eV]

        AArray=np.array([A[0,0,:] + 1.0j*A[0,1,:],A[1,0,:] + 1.0j*A[1,1,:],A[2,0,:] + 1.0j*A[2,1,:]])

        Projection1=np.sqrt(self.orientationEach[:,0])
        
        for numq,q in enumerate(self.qpts*(1/np.sum(self._atoms.cell,axis=0))): # converting the q to a fraction of wavevector in inverse m    *(1/np.sum(Input.aseatoms.cell,axis=0)*1e-10)

            #print(q)

            theta1=np.zeros_like(wave._array)
            theta2=np.zeros_like(wave._array)

            qmag=np.linalg.norm(q[:2]) * 1e-10
            
            #pre_factor=((mu0*muB**2)/(np.pi*electronCharge*omega*qmag))*np.sqrt(1/(2*Nt))
            #Projection1=np.exp(1.0j*Input.atoms[:,3].astype(float)*Input.qpts[numq,2])*np.sqrt(Input.MagMoms[:,0])
            #Projection2=eigenVecs[:N,N:,numq]*np.sqrt(Nk) + eigenVecs[N:,N:,numq]*np.sqrt(Nk+1)

            ##### Calculate occupation Numbers
            
            Nk1 = 1/(np.exp(np.abs(self.eigenVals[numq,:N])/kT)-1)#np.ones_like(1/(np.exp(eigenVals[numq,:]/kT)-1))   
            Nk2 = 1/(np.exp(np.abs(self.eigenVals[numq,N:])/kT)-1)         
            phase=np.exp(1.0j*zperp*self.qpts[numq,2])
            
            sigmaf = sigmai#np.pi/(((lambd)-(2*np.pi/(qmag)))*(E0))

            if  self.z_periodic==True:
            # #### Version 4 ####
            
                Aq1 = np.cross(AArray.T,np.array([1.0j*q[0],1.0j*q[1],1.0j*q[2]]))
                
                Aq2 = np.cross(AArray.T[:,:2],np.array([q[0],q[1]]))

                echarge=14.4  # electron charge in V/A


                pre_factor=((mu0*muB**2)/(np.pi*electronCharge*omega*qmag))*np.sqrt(1/(2*Nt))
                
                sqrt_Nk = np.sqrt(Nk1)
                sqrt_Nk_plus_1 = np.sqrt(Nk2 + 1)

                Projection2 = (self.eigenVecs[:N, N:, numq]@(sqrt_Nk*Aq2) + self.eigenVecs[N:, N:, numq]@(sqrt_Nk_plus_1*Aq2))
                
                Projection2_0 = (self.eigenVecs[:N, N:, numq]@(sqrt_Nk*Aq1[:,0]) + self.eigenVecs[N:, N:, numq]@(sqrt_Nk_plus_1*Aq1[:,0]))
                
                Projection2_1 = (self.eigenVecs[:N, N:, numq]@(sqrt_Nk*Aq1[:,1]) + self.eigenVecs[N:, N:, numq]@(sqrt_Nk_plus_1*Aq1[:,1]))

                Projection2_2 = (self.eigenVecs[:N, N:, numq]@(sqrt_Nk*Aq1[:,2]) + self.eigenVecs[N:, N:, numq]@(sqrt_Nk_plus_1*Aq1[:,2]))

                #print(np.shape(Projection2_0),np.shape(Projection1))

                theta1+=(Projection2_0@Projection1)*gradientX

                theta1+=(Projection2_1@Projection1)*gradientY

                theta1+=(Projection2_2@Projection1)*((1.0j*lambd*1e-10)/(4*np.pi))*(gradientX2+gradientY2)  #Lambd is converted back to m here

                theta1*=sigmaf*deltaz*pre_factor
                                

                #print('deltaz',deltaz)
                #print('sigmai',sigmai)
                #print('sigmaf',sigmaf)
                #print('phase',phase)
                #print('pre_factor',pre_factor)
                theta2-=sigmai*sigmaf*(Projection2@Projection1)*pre_factor*potential[0,:,:]*wave._array
            else:

                Aq1 = np.cross(AArray.T,np.array([1.0j*q[0],1.0j*q[1],-2*zperp]))
                
                Aq2 = np.cross(AArray.T[:,:2],np.array([q[0],q[1]]))

                echarge=14.4  # electron charge in V/A


                pre_factor=((mu0*muB**2)/(np.pi*electronCharge*omega*qmag))*np.sqrt(1/(2*Nt))
                
                sqrt_Nk = np.sqrt(Nk1)
                sqrt_Nk_plus_1 = np.sqrt(Nk2 + 1)

                Projection2 = (self.eigenVecs[:N, N:, numq]@(sqrt_Nk*Aq2) + self.eigenVecs[N:, N:, numq]@(sqrt_Nk_plus_1*Aq2))
                
                Projection2_0 = (self.eigenVecs[:N, N:, numq]@(sqrt_Nk*Aq1[:,0]) + self.eigenVecs[N:, N:, numq]@(sqrt_Nk_plus_1*Aq1[:,0]))
                
                Projection2_1 = (self.eigenVecs[:N, N:, numq]@(sqrt_Nk*Aq1[:,1]) + self.eigenVecs[N:, N:, numq]@(sqrt_Nk_plus_1*Aq1[:,1]))

                Projection2_2 = (self.eigenVecs[:N, N:, numq]@(sqrt_Nk*Aq1[:,2]) + self.eigenVecs[N:, N:, numq]@(sqrt_Nk_plus_1*Aq1[:,2]))

                #print(np.shape(Projection2_0),np.shape(Projection1))

                theta1+=(Projection2_0@Projection1)*gradientX

                theta1+=(Projection2_1@Projection1)*gradientY

                theta1+=(Projection2_2@Projection1)*((1.0j*lambd*1e-10)/(4*np.pi))*(gradientX2+gradientY2)  #Lambd is converted back to m here

                theta1*=sigmaf*phase*deltaz*pre_factor
                                

                #print('deltaz',deltaz)
                #print('sigmai',sigmai)
                #print('sigmaf',sigmaf)
                #print('phase',phase)
                #print('pre_factor',pre_factor)
                theta2-=sigmai*sigmaf*phase*(Projection2@Projection1)*pre_factor*potential[0,:,:]*wave._array
            #### Version 1 ####
            # for MN in range(N):
            #     for n in range(N):
            #         pre_factor=((mu0*muB**2)/(np.pi*electronCharge*omega*qmag))*np.sqrt(1/(2*Nt))

            #         Projection1=np.sqrt(Input.MagMoms[MN,0])
            #         Projection2=eigenVecs[n,N+MN,numq]*np.sqrt(Nk[MN]) + eigenVecs[N+n,N+MN,numq]*np.sqrt(Nk[MN]+1)
                    
            #         AArray=np.array([A[0,0,MN] + 1.0j*A[0,1,MN],A[1,0,MN] + 1.0j*A[1,1,MN],A[2,0,MN] + 1.0j*A[2,1,MN]])
                    
            #         Aq1 = np.cross(AArray,np.array([1.0j*q[0],1.0j*q[1],-2*zperp]))
                    
            #         theta1+=sigmaf*phase*deltaz*pre_factor*Projection1*Projection2*(Aq1[0]*gradientX + Aq1[1]*gradientY + (Aq1[2]*(1.0j*lambd/(4*np.pi))*(gradientX2+gradientY2)))
                    
            #         Aq2 = np.cross(AArray[:2],np.array([q[0],q[1]]))

            #         theta2-=sigmai*sigmaf*phase*pre_factor*Projection1*Projection2*(Aq2)*potential[0,:,:]*wave._array

            theta1_final+=theta1
            theta2_final+=theta2

        if flag_theta1:
            wave._array=theta1_final
        if flag_theta2:         
            wave._array=theta2_final
        if flag_theta1 and flag_theta2:
            wave._array=theta1_final+theta2_final    

    def __copy__(self):
        return self.__class__(atoms=self.frozen_phonons.copy(),
                              gpts=self.gpts,
                              slice_thickness=self.slice_thickness,
                              parametrization=self.parametrization,
                              cutoff_tolerance=self.cutoff_tolerance,
                              device=self.device,
                              storage=self._storage)


class PotentialArray(AbstractPotential, HasGridMixin):
    """
    Potential array object

    The potential array represents slices of the electrostatic potential as an array.

    Parameters
    ----------
    array: 3D array
        The array representing the potential slices. The first dimension is the slice index and the last two are the
        spatial dimensions.
    slice_thicknesses: float
        The thicknesses of potential slices in Å. If a float, the thickness is the same for all slices.
        If a sequence, the length must equal the length of the potential array.
    extent: one or two float, optional
        Lateral extent of the potential [Å].
    sampling: one or two float, optional
        Lateral sampling of the potential [1 / Å].
    """

    def __init__(self,
                 array: np.ndarray,
                 slice_thicknesses: Union[float, Sequence[float]],
                 mag_variables = None,
                 extent: Union[float, Sequence[float]] = None,
                 sampling: Union[float, Sequence[float]] = None):

        if (len(array.shape) != 2) & (len(array.shape) != 3):
            raise RuntimeError()

        slice_thicknesses = np.array(slice_thicknesses)

        if slice_thicknesses.shape == ():
            slice_thicknesses = np.tile(slice_thicknesses, array.shape[0])
        elif (slice_thicknesses.shape != (array.shape[0],)) & (len(array.shape) == 3):
            raise ValueError()

        self._array = array
        self._slice_thicknesses = slice_thicknesses
        self._grid = Grid(extent=extent, gpts=self.array.shape[-2:], sampling=sampling, lock_gpts=True)
        self.mag_variables = mag_variables

        #if mag_variables is not None:
        #  print(self.mag_variables)

        
        super().__init__(precalculate=False)

    def __getitem__(self, items):
        if isinstance(items, int):
            return PotentialArray(self.array[items][None], self._slice_thicknesses[items][None], mag_variables=self.magnon_package,extent=self.extent)

        elif isinstance(items, slice):
            return PotentialArray(self.array[items], self._slice_thicknesses[items], mag_variables=self.magnon_package , extent=self.extent)
        else:
            raise TypeError('Potential must indexed with integers or slices, not {}'.format(type(items)))

    def as_transmission_function(self, energy: float, in_place: bool = True, max_batch: int = 1,
                                 antialias_filter: AntialiasFilter = None):
        """
        Calculate the transmission functions for a specific energy.

        Parameters
        ----------
        energy: float
            Electron energy [eV].

        Returns
        -------
        TransmissionFunction object
        """

        xp = get_array_module(self.array)
        complex_exponential = get_device_function(xp, 'complex_exponential')

        array = self._array
        if not in_place:
            array = array.copy()

        array = complex_exponential(energy2sigma(energy) * array)

        t = TransmissionFunction(array,
                                 slice_thicknesses=self._slice_thicknesses.copy(),
                                 energy=energy)

        if antialias_filter is None:
            antialias_filter = AntialiasFilter()

        for start, end, potential_slices in t.generate_slices(max_batch=max_batch):
            t._array[start:end] = antialias_filter._bandlimit(potential_slices._array.copy())

        return t

    @property
    def num_frozen_phonon_configs(self):
        return 1

    def update_inelastic_layer(self,new_layer):
        self.mag_variables = self.mag_variables._replace(inelastic_layer=new_layer)
        return self


    def generate_frozen_phonon_potentials(self, pbar=False):
        for i in range(self.num_frozen_phonon_configs):
            yield self

    @property
    def array(self):
        """The potential array."""
        return self._array

    @property
    def num_slices(self):
        return self._array.shape[0]

    def get_slice_thickness(self, i):
        return self._slice_thicknesses[i]

    @property
    def slice_thicknesses(self):
        return self._slice_thicknesses

    @property
    def thickness(self):
        return np.sum(self._slice_thicknesses)

    def generate_slices(self, first_slice: int = 0, last_slice: int = None, max_batch: int = 1):
        if last_slice is None:
            last_slice = len(self)

        for start, end in generate_batches(last_slice - first_slice, max_batch=max_batch, start=first_slice):
            slice_thicknesses = np.array([self.get_slice_thickness(i) for i in range(start, end)])
            yield start, end, self.__class__(self.array[start:end],
                                             slice_thicknesses=slice_thicknesses,
                                             extent=self.extent)

    def tile(self, tile):
        """
        Tile the potential.

        Parameters
        ----------
        multiples: two or three int
            The number of repetitions of the potential along each axis. If three integers are given the first represents
            the number of repetitions along the z-axis.

        Returns
        -------
        PotentialArray object
            The tiled potential.
        """

        if len(tile) == 2:
            tile = tuple(tile) + (1,)

        new_array = np.tile(self.array, (tile[2], tile[0], tile[1]))

        new_extent = (self.extent[0] * tile[0], self.extent[1] * tile[1])
        new_slice_thicknesses = np.tile(self._slice_thicknesses, tile[2])

        return self.__class__(array=new_array, slice_thicknesses=new_slice_thicknesses, extent=new_extent)

    def flip(self):
        self._array = self._array[::-1]
        self._slice_thicknesses = self._slice_thicknesses[::-1]
        return self

    def write(self, path, format="hdf5", **kwargs):
        """
        Write potential to a file.

        Parameters
        ----------
        path: str
            Path to which the data is saved.
        format: str
            One of ["hdf5", "hspy"]. Default is "hdf5".
        kwargs:
            Any of the additional parameters for saving a hyperspy dataset.
        """
        if format == "hdf5":
            with h5py.File(path, 'w') as f:
                f.create_dataset('array', data=asnumpy(self.array))
                f.create_dataset('slice_thicknesses', data=self._slice_thicknesses)
                f.create_dataset('extent', data=self.extent)
        elif format == "hspy":
            self.to_hyperspy().save(**kwargs)
        else:
            raise ValueError('Format must be one of "hdf5" or "hspy"')

    @classmethod
    def read(cls, path):
        """
        Read potentia from hdf5 file.

        Parameters
        ----------
        path: str
            The file to read.

        Returns
        -------
        PotentialArray object
        """
        with h5py.File(path, 'r') as f:
            datasets = {}
            for key in f.keys():
                datasets[key] = f.get(key)[()]

        return cls(array=datasets['array'], slice_thicknesses=datasets['slice_thicknesses'], extent=datasets['extent'])


    def magnon_inelastic(self,wave,numpot,flag_theta1=True,flag_theta2=True):
        
        #self.slice_thicknesses
        
        import sympy as sym
        
        #print(self.mag_variables)


        #omega=np.linalg.norm(np.cross(self.mag_variables.atoms.cell[:][0], self.mag_variables.atoms.cell[:][1])) #* 1e-20# Was in A^2, converted to m^2
        omega=self.mag_variables.atoms.get_volume() * 1e-30

        T=self.mag_variables.Temperature

        h= 6.6256E-34                    #planks constant  J*s
        C = 299792458                    #light speed m/s
             
        M0 = 9.1091E-31                  #electron mass
        E0 = M0 * C * C                  # Rest energy (Einstein)
        E = wave.energy * electronCharge # Electron energy in J   eV*1.6*10-19 -> J

        lambd  = (h * C) / np.sqrt((2 * E * E0) + E**2) * 1E-10 # Electron Wave lenght in m

        kT=T*8.6173e-2

        ##### Calculate Gradients


        theta1_final=np.zeros_like(wave._array)
        theta2_final=np.zeros_like(wave._array)

        gradientX=np.gradient(wave._array,self.extent[0]/self.gpts[0],axis=0)  # Gradient with pixel size A
        gradientY=np.gradient(wave._array,self.extent[1]/self.gpts[1],axis=1)  # Gradient with pixel size A

        laplacian_result = ((np.roll(wave._array, 1, axis=0) - 2 * wave._array + np.roll(wave._array, -1, axis=0)) / (self.extent[0]/self.gpts[0])**2 +
                            (np.roll(wave._array, 1, axis=1) - 2 * wave._array + np.roll(wave._array, -1, axis=1)) / (self.extent[1]/self.gpts[1])**2)

        # gradientX2=np.gradient(gradientX,self.extent[0]/self.gpts[0],axis=0)  # Gradient with pixel size A
        # gradientY2=np.gradient(gradientY,self.extent[1]/self.gpts[1],axis=1)  # Gradient with pixel size A

        ##### Calculate pre-factors
        Nt=self.mag_variables.atoms.get_global_number_of_atoms()/omega # Already in A
        NtVol=self.mag_variables.atoms.get_global_number_of_atoms()/omega # Already in A

        N=len(self.mag_variables.eVals_full[0,:])//2#len(Input.atoms)

        def scos(x): return sym.N(sym.cos(x))  
        def ssin(x): return sym.N(sym.sin(x))

        #A=np.zeros([3,3,len(self.mag_variables.orientationEach)])
        A=np.zeros([3,3,len(self.mag_variables.atoms.get_initial_magnetic_moments())])

        # for numS,S in enumerate(self.mag_variables.orientationEach):
        #     A[0,0,numS]= scos(S[1])*scos(S[2]); A[0,1,numS]=-ssin(S[2]); A[0,2,numS]=ssin(S[1])*scos(S[2]) 
        #     A[1,0,numS]= scos(S[1])*ssin(S[2]); A[1,1,numS]= scos(S[2]); A[1,2,numS]=ssin(S[1])*ssin(S[2])
        #     A[2,0,numS]=-ssin(S[1])           ; A[2,1,numS]= 0         ; A[2,2,numS]=scos(S[1])

        #print(self.mag_variables.atoms.get_initial_magnetic_moments())

        for numS,S in enumerate(self.mag_variables.atoms.get_initial_magnetic_moments()):
            A[0,0,numS]= scos(S[1])*scos(S[2]); A[0,1,numS]=-ssin(S[2]); A[0,2,numS]=ssin(S[1])*scos(S[2]) 
            A[1,0,numS]= scos(S[1])*ssin(S[2]); A[1,1,numS]= scos(S[2]); A[1,2,numS]=ssin(S[1])*ssin(S[2])
            A[2,0,numS]=-ssin(S[1])           ; A[2,1,numS]= 0         ; A[2,2,numS]=scos(S[1])

        potential=self._array

        deltaz = self._slice_thicknesses[numpot] #self.thickness  ## in A 

        zperp  =(deltaz*numpot) + deltaz/2  ## Already in A
        sigmai = (2*np.pi*M0*electronCharge*lambd/h**2)#(np.pi/(lambd*wave.energy))  ## in 1/[A][eV]

        AArray=np.array([A[0,0,:] + 1.0j*A[0,1,:],A[1,0,:] + 1.0j*A[1,1,:],A[2,0,:] + 1.0j*A[2,1,:]])

        Projection1=np.sqrt(self.mag_variables.atoms.get_initial_magnetic_moments()[:,0])   ##np.sqrt(self.mag_variables.orientationEach[:,0])


        #for numq,q in enumerate(self.mag_variables.qpts*(1/np.sum(self.mag_variables.atoms.cell,axis=0))): # converting the q to a fraction of wavevector in inverse m    *(1/np.sum(Input.aseatoms.cell,axis=0)*1e-10)
        #for numq,q in enumerate(TemporaryPath['explicit_kpoints_abs']): # converting the q to a fraction of wavevector in inverse m    *(1/np.sum(Input.aseatoms.cell,axis=0)*1e-10)
        for numq,q in enumerate(self.mag_variables.qpts*2*np.pi*(1/4)): # converting the q to a fraction of wavevector in inverse m    *(1/np.sum(Input.aseatoms.cell,axis=0)*1e-10)

            #print(q,numq)
            
            theta1=np.zeros_like(wave._array)
            theta2=np.zeros_like(wave._array)

            qmag=np.linalg.norm(q[:2])* 1e-10
            qmagVol=np.linalg.norm(q[:])* 1e-10
            
            #pre_factor=((mu0*muB**2)/(np.pi*electronCharge*omega*qmag))*np.sqrt(1/(2*Nt))
            #Projection1=np.exp(1.0j*Input.atoms[:,3].astype(float)*Input.qpts[numq,2])*np.sqrt(Input.MagMoms[:,0])
            #Projection2=eigenVecs[:N,N:,numq]*np.sqrt(Nk) + eigenVecs[N:,N:,numq]*np.sqrt(Nk+1)

            ##### Calculate occupation Numbers
            
            Nk1 = 1/(np.exp(np.abs(self.mag_variables.eVals_full[numq,:N])/kT)-1)#np.ones_like(1/(np.exp(eigenVals[numq,:]/kT)-1))   
            Nk2 = 1/(np.exp(np.abs(self.mag_variables.eVals_full[numq,N:])/kT)-1)         
            #phase=np.exp(1.0j*zperp*self.mag_variables.qpts[numq,2])
            #phase=np.exp(1.0j*zperp*q[2])
            

            sigmaf = sigmai#np.pi/(((lambd)-(2*np.pi/(qmag)))*(E0))

            if  True:#self.mag_variables._atoms._z_periodic==True:

                # #### Version 4 ####
                
                Aq1 = np.cross(AArray.T,np.array([q[0],q[1],q[2]]))
                
                # print('AArray.T[:,:2]',AArray.T[:,:2])
                # print('np.array([q[0],q[1]])',np.array([q[0],q[1]]))


                #Aq2 = np.cross(AArray.T[:,:2],np.array([q[0],q[1]]))

                echarge=14.4  # electron charge in V/A


                pre_factor=((mu0*muB**2)/(np.pi*electronCharge*omega*qmagVol))*np.sqrt(1/(2*NtVol))
                
                sqrt_Nk = np.sqrt(Nk1)
                sqrt_Nk_plus_1 = np.sqrt(Nk1 + 1)

                # print('Nk1',Nk1)
                # print('Nk2',Nk2)

                # print(Aq2)

                # print('A=',(self.mag_variables.eVecs_fullR[:N, N:, numq]))
                # print('B=',(sqrt_Nk_plus_1*Aq2))
                #print(np.shape(Aq1))
                #print(np.shape(sqrt_Nk_plus_1))

                Projection2 = (self.mag_variables.eVecs_fullR[N:, N:, numq]@(sqrt_Nk_plus_1*Aq1[:,2]) + self.mag_variables.eVecs_fullR[N:, :N, numq]@(sqrt_Nk*Aq1[:,2]))
                
                Projection2_0 = (self.mag_variables.eVecs_fullR[N:, N:, numq]@(sqrt_Nk_plus_1*Aq1[:,0]) + self.mag_variables.eVecs_fullR[N:, :N, numq]@(sqrt_Nk*Aq1[:,0]))
                
                Projection2_1 = (self.mag_variables.eVecs_fullR[N:, N:, numq]@(sqrt_Nk_plus_1*Aq1[:,1]) + self.mag_variables.eVecs_fullR[N:, :N, numq]@(sqrt_Nk*Aq1[:,1]))

                Projection2_2 = (self.mag_variables.eVecs_fullR[N:, N:, numq]@(sqrt_Nk_plus_1*Aq1[:,2]) + self.mag_variables.eVecs_fullR[N:, :N, numq]@(sqrt_Nk*Aq1[:,2]))

                #print(np.shape(Projection2_0),np.shape(Projection1))

                #print(np.shape(Projection2_0@Projection1),np.shape(gradientX))

                theta1+=(Projection2_0@Projection1)*gradientX

                theta1+=(Projection2_1@Projection1)*gradientY

                theta1+=(Projection2_2@Projection1)*((1.0j*lambd)/(4*np.pi))*laplacian_result#(gradientX2+gradientY2)  #Lambd is converted back to m here

                theta1*=1.0j*sigmaf*deltaz*pre_factor
                                

                #print('deltaz',deltaz)
                #print('sigmai',sigmai)
                #print('sigmaf',sigmaf)
                #print('phase',phase)
                #print('pre_factor',pre_factor)
                theta2-=sigmai*sigmaf*(Projection2@Projection1)*pre_factor*potential[0,:,:]*wave._array

                #### Version 1 ####
                # for MN in range(N):
                #     for n in range(N):
                #         pre_factor=((mu0*muB**2)/(np.pi*electronCharge*omega*qmag))*np.sqrt(1/(2*Nt))

                #         Projection1=np.sqrt(Input.MagMoms[MN,0])
                #         Projection2=eigenVecs[n,N+MN,numq]*np.sqrt(Nk[MN]) + eigenVecs[N+n,N+MN,numq]*np.sqrt(Nk[MN]+1)
                        
                #         AArray=np.array([A[0,0,MN] + 1.0j*A[0,1,MN],A[1,0,MN] + 1.0j*A[1,1,MN],A[2,0,MN] + 1.0j*A[2,1,MN]])
                        
                #         Aq1 = np.cross(AArray,np.array([1.0j*q[0],1.0j*q[1],-2*zperp]))
                        
                #         theta1+=sigmaf*phase*deltaz*pre_factor*Projection1*Projection2*(Aq1[0]*gradientX + Aq1[1]*gradientY + (Aq1[2]*(1.0j*lambd/(4*np.pi))*(gradientX2+gradientY2)))
                        
                #         Aq2 = np.cross(AArray[:2],np.array([q[0],q[1]]))

                #         theta2-=sigmai*sigmaf*phase*pre_factor*Projection1*Projection2*(Aq2)*potential[0,:,:]*wave._array

                theta1_final+=theta1
                theta2_final+=theta2

            if flag_theta1:
                wave._array=theta1_final
            if flag_theta2:         
                wave._array=theta2_final
            if flag_theta1 and flag_theta2:
                wave._array=theta1_final+theta2_final


            else:

                # #### Version 4 ####
                
                Aq1 = np.cross(AArray.T,np.array([1.0j*q[0],1.0j*q[1],-2*zperp]))
                
                Aq2 = np.cross(AArray.T[:,:2],np.array([q[0],q[1]]))

                echarge=14.4  # electron charge in V/A


                pre_factor=((mu0*muB**2)/(np.pi*electronCharge*omega*qmag))*np.sqrt(1/(2*Nt))
                
                sqrt_Nk = np.sqrt(Nk1)
                sqrt_Nk_plus_1 = np.sqrt(Nk2 + 1)

                Projection2 = (self.mag_variables.eVecs_fullR[N:, :N, numq]@(sqrt_Nk*Aq2) + self.mag_variables.eVecs_fullR[N:, N:, numq]@(sqrt_Nk_plus_1*Aq2))
                
                Projection2_0 = (self.mag_variables.eVecs_fullR[N:, :N, numq]@(sqrt_Nk*Aq1[:,0]) + self.mag_variables.eVecs_fullR[N:, N:, numq]@(sqrt_Nk_plus_1*Aq1[:,0]))
                
                Projection2_1 = (self.mag_variables.eVecs_fullR[N:, :N, numq]@(sqrt_Nk*Aq1[:,1]) + self.mag_variables.eVecs_fullR[N:, N:, numq]@(sqrt_Nk_plus_1*Aq1[:,1]))

                Projection2_2 = (self.mag_variables.eVecs_fullR[N:, :N, numq]@(sqrt_Nk*Aq1[:,2]) + self.mag_variables.eVecs_fullR[N:, N:, numq]@(sqrt_Nk_plus_1*Aq1[:,2]))

                #print(np.shape(Projection2_0),np.shape(Projection1))

                #print(np.shape(Projection2_0@Projection1),np.shape(gradientX))

                theta1+=(Projection2_0@Projection1)*gradientX

                theta1+=(Projection2_1@Projection1)*gradientY

                theta1+=(Projection2_2@Projection1)*((1.0j*lambd*1e-10)/(4*np.pi))*(gradientX2+gradientY2)  #Lambd is converted back to m here

                theta1*=sigmaf*deltaz*pre_factor
                                

                #print('deltaz',deltaz)
                #print('sigmai',sigmai)
                #print('sigmaf',sigmaf)
                #print('phase',phase)
                #print('pre_factor',pre_factor)
                theta2-=sigmai*sigmaf*(Projection2@Projection1)*pre_factor*potential[0,:,:]*wave._array

                #### Version 1 ####
                # for MN in range(N):
                #     for n in range(N):
                #         pre_factor=((mu0*muB**2)/(np.pi*electronCharge*omega*qmag))*np.sqrt(1/(2*Nt))

                #         Projection1=np.sqrt(Input.MagMoms[MN,0])
                #         Projection2=eigenVecs[n,N+MN,numq]*np.sqrt(Nk[MN]) + eigenVecs[N+n,N+MN,numq]*np.sqrt(Nk[MN]+1)
                        
                #         AArray=np.array([A[0,0,MN] + 1.0j*A[0,1,MN],A[1,0,MN] + 1.0j*A[1,1,MN],A[2,0,MN] + 1.0j*A[2,1,MN]])
                        
                #         Aq1 = np.cross(AArray,np.array([1.0j*q[0],1.0j*q[1],-2*zperp]))
                        
                #         theta1+=sigmaf*phase*deltaz*pre_factor*Projection1*Projection2*(Aq1[0]*gradientX + Aq1[1]*gradientY + (Aq1[2]*(1.0j*lambd/(4*np.pi))*(gradientX2+gradientY2)))
                        
                #         Aq2 = np.cross(AArray[:2],np.array([q[0],q[1]]))

                #         theta2-=sigmai*sigmaf*phase*pre_factor*Projection1*Projection2*(Aq2)*potential[0,:,:]*wave._array

                theta1_final+=theta1
                theta2_final+=theta2

            if flag_theta1:
                wave._array=theta1_final
            if flag_theta2:         
                wave._array=theta2_final
            if flag_theta1 and flag_theta2:
                wave._array=theta1_final+theta2_final 
    
        return wave

    def transmit(self, waves, conjugate=False):
        """
        Transmit a wave function.

        Parameters
        ----------
        waves: Waves object
            Wave function to transmit.

        Returns
        -------
        TransmissionFunction
        """
        return self.as_transmission_function(waves.energy).transmit(waves, conjugate=conjugate)

    def project(self):
        """
        Create a 2d measurement of the projected potential.

        Returns
        -------
        Measurement
        """
        calibrations = calibrations_from_grid(self.grid.gpts, self.grid.sampling, names=['x', 'y'])
        array = asnumpy(self.array.sum(0))
        array -= array.min()
        return Measurement(array, calibrations)

    def __copy___(self):
        return self.__class__(array=self.array.copy(),
                              slice_thicknesses=self._slice_thicknesses.copy(),
                              extent=self.extent)

    def to_hyperspy(self):
        """
        Changes the PotentialArray object to a `hyperspy.Signal2D` Object.
        """
        from hyperspy._signals.signal2d import Signal2D
        signal_shape = self.array.shape
        axes = []

        # as the first dimension is always the thickness that is added first
        axes.append({"offset": 0,
                     "scale": self.thickness / self.num_slices,
                     "units": "Å",
                     "name": "z",
                     "size": self.num_slices})

        # loop for x and y axes
        for i, size in zip(self.project().calibrations, signal_shape[1:]):
            if i is None:
                axes.append({"offset": 0,
                             "scale": 1,
                             "units": "",
                             "name": "",
                             "size": size})
            else:
                axes.append({"offset": i.offset,
                             "scale": i.sampling,
                             "units": i.units,
                             "name": i.name,
                             "size": size})

        sig = Signal2D(self.array, axes=axes)

        return sig


class TransmissionFunction(PotentialArray, HasAcceleratorMixin):
    """Class to describe transmission functions."""

    def __init__(self,
                 array: np.ndarray,
                 slice_thicknesses: Union[float, Sequence[float]],
                 extent: Union[float, Sequence[float]] = None,
                 sampling: Union[float, Sequence[float]] = None,
                 energy: float = None):
        self._accelerator = Accelerator(energy=energy)
        super().__init__(array, slice_thicknesses, extent, sampling)

    def as_transmission_function(self, energy, in_place=True, max_batch=1, antialias_filter=None):
        if energy != self.energy:
            raise RuntimeError()

        return self

    def generate_transmission_functions(self, energy, first_slice=0, last_slice=None, max_batch=1):
        if energy != self.energy:
            raise RuntimeError()
        return self.generate_slices(first_slice=first_slice, last_slice=last_slice, max_batch=max_batch)

    def transmit(self, waves, conjugate=False):
        self.accelerator.check_match(waves)
        xp = get_array_module(waves._array)

        if len(waves.array.shape) == 2:

            if self.array.shape[0] == 1:
                array = self.array[0]
            else:
                raise RuntimeError()

        else:
            array = self.array

        if conjugate:
            waves._array *= xp.conjugate(copy_to_device(array, xp))
        else:
            waves._array *= copy_to_device(array, xp)
        return waves
