from pyscf import dft, gto
import numpy as np
from pyscf.prop.polarizability.rks import Polarizability
from abc import ABC, abstractmethod

class Scorer(ABC):
    @abstractmethod
    def score(self, filename):
        pass

class PySCFScorer(Scorer):
    def __init__(self, metric, basis, xc):
        self.metric = metric
        self.basis = basis
        self.xc = xc

    def score(self, filename):
        mf = self.calc(filename)
        if self.metric == 'energy':
            return self.calc_energy(mf)
        elif self.metric == 'homo_lumo_gap':
            return self.calc_homo_lomo_gap(mf)
        elif self.metric == 'chemical_potential':
            return self.calc_chemical_potential(mf)
        elif self.metric == 'ionization_potential':
            return self.calc_ionization_potential(mf)
        elif self.metric == 'electron_affinity':
            return self.calc_electron_affinity(mf)
        elif self.metric == 'dipole_magnitude':
            return self.calc_dipole_magnitude(mf)
        elif self.metric == 'polarizability':
            return self.calc_polarizability(mf)
        else:
            raise ValueError('Invalid metric.')

    def calc(self, filename):
        # Convert to PySCF
        mol = gto.M(atom=filename, basis=self.basis)

        # Calculate energy
        mf = dft.RKS(mol)
        mf.xc = self.xc
        mf.kernel()
        
        return mf

    def calc_energy(self, mf):
        energy = mf.e_tot

        return energy

    def calc_homo_lumo(self, mf):
        mo_energy = mf.mo_energy
        homo = max(mo_energy[mf.mo_occ > 0])
        lumo = min(mo_energy[mf.mo_occ == 0])
        
        return homo, lumo

    def calc_homo_lomo_gap(self, mf):
        homo, lumo = self.calc_homo_lumo(mf)

        return lumo - homo

    def calc_chemical_potential(self, mf):
        homo = max(mf.mo_energy[mf.mo_occ > 0])
        lumo = min(mf.mo_energy[mf.mo_occ == 0])
        chemical_potential = -0.5 * (homo + lumo)
        
        return chemical_potential

    def calc_ionization_potential(self, mf):
        homo, lumo = self.calc_homo_lumo(mf)

        return -homo

    def calc_electron_affinity(self, mf):
        homo, lumo = self.calc_homo_lumo(mf)

        return -lumo

    def calc_dipole_magnitude(self, mf):
        dipole_moment = mf.dip_moment()
        dipole_magnitude = np.linalg.norm(dipole_moment)

        return dipole_magnitude

    def calc_polarizability(self, mf):
        polar = Polarizability(mf)
        pol_tensor = polar.kernel()
        pol_norm = np.linalg.norm(pol_tensor)

        return pol_norm