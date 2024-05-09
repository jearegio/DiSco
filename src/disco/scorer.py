import torch
import json
from abc import ABC, abstractmethod
from pyscf import dft, gto
from pyscf.tools import cubegen
import numpy as np
from torch_geometric.data import Data
# import paddle
# from pahelix.model_zoo.gem_model import GeoGNNModel
# from gem.src.model import DownstreamModel
# from gem.src.featurizer import DownstreamCollateFn, DownstreamTransformFn
# from gem.src.utils import get_downstream_task_names
from rdkit import Chem
from rdkit.Geometry import Point3D
from LeftNet.utils import EnsembleModel

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
        mf, mol = self.calc(filename)
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
        elif self.metric == 'coper':
            return self.calc_coper(mf, mol)
        # elif self.metric == 'polarizability':
        #     return self.calc_polarizability(mf)
        else:
            raise ValueError('Invalid metric.')

    def calc(self, filename):
        # Convert to PySCF
        mol = gto.M(atom=filename, basis=self.basis)

        # Calculate energy
        mf = dft.RKS(mol)
        mf.xc = self.xc
        mf.kernel()
        
        return mf, mol

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

    def calc_coper(self, mf, mol, l=100, outfile="out.cube"):
        density = mf.make_rdm1()
        cube = cubegen.density(mol, outfile, density, nx=l, ny=l, nz=l)

        z = np.sum(cube)    # partition function
        cube /= z           # normalize densities
        entropy = -np.sum(cube * np.log(cube))   

        coper = np.exp(entropy - (3 * np.log(l)))

        return coper

    # def calc_polarizability(self, mf):
    #     polar = Polarizability(mf)
    #     pol_tensor = polar.kernel()
    #     pol_norm = np.linalg.norm(pol_tensor)

    #     return pol_norm

class GEMScorer(Scorer):
    def __init__(self, metric):
        compound_encoder_config = load_json("src/disco/gem/model_configs/geognn_l8.json")
        model_config = load_json("src/disco/gem/model_configs/down_mlp3.json")

        task_names = get_downstream_task_names(metric, None)

        task_type = 'regr'
        model_config['task_type'] = task_type
        model_config['num_tasks'] = len(task_names)
        print('model_config:')
        print(model_config)

        ### build model
        compound_encoder = GeoGNNModel(compound_encoder_config)
        compound_encoder.set_state_dict(paddle.load(f"src/disco/gem/models/{metric}/compound_encoder.pdparams"))
        model = DownstreamModel(model_config, compound_encoder)
        model.set_state_dict(paddle.load(f"src/disco/gem/models/{metric}/model.pdparams"))
        model.eval()

        # set up collate function
        collate_fn = DownstreamCollateFn(
            atom_names=compound_encoder_config['atom_names'], 
            bond_names=compound_encoder_config['bond_names'],
            bond_float_names=compound_encoder_config['bond_float_names'],
            bond_angle_float_names=compound_encoder_config['bond_angle_float_names'],
            task_type=model_config['task_type'])

        self.metric = metric
        self.model = model
        self.collate_fn = collate_fn

    def read_xyz_file(self, filename):
        mol = Chem.RWMol()  # Create an editable RDKit molecule

        with open(filename, 'r') as file:
            lines = file.readlines()
            num_atoms = int(lines[0].strip())  # First line is the number of atoms

            # Initialize the conformer once we know the number of atoms
            conf = Chem.Conformer(num_atoms)

            for i in range(2, 2 + num_atoms):  # Start reading atoms from line 2
                parts = lines[i].strip().split()
                element = parts[0]
                x, y, z = map(float, parts[1:4])  # Convert strings to floats

                # Add atom to molecule
                atom = Chem.Atom(element)
                atom_idx = mol.AddAtom(atom)

                # Set atom coordinates
                conf.SetAtomPosition(atom_idx, Point3D(x, y, z))

            # Add the conformer to the molecule after all atoms are added and positions set
            mol.AddConformer(conf, assignId=True)

        # Sanitize the molecule to calculate all required properties like valence
        Chem.SanitizeMol(mol)

        return mol.GetMol()  # Convert to immutable RDKit molecule

    def predict(self, mol):
        featurizer_obj = DownstreamTransformFn(is_inference=True)
        graph_data = featurizer_obj(mol)  # Modified to take RDKit Mol directly
        
        # Prepare graphs for model input
        atom_bond_graph, bond_angle_graph, _ = self.collate_fn([graph_data])
        # Predict using the model
        prediction = self.model(atom_bond_graph, bond_angle_graph)

        return prediction

    def score(self, filename):
        mol = self.read_xyz_file(filename)
        prediction = self.predict(mol).item()
        
        return prediction

def load_json(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)

    return data

class LeftNetScorer(Scorer):
    def __init__(self, device):
        self.model = EnsembleModel(sweep_id="ov9qdqml", num_models=5).to("cpu")
        self.model.load_state_dict(torch.load("src/disco/LeftNet/ensemble_model_weights.pth", map_location='cpu'))
        print("Loaded ensemble model")
        self.model.eval()

        self.device = device

    def score(self, filename):
        data = self.parse_xyz(filename).to(self.device)
        with torch.no_grad():
            out = self.model(data.z, data.pos, data.batch).item()

        return out
        
    def parse_xyz(self, file_path):
        # QM9 EDM only has 5 types of atoms
        atom_dict = {
            'H': 1,    # Hydrogen
            'C': 6,    # Carbon
            'N': 7,    # Nitrogen
            'O': 8,    # Oxygen
            'F': 9,    # Fluorine
        }

        with open(file_path, 'r') as file:
            lines = file.readlines()
            num_atoms = int(lines[0].strip())
            z = []
            pos = []
            for line in lines[2:2+num_atoms]:  # Skip the first two lines
                parts = line.split()
                z.append(atom_dict[parts[0]])  # Element type
                pos.append([float(parts[1]), float(parts[2]), float(parts[3])])

            z = torch.tensor(z, dtype=torch.long)
            pos = torch.tensor(pos, dtype=torch.float)
            batch = torch.zeros(size=(z.size()), dtype=torch.int64)

            data = Data(z=z, pos=pos, batch=batch)

        return data