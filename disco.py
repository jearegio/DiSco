import torch
import utils
from qm9 import dataset
from qm9.models import get_model
import torch
import pickle
import qm9.visualizer as vis
from qm9.analyze import check_stability
from os.path import join
from qm9.sampling import sample
from configs.datasets_config import get_dataset_info
import os
from datetime import datetime
from pyscf import gto, dft

def calc_energy(mol, xc):
    mf = dft.RKS(mol)
    mf.xc = xc
    energy = mf.kernel()

    return energy

class Disco:
    def __init__(self, basis, xc, model_path='outputs/edm_qm9'):
        with open(join(model_path, 'args.pickle'), 'rb') as f:
            self.args = pickle.load(f)
        utils.create_folders(self.args)

        # CAREFUL with this -->
        if not hasattr(self.args, 'normalization_factor'):
            self.args.normalization_factor = 1
        if not hasattr(self.args, 'aggregation_method'):
            self.args.aggregation_method = 'sum'

        # initialize device
        self.args.cuda = not self.args.no_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.args.cuda else "cpu")
        self.args.device = self.device

        # initialize data
        self.dataset_info = get_dataset_info(self.args.dataset, self.args.remove_h)
        dataloaders, charge_scale = dataset.retrieve_dataloaders(self.args)

        # initialize diffusion model
        flow, _, _ = get_model(self.args, self.device, self.dataset_info, dataloaders['train'])
        flow.to(self.device)
        fn = 'generative_model_ema.npy' if self.args.ema_decay > 0 else 'generative_model.npy'
        flow_state_dict = torch.load(join(model_path, fn), map_location=self.device)
        flow.load_state_dict(flow_state_dict)

        self.output_dir = 'outputs/disco'
        self.model = flow
        self.basis = basis
        self.xc = xc

    def disco_sample(self, n_tries, n_nodes):
        nodesxsample = torch.full((n_tries, 1), n_nodes)
        x, h, node_mask = sample(self.args, self.device, self.model, self.dataset_info, nodesxsample=nodesxsample)
        one_hot = h['categorical']
        charges = h['integer']

        filenames_stable = []
        x_stable, h_stable = torch.empty(), torch.empty()
        for i in range(n_tries):
            num_atoms = int(node_mask[i:i+1].sum().item())
            atom_type = one_hot[i:i+1, :num_atoms].argmax(2).squeeze(0).cpu().detach().numpy()
            x_squeeze = x[i:i+1, :num_atoms].squeeze(0).cpu().detach().numpy()
            mol_stable = check_stability(x_squeeze, atom_type, self.dataset_info)[0]

            if mol_stable:
                filename = f'{self.output_dir}/molecule_{len(filenames_stable)}.xyz'
                vis.save_xyz_file(filename, one_hot[i:i+1], charges[i:i+1], x[i:i+1], self.dataset_info, node_mask=node_mask[i:i+1])
                filenames_stable.append(filename)
                x_stable = torch.cat(x_stable, x[i])
                h_stable = torch.cat(h_stable, h[i])

    def disco_align(self, x, h, scores, node_mask, edge_mask, context, beta, lr=0.001, num_epochs=10):
        optimizer = torch.optim.Adam(lr=lr)

        # Weighted evaluations
        exp_scores = torch.exp(beta * scores)
        Z = torch.sum(exp_scores)
        weights = exp_scores / Z

        self.model.train()
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            neg_log_pxh = self.model(x, h, node_mask=node_mask, edge_mask=edge_mask, context=context)
            weighted_loss = (neg_log_pxh * weights).mean()
            weighted_loss.backward()
            optimizer.step()

            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {weighted_loss.item()}')

    def disco_score(self, filenames):
        mols = [gto.M(atom=filename, basis=self.basis) for filename in filenames]
        energies = torch.tensor([calc_energy(mol, xc=self.xc) for mol in mols])

        return energies

    def disco_step(self, n_tries, beta, num_epochs):
        x, h, filenames = self.disco_sample(n_tries)
        scores = self.disco_score(filenames)
        self.disco_align(x, h, scores, node_mask, edge_mask, beta, num_epochs)

    def disco(self, disco_cycles):
        for cycle in disco_cycles:
            self.disco_step()