import pickle
import torch
from os.path import join
import edm.qm9.visualizer as vis
import edm.utils as utils
from edm.configs.datasets_config import get_dataset_info
from edm.qm9 import dataset
from edm.qm9.analyze import check_stability
from edm.qm9.models import get_model, get_optim
from edm.qm9.sampling import sample

class DiSco:
    def __init__(self, disco_args, model_path='edm/outputs/edm_qm9'):
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
        flow, nodes_dist, prop_dist = get_model(self.args, self.device, self.dataset_info, dataloaders['train'])
        flow.to(self.device)
        fn = 'generative_model_ema.npy' if self.args.ema_decay > 0 else 'generative_model.npy'
        flow_state_dict = torch.load(join(model_path, fn), map_location=self.device)
        flow.load_state_dict(flow_state_dict)
        self.model = flow
        self.nodes_dist = nodes_dist

        self.disco_args = disco_args

    def disco_diffuse(self):
        # Sample molecules from diffusion model
        nodesxsample = torch.full((self.disco_args['n_tries'], 1), self.disco_args['n_nodes'])
        x, h, node_mask, edge_mask = sample(self.args, self.device, self.model, self.dataset_info, nodesxsample=nodesxsample)
        one_hot = h['categorical']
        charges = h['integer']

        # Find stable molecules
        filenames_stable = []
        idx_stable = []
        stable_counter = 0
        for i in range(self.disco_args['n_tries']):
            num_atoms = int(node_mask[i:i+1].sum().item())
            atom_type = one_hot[i:i+1, :num_atoms].argmax(2).squeeze(0).cpu().detach().numpy()
            x_squeeze = x[i:i+1, :num_atoms].squeeze(0).cpu().detach().numpy()
            mol_stable = check_stability(x_squeeze, atom_type, self.dataset_info)[0]

            if mol_stable:
                filename = f'{self.output_dir}/molecule_{stable_counter+1}.xyz'
                vis.save_xyz_file(filename, one_hot[i:i+1], charges[i:i+1], x[i:i+1], self.dataset_info, node_mask=node_mask[i:i+1])
                filenames_stable.append(filename)
                idx_stable.append(i)
                stable_counter += 1
                
            if stable_counter == self.disco_args['n_samples']:
                break 

        # Return positions and categories of stable molecules
        x_stable = x[idx_stable, :, :]
        h_stable = {}
        h_stable['integer'] = h['integer'][idx_stable, :, :]
        h_stable['categorical'] = h['categorical'][idx_stable, :, :]
        node_mask_stable = node_mask[idx_stable, :, :]
        edge_mask_stable = edge_mask.reshape(self.disco_args['n_tries'], self.dataset_info['max_n_nodes'], self.dataset_info['max_n_nodes'])[idx_stable, :, :] # TODO: is this correct?

        return x_stable, h_stable, node_mask_stable, edge_mask_stable, filenames_stable

    def disco_score(self, filenames):
        scores = torch.tensor([self.score_func(filename) for filename in filenames])

        return scores

    def disco_align(self, x, h, scores, node_mask, edge_mask):
        # Standardize scores
        scores = (scores - scores.mean()) / scores.std()

        # Optimization objective
        if self.disco_args['objective'] == 'maximize':
            scores = scores
        elif self.disco_args['objective'] == 'minimize':
            scores = -scores

        # Weighted scores
        exp_scores = torch.exp(self.disco_args['beta'] * scores)
        Z = torch.sum(exp_scores)
        weights = exp_scores / Z
        weights = weights.to(self.device)

        # Train
        losses = []
        optimizer = get_optim(self.args, self.model)
        self.model.train()
        for epoch in range(self.disco_args['num_epochs']):
            optimizer.zero_grad()
            neg_log_pxh = self.model.compute_nll(x, h, node_mask=node_mask, edge_mask=edge_mask)
            weighted_loss = (neg_log_pxh * weights).mean()
            weighted_loss.backward()
            optimizer.step()

            losses.append(weighted_loss.item())
        
        losses = torch.tensor(losses)
        
        return losses

    def disco_step(self):
        x, h, node_mask, edge_mask, filenames = self.disco_diffuse()
        scores = self.disco_score(filenames)
        losses = self.disco_align(x, h, scores, node_mask, edge_mask)

        return scores, losses

    def run(self):
        for cycle in range(self.disco_args['disco_cycles']):
            scores, losses = self.disco_step()

            # Save training data
            cycle_dir = f"{self.disco_args['output_dir']}/cycle_{cycle+1}"
            torch.save(scores, f'{cycle_dir}/scores.pt')
            torch.save(losses, f'{cycle_dir}/losses.pt')