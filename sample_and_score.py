from eval_sample import disco_sample
from pyscf import gto, dft
import pandas as pd
from datetime import datetime

def calc_energy(mol, xc):
    mf = dft.RKS(mol)
    mf.xc = xc
    energy = mf.kernel()

    return energy

def sample_and_calculate(n_nodes, n_tries, basis, xc):
    now = datetime.now()
    timestamp = now.strftime("%m-%d-%Y_%H-%M-%S")
    output_dir = f'outputs/disco/{timestamp}'

    print("Sampling...")
    stable_filenames, unstable_filenames = disco_sample(n_nodes=n_nodes, n_tries=n_tries, output_dir=output_dir)

    print("Calculating...")
    mols = [gto.M(atom=filename, basis=basis) for filename in stable_filenames]
    energies = [calc_energy(mol, xc=xc) for mol in mols]
    
    print("Saving...")
    df = pd.DataFrame({'filenames': stable_filenames, 'energy': energies})
    df.to_pickle(f'{output_dir}/df.pkl')

n_nodes = 10
n_tries = 1000
basis = 'sto-3g'
xc = 'b3lyp'
sample_and_calculate(n_nodes=n_nodes, n_tries=n_tries, basis=basis, xc=xc)
