import os

# Set the number of OpenMP threads
os.environ["OMP_NUM_THREADS"] = "4"

from disco import DiSco
from disco_args import DiScoArgs

# Set up training arguments
disco_args = DiScoArgs(
    disco_cycles=100,
    objective='minimize', 
    n_nodes=10,
    n_samples=100, 
    n_tries=1000,
    num_epochs=100, 
    beta=1,
    metric='coper',
    basis='sto-3g',
    xc='b3lyp',
    device='cuda:2')

# Run DiSco
disco = DiSco(disco_args)
disco.run()