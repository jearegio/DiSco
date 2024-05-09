import os

# Set the number of OpenMP threads
os.environ["OMP_NUM_THREADS"] = "1"
print("PID: ", os.getpid())

from disco import DiSco
from disco_args import DiScoArgs

# Set up training arguments
disco_args = DiScoArgs(
    disco_cycles=100,
    objective='maximize', 
    n_nodes=10,
    n_samples=10, 
    n_tries=100,
    num_epochs=100, 
    beta=1,
    scorer='leftnet',
    device='cuda:2')

# Run DiSco
disco = DiSco(disco_args)
disco.run()