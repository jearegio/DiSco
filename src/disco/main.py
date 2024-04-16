import functools
from disco import DiSco
from disco_args import DiScoArgs

# Set up training arguments
disco_args = DiScoArgs(
    disco_cycles=25,
    objective='maximize', 
    n_nodes=10,
    n_samples=100, 
    n_tries=1000,
    num_epochs=100, 
    beta=1,
    metric='energy',
    basis='sto-3g',
    xc='b3lyp')

# Run DiSco
disco = DiSco(disco_args)
disco.run()