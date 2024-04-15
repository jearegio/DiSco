import functools
from disco import DiSco
from disco_args import DiScoArgs
from score_funcs import calc_energy

# Set up scoring function
basis = 'sto-3g'
xc = 'b3lyp'
score_func = functools.partial(calc_energy, basis=basis, xc=xc)

# Set up training arguments
disco_args = DiScoArgs(
    disco_cycles=2,
    objective='minimize', 
    n_nodes=10,
    n_samples=200, 
    n_tries=2000,
    num_epochs=100, 
    score_func=score_func,
    beta=1,)

# Run DiSco
disco = DiSco(disco_args)
disco.run()