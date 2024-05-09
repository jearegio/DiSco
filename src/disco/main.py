import os

# Set the number of OpenMP threads
os.environ["OMP_NUM_THREADS"] = "1"
print("PID: ", os.getpid())

from disco import DiSco
from disco_args import DiScoArgs
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument('--disco_cycles', type=int, help='Number of DiSco cycles')
    parser.add_argument('--objective', type=str, help='Maximize or minimize')
    parser.add_argument('--n_nodes', type=int, help='Number of atoms')
    parser.add_argument('--n_samples', type=int, help='Number of samples per cycle')
    parser.add_argument('--n_tries', type=int, help='Number of total sample attempts')
    parser.add_argument('--num_epochs', type=int, help='Number of training epochs')
    parser.add_argument('--beta', type=int, help='Temperature for reward weights')
    parser.add_argument('--scorer', type=str, help='pyscf, gem, or leftnet')
    parser.add_argument('--device', type=str, help='Device to run')
    args = parser.parse_args()

    # Set up training arguments
    disco_args = DiScoArgs(
        disco_cycles=args.disco_cycles,
        objective=args.objective, 
        n_nodes=args.n_nodes,
        n_samples=args.n_samples, 
        n_tries=args.n_tries,
        num_epochs=args.num_epochs, 
        beta=args.beta,
        scorer=args.scorer,
        device=args.device)

    # Run DiSco
    disco = DiSco(disco_args)
    disco.run()