from pyscf import dft, gto

def calc_energy(filename, basis, xc):
    # Convert to PySCF
    mol = gto.M(atom=filename, basis=basis)

    # Calculate energy
    mf = dft.RKS(mol)
    mf.xc = xc
    energy = mf.kernel()

    return energy