'''
Closed-shell Restricte Hartree-Fock Method.

Demonstrate that the libint2py API platforom is convenient for algorithm implementation. 

By Ye @ Oct 2017
'''
from pylibint import molecule
import numpy as np
from scipy.linalg import eigh as generalized_eigensolution


class RHF:
    '''
    Attributes:
    -----------
    [orbital_energies]      numpy.array 
    [orbital_coefficients]  numpy.array
    [density_matrix]        numpy.array
    [energy]                int         E_RHF = E_elec + E_nuc

    '''
    def __init__(self, mole, epsilon=1e-10, max_iter=1000, verbose=False, initial_density_matrix=None):
        assert mole.nele % 2 == 0               # Closed-shell checking

        H_core = mole.core_hamiltonian_matrix() # H^core_ij
        G = mole.eletronic_repulsion_matrix()   # G_ijkl = (ij|kl)

        if verbose: print('-- Initial Density Matrix --')
        if initial_density_matrix is not np.array:
            P = np.zeros((mole.norb,)*2)            # Initial guess of density matrix.
            np.fill_diagonal(P[:mole.nele//2,:mole.nele//2], 2)
        else:
            P = initial_density_matrix
        if verbose: print(P)
        
        for i in range(max_iter):               # SCF
            B = np.einsum('kl,ijlk->ij', P, G)-.5*np.einsum('kl,ilkj->ij',P,G)
            Fock_matrix = H_core + B
            E_elec = .5 * np.einsum('ij,ji->', P, H_core + Fock_matrix)
            if verbose: print('Iter =', i, '\tE_elec =', E_elec)
            E, C = generalized_eigensolution(Fock_matrix, mole.overlap_matrix)
            P1 = 2. * C[:,:mole.nele//2].dot(C[:,:mole.nele//2].T)
            if np.linalg.norm(P1 - P)/mole.norb < epsilon: break
            P = (P1+P)/2.
        else:
            print('Warning: Unconverged after',max_iter,'SCF iterations.')

        self.orbital_energies = E
        self.orbital_coefficients = C
        self.density_matrix = P
        self.energy = E_elec + mole.nuclear_repulsion
             
if __name__ == '__main__':
    np.set_printoptions(linewidth=np.nan,threshold=np.nan)
    A = molecule([['O',  0.00000, -0.07579, 0.00000],
                  ['H',  0.86681,  0.60144, 0.00000],
                  ['H', -0.86681,  0.60144, 0.00000]], 
                 unit='angstrom',
                 basis = 'cc-pvdz')
    #A = molecule([['H',  0.,  0., 0.00000],
    #              ['H',  0.,  0., 1.40000]], 
    #             unit='au',
    #             basis = 'sto-3g')
    #print('n_electron=',A.nele)
    #print('n_orbital=',A.norb)
    #print('-- Overlap --')
    #print(A.overlap_matrix)
    #print('-- Kinetic --')
    #print(A.kinetic_matrix)
    #print('-- Nuclear Attraction --')
    #print(A.nuclear_attraction_matrix)
    A_rhf = RHF(A)
    print('Hartree-Fock ground state energy =', A_rhf.energy)
