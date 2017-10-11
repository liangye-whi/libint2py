from ctypes import CDLL, py_object, c_char_p, c_int
import numpy as np

class molecule:
    """molecule definition {coordinates, basis and integral matrices}. """
    def _parse_atomic_number(self, atoms):
        periodic_table = ['H','He','Li','Be','B','C','N','O','F','Ne','Na','Mg','Al','Si','P',
            'S','Cl','Ar','K','Ca','Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn','Ga','Ge','As',
            'Se','Br','Kr','Rb','Sr','Y','Zr','Nb','Mo','Tc','Ru','Rb','Pd','Ag','Cd','In','Sn',
            'Sb','Te','I','Xe','Cs','Ba']
        atomic_number = dict(zip(periodic_table,range(1,len(periodic_table)+1)))
        for atom in atoms:
            assert len(atom) == 4
            if type(atom[0]) is str:
                assert atom[0] in periodic_table
                atom[0] = atomic_number[atom[0]]
        return atoms

    def __init__(self, atoms, basis):
        '''atoms: List of atomic symbols/numbers and coordinates.
        example:
        >>> [['H', 0.0, 0.0, 1.4], ['H', 0.0, 0.0, 0.0]] # for H2 with bond length 1.4 angstrom.

        basis: String of basis set name.
        '''
        self.atoms = self._parse_atomic_number(atoms)
        self.basis = basis

        lib = CDLL('./libint2py.so')
        lib.integrals.restype = py_object
        lib.integrals.argtypes = [py_object, c_char_p]

        basis_b = basis.encode('UTF-8')
        integral_matrices = lib.integrals(self.atoms, basis_b)

        self.overlap_matrix = np.array(integral_matrices[0])
        n = int(np.sqrt(self.overlap_matrix.shape[0]))
        self.overlap_matrix = self.overlap_matrix.reshape(n,n)

        self.kinetic_matrix = np.array(integral_matrices[1])
        n = int(np.sqrt(self.kinetic_matrix.shape[0]))
        self.kinetic_matrix = self.kinetic_matrix.reshape(n,n)

        self.nuclear_attraction_matrix = np.array(integral_matrices[2])
        n = int(np.sqrt(self.nuclear_attraction_matrix.shape[0]))
        self.nuclear_attraction_matrix = self.nuclear_attraction_matrix.reshape(n,n)



if __name__ == "__main__":
    np.set_printoptions(linewidth=np.nan,threshold=np.nan)

    A = molecule([['O',  0.00000, -0.07579, 0.00000],
                  ['H',  0.86681,  0.60144, 0.00000],
                  ['H', -0.86681,  0.60144, 0.00000]],
                 'sto-3g')
    print('-- Overlap --')
    print(A.overlap_matrix)
    print('-- Kinetic --')
    print(A.kinetic_matrix)
    print('-- Nuclear Attraction --')
    print(A.nuclear_attraction_matrix)
    print('-- Core Hamiltonian --')
    print(A.kinetic_matrix + A.nuclear_attraction_matrix)
