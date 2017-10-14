from ctypes import CDLL, py_object, c_char_p, c_int
import numpy as np

class molecule:
    """molecule definition {coordinates, basis and integral matrices}. """
    def __init__(self, atoms, unit='au', basis='sto-3g'):
        '''atoms: List of atomic symbols/numbers and coordinates.
        example:
        >>> [['H', 0.0, 0.0, 1.4], ['H', 0.0, 0.0, 0.0]] # for H2 with bond length 1.4 angstrom.

        basis: String of basis set name.
        '''
        self.atoms = self._parse_atomic_number(atoms, unit)
        self.basis = basis
        self.nele = sum(atom[0] for atom in self.atoms)

        lib = CDLL('./libint2py.so')
        lib.integrals.restype = py_object
        lib.integrals.argtypes = [py_object, c_char_p]

        basis_b = basis.encode('UTF-8')
        integral_matrices = lib.integrals(self.atoms, basis_b)

        self.overlap_matrix = np.array(integral_matrices[0])
        self.norb = int(np.sqrt(self.overlap_matrix.shape[0]))
        self.overlap_matrix = self.overlap_matrix.reshape(self.norb, self.norb)

        self.kinetic_matrix = np.array(integral_matrices[1])
        self.kinetic_matrix = self.kinetic_matrix.reshape(self.norb,self.norb)

        self.nuclear_attraction_matrix = np.array(integral_matrices[2])
        self.nuclear_attraction_matrix = self.nuclear_attraction_matrix.reshape(self.norb,self.norb)
        
        self.electron_repulsion_list = integral_matrices[3]
        self.neri = len(self.electron_repulsion_list)
        ##+ Decorate the elecronic repulsion integrals.
        #eri_index = []
        #for i in range(self.norb):
        #    for j in range(i+1):
        #        for k in range(i):
        #            for l in range(k+1):
        #                eri_index.append((i,j,k,l))
        #        for l in range(j+1):
        #            eri_index.append((i,j,i,l))
        #self.electron_repulsion_dict = dict(list(zip(eri_index, electron_repulsion_list)))
        ##- Decorate the elecronic repulsion integrals.
        nuclear_repulsion = 0.
        for i, atom_a in enumerate(atoms):
            for atom_b in atoms[i+1:]:
                r = np.array(atom_a[1:]) - np.array(atom_b[1:])
                dist = np.linalg.norm(r)
                nuclear_repulsion += atom_a[0]*atom_b[0] / dist
        self.nuclear_repulsion=nuclear_repulsion

    def _parse_atomic_number(self, atoms, unit):
        periodic_table = ['H','He','Li','Be','B','C','N','O','F','Ne','Na','Mg','Al','Si','P',
            'S','Cl','Ar','K','Ca','Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn','Ga','Ge','As',
            'Se','Br','Kr','Rb','Sr','Y','Zr','Nb','Mo','Tc','Ru','Rb','Pd','Ag','Cd','In','Sn',
            'Sb','Te','I','Xe','Cs','Ba']
        atomic_number = dict(zip(periodic_table,range(1,len(periodic_table)+1)))

        length_convert_factor = 1.
        if unit == 'au' or unit == 'Bohr' or unit == 'bohr' or unit == 'a.u.':
            pass
        elif unit == 'angstrom' or unit == 'Angstrom':
            length_convert_factor = 1. / 0.52917721092
        else:
            print('Cannot recognize unit of length. Assume it is a.u.')

        for atom in atoms:
            assert len(atom) == 4
            if type(atom[0]) is str:
                assert atom[0] in periodic_table
                atom[0] = atomic_number[atom[0]]

            for i in range(1,4):
                atom[i] = atom[i] * length_convert_factor

        return atoms

if __name__ == "__main__":
    np.set_printoptions(linewidth=np.nan,threshold=np.nan)

    #A = molecule([['O',  0.00000, -0.07579, 0.00000],
    #              ['H',  0.86681,  0.60144, 0.00000],
    #              ['H', -0.86681,  0.60144, 0.00000]], 
    #             unit='angstrom',
    #             basis = 'sto-3g')
    A = molecule([['H',  0.,  0., 0.00000],
                  ['H',  0.,  0., 1.40000]], 
                 unit='au',
                 basis = 'sto-3g')
    print('-- Overlap --')
    print(A.overlap_matrix)
    print('-- Kinetic --')
    print(A.kinetic_matrix)
    print('-- Nuclear Attraction --')
    print(A.nuclear_attraction_matrix)
    print('-- Core Hamiltonian --')
    print(A.kinetic_matrix + A.nuclear_attraction_matrix)
