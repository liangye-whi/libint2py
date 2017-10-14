'''
Closed-shell Restricte Hartree-Fock Method.

By Ye @ Oct 2017
'''
from pylibint import molecule
import numpy as np
from scipy.linalg import eigh as eig

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
print('n_electron=',A.nele)
print('n_orbital=',A.norb)
print('-- Overlap --')
print(A.overlap_matrix)
print('-- Kinetic --')
print(A.kinetic_matrix)
print('-- Nuclear Attraction --')
print(A.nuclear_attraction_matrix)

print('-- Core Hamiltonian --')
H_core = A.kinetic_matrix + A.nuclear_attraction_matrix
print(H_core)

G = np.zeros((A.norb,A.norb,A.norb,A.norb))
m = 0
for i in range(A.norb):
    for j in range(i+1):
        for k in range(i):
            for l in range(k+1):
                G[i,j,k,l] = A.electron_repulsion_list[m]
                G[k,l,i,j] = A.electron_repulsion_list[m]
                G[i,j,l,k] = A.electron_repulsion_list[m]
                G[l,k,i,j] = A.electron_repulsion_list[m]
                G[j,i,k,l] = A.electron_repulsion_list[m]
                G[k,l,j,i] = A.electron_repulsion_list[m]
                G[j,i,l,k] = A.electron_repulsion_list[m]
                G[l,k,j,i] = A.electron_repulsion_list[m]
                m += 1
        for l in range(j+1):
            G[i,j,i,l] = A.electron_repulsion_list[m]
            G[i,l,i,j] = A.electron_repulsion_list[m]
            G[i,j,l,i] = A.electron_repulsion_list[m]
            G[l,i,i,j] = A.electron_repulsion_list[m]
            G[j,i,i,l] = A.electron_repulsion_list[m]
            G[i,l,j,i] = A.electron_repulsion_list[m]
            G[j,i,l,i] = A.electron_repulsion_list[m]
            G[l,i,j,i] = A.electron_repulsion_list[m]
            m += 1
#for i in range(A.norb):
#    for j in range(A.norb):
#        for k in range(A.norb):
#            for l in range(A.norb):
#                print((i,j,k,l),'=',G[i,j,k,l])
print('-- Initial Density Matrix --')
P = np.zeros((A.norb, A.norb))
np.fill_diagonal(P[:A.nele//2,:A.nele//2], 2)
print(P)
E_prev = 0.
epsilon = 1e-10
max_iter = 10000
for i in range(max_iter):

    B = np.einsum('kl,ijlk->ij', P, G)-.5*np.einsum('kl,ilkj->ij',P,G)
    F = H_core + B
    #print('---- Fock ----')
    #print(F)
    E0 = .5 * np.einsum('ij,ji->', P, H_core + F)
    print('Iter =', i, '\tE_elec =', E0)

    E, C = eig(F, A.overlap_matrix)

    P = 2. * C[:,:A.nele//2].dot(C[:,:A.nele//2].T)

    if abs(E0 - E_prev) < epsilon:
        break
     
    E_prev = E0
print('Hartree-Fock ground state energy =', E0 + A.nuclear_repulsion)
