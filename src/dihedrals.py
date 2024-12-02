import os
import numpy as np
from ase import Atoms
from ase.io import read
from ase.geometry import *
from ase.neighborlist import neighbor_list

def make_neigh(atoms, cutoff=2.85):
    """
    Create a list of neighboring atoms within a specified cutoff.

    Parameters:
    - atoms: ase.Atoms object
    - cutoff: float, radial neighborhood cutoff

    Returns:
    - bonded_atoms: list of arrays, each containing indices of bonded neighbors for each atom
    """
    # Generate pairs of neighbors within the cutoff
    i, j = neighbor_list('ij', atoms, cutoff)
    neigh = np.stack((i.T, j.T), axis=1)

    # Initialize a list of empty lists for each atom
    bonded_atoms = [[] for _ in range(len(atoms))]

    # Populate the list with neighboring atoms
    for bond in neigh:
        a, b = bond
        bonded_atoms[a].append(b)
        bonded_atoms[b].append(a)
    
    # Ensure uniqueness of neighbors for each atom
    bonded_atoms = [np.unique(bonds) for bonds in bonded_atoms]

    return bonded_atoms

def comp_dihedrals(atoms, cutoff=2.85):
    """
    Compute dihedral angles for all unique atomic quartets within a specified cutoff.

    Parameters:
    - atoms: ase.Atoms object
    - cutoff: float, radial neighborhood cutoff

    Returns:
    - dihedrals: list of floats, dihedral angles in degrees
    """
    # Generate a list of bonded neighbors
    bonded_atoms = make_neigh(atoms, cutoff)

    dihedrals = []
    paths = []

    # Iterate through all atoms
    for id2 in range(len(atoms)):
        # Loop through neighbors of atom id2
        for id3 in bonded_atoms[id2]:
            # Exclude id3 from neighbors of id2
            n2_ids = [item for item in bonded_atoms[id2] if item != id3]

            # Loop through remaining neighbors of id2
            for id1 in n2_ids:
                # Exclude id2 from neighbors of id3
                n3_ids = [item for item in bonded_atoms[id3] if item != id2]

                # Loop through neighbors of id3
                for id4 in n3_ids:
                    # Compute dihedral angle and define the path
                    dih = atoms.get_dihedral(id1, id2, id3, id4, mic=True)
                    path = [id1, id2, id3, id4]
                    
                    # Avoid duplicate paths (check reversed path)
                    if path[::-1] not in paths:
                        paths.append(path)
                        dihedrals.append(dih)

    return dihedrals
