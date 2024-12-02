import pandas as pd
import numpy as np
from ase.build import bulk
from ase.io import read
from quippy.descriptors import Descriptor
import copy
from ase.neighborlist import *
from ovito.pipeline import StaticSource, Pipeline
from ovito.io.ase import ase_to_ovito
from ovito.io import *
from ovito.modifiers import *
from ovito.data import *
from ovito.pipeline import *
from ase import Atoms
import math

def convert_xyz_pickle(data_files, energy_key = "energy", force_key = "forces"):
    # Read in the data
    dataset = {
        "energy": [],
        "forces": [],
        "ase_atoms": [],
    }

    structures = read(data_files, index=":")

    for atoms in structures:
        dataset["energy"].append(atoms.info[energy_key])
        dataset["forces"].append(atoms.arrays[force_key])
        dataset["ase_atoms"].append(atoms)

    df = pd.DataFrame(dataset)
    df.to_pickle(f"{data_files}.pkl.gzip", compression="gzip")


def cycle_colors(n):
    colors = ['#332288', '#117733', '#44AA99', 
               '#88CCEE', '#DDCC77', '#CC6677',
               '#AA4499', '#882255', '#981558', '#D83B3B']
    return colors[n % len(colors)]

def cycle_markers(n):
    markers = ['1', '^', 'v', '<', '>', 'o', 's', 'p', '*', 'h', 'H', 'D', 'd',]
    return markers[n % len(markers)]

def get_max_F(forces):
    return max([max(f) for f in forces])

def get_len(atom):
    return len(atom)
    
def get_label(atom):
    lab=str(atom.info['label'])
    return lab

def get_vol(atom):
    vol=atom.get_volume()/len(atom)
    return vol


def build_df_unary(df_unary):
    # calculate volume per atom
    df_unary["vol_per_atom"]=df_unary["ase_atoms"].apply(get_vol)

    # obtain max force component
    df_unary["F_max"] = df_unary["forces"].apply(get_max_F)
    df_unary["nb_atoms"]=df_unary["ase_atoms"].apply(get_len)
    
    # obtain label
    df_unary["label"] = df_unary["ase_atoms"].apply(get_label)
    return df_unary


def ptm(atoms):
    data = ase_to_ovito(atoms)
    pipeline = Pipeline(source = StaticSource(data = data))
    pipeline.modifiers.append(CreateBondsModifier(cutoff = 2.85))

    types = pipeline.source.data.particles_.particle_types_
    types.type_by_id_(1).load_defaults()

    # Polyhedral template matching:
    mod = PolyhedralTemplateMatchingModifier()
    mod.structures[PolyhedralTemplateMatchingModifier.Type.FCC].enabled = False
    mod.structures[PolyhedralTemplateMatchingModifier.Type.HCP].enabled = False
    mod.structures[PolyhedralTemplateMatchingModifier.Type.BCC].enabled = False
    mod.structures[PolyhedralTemplateMatchingModifier.Type.CUBIC_DIAMOND].enabled = True
    mod.structures[PolyhedralTemplateMatchingModifier.Type.HEX_DIAMOND].enabled = True
    pipeline.modifiers.append(mod)

    data = pipeline.compute()

    return data,pipeline

def config_sort_v2(atoms):
    ''' Simplistic sorting, only considers PTM analysis
    Separates data into structure classes based on the percentage of diamond-like environments identified by PTM, as implemented by OVITO
    Parameters:
    atoms: ase.Atoms object
     
    Returns:
    config_type: string that represents the structure type
    '''

    data,pipeline = ptm(atoms)

    # Structure type of particles (similarity to diamond), where type='0' is unidentified (amorphous) and 6 + 7 are diamond structures
    type_count=dict(zip(*np.unique(data.particles["Structure Type"][...], return_counts=True)))
    
    if type_count.get(int(0)) is not None and type_count[0]>len(atoms)*0.995:
        config_type='CRN'

    elif type_count.get(int(0)) is not None and type_count[0]>len(atoms)*0.8:
        config_type='Paracrystalline'

    else:
        if type_count.get(int(0)) is None :
            config_type='Diamond'

        else:
            config_type='Polycrystalline'

    return config_type

def ptm_cat(row):
    atoms = row['ase_atoms']
    data,pipeline=ptm(atoms)

    if data.tables['structures']['Count'][0] == len(atoms):
        return 'CRN'
    elif (data.tables['structures']['Count'][6]+ data.tables['structures']['Count'][7])/len(atoms)<1:
        return 'Crystalline'
    else:
        return 'Diamond'
    

def marker_size(nb_atoms):
    dict={64:1, 216:3, 512:8, 1000:15}
    return dict[nb_atoms]

def structure_soap_cSi(atoms, zeta = 4):
    """
    Computes the structure SOAP similarity to cubic diamond Si.

    Parameters:
    atoms: ase.Atoms object

    Returns:
    k_dia_av: single value, SOAP similarity of the structure to cubic diamond Si
    """
    soap_descriptor_str = ('soap l_max=3 n_max=6 ' # basis function size
                       'atom_sigma=0.5 cutoff=5.0 ' # smoothness and cutoff (Angstrom) 
                       'average=T') # average descriptor over whole cell, or one descriptor for each atom
    soap_cell_average = Descriptor(soap_descriptor_str)
    
    diamond = bulk('Si', 'diamond', a=5.43)
    diamond_descriptor = soap_cell_average.calc_descriptor(diamond).squeeze()

    average_descriptor = soap_cell_average.calc_descriptor(atoms)
    average_descriptor = np.array(average_descriptor).squeeze() 
    k_dia_av = np.dot(average_descriptor, diamond_descriptor)**zeta
    k_dia_av=k_dia_av.reshape(-1, 1)

    return k_dia_av

def atomistic_soap_cSi(atoms, zeta = 4):
    """
    Computes the atomistic SOAP similarity to cubic diamond Si.

    Parameters:
    atoms: ase.Atoms object

    Returns:
    k_dia_atomistic: array of the atomistic SOAP similarities to cubic diamond Si
    """
    soap_descriptor_str = ('soap l_max=3 n_max=6 ' # basis function size
                       'atom_sigma=0.5 cutoff=5.0 ' # smoothness and cutoff (Angstrom) 
                       'average=T') # average descriptor over whole cell, or one descriptor for each atom
    soap_cell_average = Descriptor(soap_descriptor_str)

    diamond = bulk('Si', 'diamond', a=5.43)
    diamond_descriptor = soap_cell_average.calc_descriptor(diamond).squeeze()

    soap_atomistic = Descriptor(soap_descriptor_str.replace('average=T', 'average=F'))
    atomistic_descriptor = soap_atomistic.calc_descriptor(atoms)
    atomistic_descriptor = np.array(atomistic_descriptor).squeeze()
    k_dia_atomistic = np.dot(atomistic_descriptor, diamond_descriptor)**zeta
    k_dia_atomistic=k_dia_atomistic.reshape(-1, 1)
    return k_dia_atomistic

def CNA(atoms):
    """
    Computes the Common neighbor analysis using the OVITO modifier.

    Parameters:
    atoms: ase.Atoms object

    Returns:
    CNA: array of the predicted structure type, where:
    0: other
    1: cubic diamond
    2: cubic diamond 1st neighbor
    3: cubic diamond 2nd neighbor
    4: hexagonal diamond
    5: hexagonal diamond 1st neighbor
    6: hexagonal diamond 2nd neighbor
    """

    data = ase_to_ovito(atoms)
    pipeline = Pipeline(source = StaticSource(data = data))
    mod = IdentifyDiamondModifier()
    pipeline.modifiers.append(mod)
    data=pipeline.compute(0)
    cna=np.asarray(data.particles["Structure Type"][...])    
    return cna


def NN_E_mtp(atoms):
    """
    Computes the local atomic energy averaged over nearest neighbors.

    Parameters:
    atoms: ase.Atoms object

    Returns:
    E: array of the local atomic energies averaged over nearest neighbors within a spherical cutoff, relative to c-Si
    """
    # build neighborlist
    i,j=neighbor_list('ij', atoms, 2.85)
    neigh=np.stack((i.T,j.T), axis=1) #array of bonds
    coord = np.bincount(i)

    # iterate over neighbors and compute averaged energy
    E=np.zeros(len(atoms))
    for ind in range(len(atoms)):
        for bond in neigh:
            if ind==bond[0]:
                E[ind]+=atoms.arrays['c_PEa'][bond[1]]
        E[ind]+=atoms.arrays['c_PEa'][ind]
        E[ind]/=(coord[ind]+1)
    return E +(163161.251478124/1000)


def cluster_analysis(row):
    """
    Perform cluster analysis on atomic structures to identify clusters and compute statistics.

    Parameters:
    -----------
    row : dataframe row

    Returns:
    --------
    list
        A list containing:
        - sizes (numpy.ndarray): Array of cluster sizes.
        - av_size (float): Average size of clusters.
        - nb_clusters (int): Total number of clusters identified.
        - av_dist (float): Average distance between the nearest atoms in different clusters.
        - all_dist (numpy.ndarray): Array of all nearest distances between atoms in different clusters.
        - av_dist_cm (float): Average distance between centers of mass of clusters.
        - all_dist_cm (numpy.ndarray): Array of distances between the centers of mass of clusters.
    
    Notes:
    ------
    This function:
    1. Identifies clusters in the atomic structure using bonding information.
    2. Computes cluster statistics, including sizes, count, and inter-cluster distances.
    3. Uses Ovito's `ClusterAnalysisModifier` for cluster identification and related metrics.
    """
    atoms = row['ase_atoms']
    data, pipeline = ptm(atoms) 

    # Select type:
    pipeline.modifiers.append(SelectTypeModifier(
        property='Structure Type', 
        types={6, 7}))
    
    # Cluster analysis:
    pipeline.modifiers.append(ClusterAnalysisModifier(
        neighbor_mode=ClusterAnalysisModifier.NeighborMode.Bonding, 
        only_selected=True,
        compute_com=True,  
        sort_by_size=True))
    
    data = pipeline.compute()

    cluster_table = data.tables['clusters']
    sizes = np.asarray(cluster_table['Cluster Size'][...])
    av_size = np.mean(sizes)
    nb_clusters = len(sizes)

    # Distance between centers of mass (CMS)
    if len(sizes) > 1:
        cluster_property = np.asarray(data.particles['Cluster'][...])
        dists = []
        for i in np.arange(1, len(sizes)):
            inds_i = np.where(cluster_property == i)
            
            for j in range(i + 1, len(sizes) + 1):
                inds_j = np.where(cluster_property == j)
                if type(inds_j) == tuple:
                    inds_j = inds_j[0]
                min_dist = math.sqrt(2) * atoms.cell[0][0]

                for ind_i in inds_i:
                    for ind_j in inds_j:
                        dist = atoms.get_distance(ind_i, ind_j, mic=True)
                        min_dist = min(min_dist, dist)
                dists.append(min_dist)
        
        av_dist = np.mean(dists)
        all_dist = np.asarray(dists).flatten()
    else:
        all_dist = np.zeros((1))
        av_dist = 0

    c_ms = np.asarray(cluster_table['Center of Mass'][...])
    
    if c_ms.size > 3:
        temp = Atoms(np.repeat('Si', len(c_ms)), positions=c_ms, cell=atoms.cell)
        temp.set_pbc([True, True, True])
        dists = temp.get_all_distances(mic=True)
        all_dist_cm = np.delete(np.unique(dists), 0).flatten()
        av_dist_cm = np.sum(all_dist_cm / len(c_ms))
    else:
        all_dist_cm = np.zeros((1))
        av_dist_cm = 0

    return [sizes, av_size, nb_clusters, av_dist, all_dist, av_dist_cm, all_dist_cm]
