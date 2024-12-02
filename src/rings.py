from ase.io import read
import numpy as np
from matscipy import rings

def rings_distribution(atoms, cutoff=None, maxlength=10):
    """
    Compute the distribution of ring sizes in atomic structures over multiple frames.

    Parameters:
    - atoms: list or ase.Atoms object
      A single `ase.Atoms` object or a list of such objects representing atomic structures.
    - cutoff: float, optional
      The cutoff distance for neighbor identification (default: None, uses default behavior of `rings.ring_statistics`).
    - maxlength: int, optional
      The maximum ring size to consider in the analysis (default: 10).
    - ddof: int, optional
      Degrees of freedom for statistical adjustments, not used in this implementation (default: 1).

    Returns:
    - dict: A dictionary with the following keys:
      - 'rings_length': list of int, ring lengths from 0 to `maxlength`.
      - 'rings_dist': list of float, average ring count for each length over all frames.
      - 'rings_dist_frac': list of float, average fractional ring count (as percentage) for each length over all frames.
    """
    # Ensure `atoms` is a list, even if a single Atoms object is passed
    if not isinstance(atoms, list):
        atoms = [atoms]
    
    nframe = len(atoms)  # Number of frames to process
    rings_dist = np.zeros(maxlength + 1)  # Store counts of rings for each length
    rings_dist_frac = np.zeros(maxlength + 1)  # Store fractional counts as percentages
    rings_all = []  # Store full ring count for each frame
    rings_frac_all = []  # Store fractional ring count for each frame

    # Loop through all frames
    for i in range(nframe):
        # Compute ring statistics for the current frame
        rings_tmp = rings.ring_statistics(atoms[i], cutoff, maxlength=maxlength + 3)
        
        # Pad the array to ensure it matches the desired maximum length
        dist_tmp = np.pad(rings_tmp, (0, maxlength + 5 - len(rings_tmp)))
        
        # Extract ring counts up to `maxlength`
        dist = dist_tmp[:maxlength + 1]
        dist_frac = dist / np.sum(dist) * 100  # Convert to percentage

        # Accumulate ring distributions across frames
        rings_dist += dist
        rings_dist_frac += dist_frac

        # Store ring distributions for each frame
        rings_all.append(list(dist))
        rings_frac_all.append(list(dist_frac))

    # Compute averages over all frames
    rings_dist /= nframe
    rings_dist_frac /= nframe

    # Return results in a dictionary
    rings_length = np.arange(0, maxlength + 1)
    result = {
        'rings_length': rings_length.tolist(),
        'rings_dist': rings_dist.tolist(),
        'rings_dist_frac': rings_dist_frac.tolist()
    }

    return result

