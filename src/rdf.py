import numpy as np
from ase.geometry import analysis

def rdf_instant(atom, bins, rmax):
    """
    Compute the radial distribution function (RDF) for an atomic system.

    Parameters:
    - atom: ase.Atoms object
    - bins: int
      Number of bins for dividing the radial distance range.
    - rmax: float
      Maximum radial distance to consider.

    Returns:
    - dist_count: numpy.ndarray
      Array of radial distances corresponding to the center of each bin, here (r)
    - pdf: numpy.ndarray
      The computed RDF values for each bin.
    """
    # Generate an array of distances corresponding to the bin centers
    dist_count = np.arange(rmax / (2 * bins), rmax + rmax / (2 * bins), rmax / bins)
    
    # Perform analysis and calculate the RDF
    ana = analysis.Analysis(atom)
    pdf = ana.get_rdf(rmax, bins)

    return dist_count, pdf
