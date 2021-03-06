# dionysus tools
from dionysus import *
from dionysus import Simplex
from dionysus import Filtration
from dionysus import StaticPersistence
from dionysus import vertex_cmp
from dionysus import data_cmp
from dionysus import data_dim_cmp
from dionysus import Rips
from dionysus import PairwiseDistances
from dionysus import points_file
from dionysus import ExplicitDistances

# other packages
import tdawavelets.persistence_diagram as dg
import numpy as np


def levelset_filtration_2d(f, sublevel = True):
    """Prepare the level-set filtration for Dionysus


    """
    # prevents overwriting
    temp_f = f.copy()

    if not sublevel:
        temp_f *= -1

    # store filtration points as a list
    filtration = []

    # used for cycling through function and vertices
    x, y = f.shape

    # index values for vertizes
    vertices = np.arange(x * y).reshape(x,y)

    for i in range(x - 1):
        for j in range(y - 1):
            p0 = vertices[i, j]
            p1 = vertices[i + 1, j]
            p2 = vertices[i, j + 1]
            p3 = vertices[i + 1, j + 1]

            # add vertices
            filtration.append([
                                (p0, ),
                                temp_f[i, j]
                             ])

            # add 1-simplices
            filtration.append([
                                (p0, p1),
                                max(temp_f[i,j], temp_f[i+1, j])
                             ])

            filtration.append([
                                (p0, p2),
                                max(temp_f[i,j], temp_f[i, j+1])
                             ])

            filtration.append([
                                (p1, p2),
                                max(temp_f[i+1,j], temp_f[i, j+1])
                             ])


            # add 2-simplices
            filtration.append([
                                (p0, p1, p2),
                                max(temp_f[i,j], temp_f[i+1, j], temp_f[i, j+1])
                             ])

            filtration.append([
                                (p1, p2, p3),
                                max(temp_f[i + 1, j], temp_f[i, j + 1], temp_f[i + 1, j+1])
                             ])

            # add boundary simplices
            if i == x - 2:
                filtration.append([
                                    (p1, ),
                                    temp_f[i + 1, j]
                                 ])
                filtration.append([
                                    (p1, p3),
                                    max(temp_f[i + 1, j], temp_f[i + 1, j+1])
                                 ])

            if j == y - 2:
                filtration.append([
                                    (p2, ),
                                    temp_f[i, j + 1]
                                 ])
                filtration.append([
                                    (p2, p3),
                                    max(temp_f[i, j + 1], temp_f[i + 1, j+1])
                                 ])

            if (i == x - 2) and (j == y - 2):
                filtration.append([
                                    (p3, ),
                                    temp_f[i + 1, j + 1]
                                 ])

    return(filtration)

def _persistence_diagram_grid(smap, p, max_death = None):
    """Constructs a persistence diagram object from persistence computed in Dionysus.

    Arguments:

        1. smap: Simplex map from a persistence object (see Dionysus)
        2. p: persistence object (see Dionysus)

    Returns:

        1. Numpy array containing the persistence diagram.
            * Column 1: Homology group
            * Column 2: Birth time
            * Column 3: Death time
    """

    # list to store persistence diagram points
    new_pd = []

    # add dimension, birth, death points to persistence diagram list
    for i in p:
        if i.sign():
            b = smap[i]

            if i.unpaired():
                if max_death is not None:
                    new_pd.append([b.dimension(), b.data, max_death])
                else:
                    new_pd.append([b.dimension(), b.data, np.inf])

                continue

            d = smap[i.pair()]

            if b.data != d.data:
                if max_death is not None:
                    new_pd.append([b.dimension(), b.data, min(d.data, max_death)])
                else:
                    new_pd.append([b.dimension(), b.data, d.data])

                continue

    # sort by homology group and
    # return as a numpy array
    # new_pd.sort()
    # return(np.vstack(new_pd).astype(np.float32))

    new_pd = np.vstack(new_pd).astype(np.float32)

    # remove features with zero persistence
    new_pd = np.delete(new_pd, np.where(new_pd[:,2] - new_pd[:,1] == 0)[0], axis = 0)

    # sort by persistence within each homology group
    for hom in np.unique(new_pd[:,0]):
        # get the ids of the groups associated with this homology group
        hom_idx = np.where(new_pd[:,0] == hom)[0]
        temp_pd = new_pd[hom_idx, 1:]

        # get the sorted indices and sort the array
        sorted_idx = np.argsort(temp_pd[:, 1] - temp_pd[:, 0])[::-1]
        new_pd[hom_idx, 1:] = temp_pd[sorted_idx, ]

    return(new_pd)


def _persistence_diagram_rips(smap, p, max_hom, max_death = None):
    """Constructs a persistence diagram object from persistence computed in Dionysus.

    Arguments:

        1. smap: Simplex map from a persistence object (see Dionysus)
        2. p: persistence object (see Dionysus)

    Returns:

        1. Numpy array containing the persistence diagram.
            * Column 1: Homology group
            * Column 2: Birth time
            * Column 3: Death time
    """

    # list to store persistence diagram points
    new_pd = []

    # add [dimension, birth, death] points to persistence diagram list
    for i in p:
        if i.sign():
            b = smap[i]

            if b.dimension() > max_hom:
                continue

            if i.unpaired():
                if max_death is not None:
                    new_pd.append([b.dimension(), b.data, max_death])
                else:
                    new_pd.append([b.dimension(), b.data, np.inf])

                continue

            d = smap[i.pair()]
            new_pd.append([b.dimension(), b.data, d.data])

    # sort by homology group and convert to numpy array
    new_pd.sort()
    new_pd = np.vstack(new_pd).astype(np.float32)

    # remove features with zero persistence
    new_pd = np.delete(new_pd, np.where(new_pd[:,2] - new_pd[:,1] == 0)[0], axis = 0)

    # sort by persistence within each homology group
    for hom in np.unique(new_pd[:,0]):
        # get the ids of the groups associated with this homology group
        hom_idx = np.where(new_pd[:,0] == hom)[0]
        temp_pd = new_pd[hom_idx, 1:]

        # get the sorted indices and sort the array
        sorted_idx = np.argsort(temp_pd[:, 1] - temp_pd[:, 0])[::-1]
        new_pd[hom_idx, 1:] = temp_pd[sorted_idx, ]

    return(new_pd)

def compute_grid_diagram(f, sublevel = True, max_death = None):
    """Workflow to construct a Persistence Diagram object from the level sets
    of the given function.

    Arguments
        1.

    Returns
        1.

    Raises
        1.

    """

    # assume kernel methods are used here
    if sublevel == False and max_death is None:
        max_death = 0

    # construct list of times the simplicies are
    # added to the simplicial complex
    filtration = levelset_filtration_2d(f = f,
                                        sublevel = sublevel)

    # construct a simplex list for Dionysus
    scomplex = [Simplex(a, b) for (a,b) in filtration]

    # construct Dionysus filtration object
    filt = Filtration(scomplex, data_cmp)

    # compute persistent homology
    p = StaticPersistence(filt)
    p.pair_simplices(True)
    smap = p.make_simplex_map(filt)

    # generate numpy persistence diagram
    pd = _persistence_diagram_grid(smap, p, max_death)

    if not sublevel:
        pd[:, 1:] *= -1
        pd = pd[:, (0, 2, 1)]

    return(dg.PersistenceDiagram(PD = pd))

def compute_rips_diagram(points, max_hom, max_death):
    """Workflow to construct a Persistence Diagram object from the level sets
    of the given function.

    Arguments
        1.

    Returns
        1.

    Raises
        None

    """

    # get pairwise distances
    distances = PairwiseDistances(points)

    rips = Rips(distances)
    simplices = Filtration()
    rips.generate(int(max_hom + 1), float(max_death), simplices.append)

    # step to speed up computation
    for s in simplices:
        s.data = rips.eval(s)

    # compute persistence
    simplices.sort(data_dim_cmp)
    p = StaticPersistence(simplices)
    p.pair_simplices()

    # construct persistence diagram
    smap = p.make_simplex_map(simplices)
    pd = _persistence_diagram_rips(smap, p, max_hom, max_death)

    return(dg.PersistenceDiagram(PD = pd))




if __name__ == "__main__":
    """For testing and illustrative purposes only"""
