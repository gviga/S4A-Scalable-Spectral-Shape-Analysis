'''
This code serves as a library to compute functions to evaluate the accuracy of a permutation or the accuracy of a reconstruction without a given ground truth.

Implemented functions:
- Chamfer distance
- Hausdorf distance
- Bijectivity
- Coverage
- Continuity
- TODO: Dirichlet
'''

import numpy as np
from sklearn.neighbors import NearestNeighbors


def chamfer_distance(x, y, metric='l2', direction='bi'):
    """Chamfer distance between two point clouds
    Parameters
    ----------
    x: numpy array [n_points_x, 3]
        first point cloud  (IN OUR SETTING the permutation of the target)
    y: numpy array [n_points_y, 3]
        second point cloud
    metric: string or callable, default ‘l2’
        metric to use for distance computation. Any metric from scikit-learn or scipy.spatial.distance can be used.
    direction: str
        direction of Chamfer distance.
            'y_to_x':  computes average minimal distance from every point in y to x
            'x_to_y':  computes average minimal distance from every point in x to y
            'bi': compute both
    Returns
    -------
    chamfer_dist: float
        computed bidirectional Chamfer distance:
            sum_{x_i \in x}{\min_{y_j \in y}{||x_i-y_j||**2}} + sum_{y_j \in y}{\min_{x_i \in x}{||x_i-y_j||**2}}
    """
    
    if direction=='y_to_x':
        x_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(x)
        min_y_to_x = x_nn.kneighbors(y)[0]
        chamfer_dist = np.mean(min_y_to_x)
    elif direction=='x_to_y':
        y_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(y)
        min_x_to_y = y_nn.kneighbors(x)[0]
        chamfer_dist = np.mean(min_x_to_y)
    elif direction=='bi':
        x_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(x)
        min_y_to_x = x_nn.kneighbors(y)[0]
        y_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(y)
        min_x_to_y = y_nn.kneighbors(x)[0]
        chamfer_dist = np.mean(min_y_to_x) + np.mean(min_x_to_y)
    else:
        raise ValueError("Invalid direction type. Supported types: \'y_x\', \'x_y\', \'bi\'")
        
    return chamfer_dist

def hausdorff_distance(x, y, metric='l2', direction='bi'):
    """Hausdorff distance between two point clouds
    Parameters
    ----------
    x: numpy array [n_points_x, 3]
        first point cloud (IN OUR SETTING the permutation of the target)
    y: numpy array [n_points_y, 3]
        second point cloud
    metric: string or callable, default ‘l2’
        metric to use for distance computation. Any metric from scikit-learn or scipy.spatial.distance can be used.
    direction: str
        direction of Hausdorff distance.
            'y_to_x':  computes maximal minimal distance from every point in y to x
            'x_to_y':  computes maximal minimal distance from every point in x to y
            'bi': compute both and return the maximum
    Returns
    -------
    hausdorff_dist: float
        computed bidirectional Hausdorff distance:
            max(max_{x_i \in x}{\min_{y_j \in y}{||x_i-y_j||}}, max_{y_j \in y}{\min_{x_i \in x}{||x_i-y_j||}})
    """
    
    if direction == 'y_to_x':
        x_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(x)
        min_y_to_x = x_nn.kneighbors(y)[0]
        hausdorff_dist = np.max(min_y_to_x)
        
    elif direction == 'x_to_y':
        y_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(y)
        min_x_to_y = y_nn.kneighbors(x)[0]
        hausdorff_dist = np.max(min_x_to_y)
        
    elif direction == 'bi':
        x_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(x)
        min_y_to_x = x_nn.kneighbors(y)[0]
        y_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(y)
        min_x_to_y = y_nn.kneighbors(x)[0]
        hausdorff_dist = max(np.max(min_y_to_x), np.max(min_x_to_y))
        
    else:
        raise ValueError("Invalid direction type. Supported types: 'y_to_x', 'x_to_y', 'bi'")
        
    return hausdorff_dist

def mse(x, y):
    """
    Computes the Mean Squred error between two point clouds.
    
    Parameters
    ----------
    x: numpy array [n_points, 3]
        First point cloud.
    y: numpy array [n_points, 3]
        Second point cloud. Must have the same number of points as x.
    
    Returns
    -------
    mse: float
        The mean squared error between the two point clouds.
    """
    
    # Ensure both point clouds have the same shape
    if x.shape != y.shape:
        raise ValueError("The two point clouds must have the same shape.")
        
    # Compute the mean of the squared differences
    mse = np.mean((x - y)**2)
    
    return mse

def bijectivity(x,p21, p12):
    """
    Computes the bijectivity error of the recontruction of a pointcloud.
    
    Parameters
    ----------
    x: numpy array [n_points, 3]
        First point cloud.
    p21: numpy array [n_points,1]
        permutation 21 
    p12: numpy array [n_points,1]
        permutation 12
    Returns
    -------
    bij: float
        The mean squared error between the two point clouds.
    """
    #compute the nearest correspondence using nearest neighbour
    bij=mse(x[p12[p21]],x)

    return bij

#dirichlet loss
def dirichlet_energy(L, f):
    """
    Calculate the Dirichlet energy for a given function on a 3D mesh.

    Parameters:
    L (scipy.sparse.csr_matrix): The Laplacian matrix of the mesh.
    f (np.ndarray): A vector containing the function values at each vertex.

    Returns:
    float: The Dirichlet energy.
    """
    # Ensure L is in sparse format
    if not isinstance(L, csr_matrix):
        L = csr_matrix(L)

    # Compute the Dirichlet energy
    energy = f.T @ L @ f
    
    return energy


#from pyFM continuity and coverage

def continuity(p2p, D1_geod, D2_geod, edges):
    """
    Computes continuity of a vertex to vertex map. The map goes from
    the target shape to the source shape.

    Parameters
    ----------------------
    p2p     :
            (n2,) - vertex to vertex map giving the index of the matched vertex on the source shape
                 for each vertex on the target shape (from a functional map point of view)
    gt_p2p  :
            (n2,) - ground truth mapping between the pairs
    D1_geod :
            (n1,n1) - geodesic distance between pairs of vertices on the source mesh
    D2_geod :
            (n1,n1) - geodesic distance between pairs of vertices on the target mesh
    edges   :
            (n2,2) edges on the target shape

    Returns
    -----------------------
    continuity : float
            average continuity of the vertex to vertex map
    """
    source_len = D2_geod[(edges[:,0], edges[:,1])]
    target_len = D1_geod[(p2p[edges[:,0]], p2p[edges[:,1]])]

    continuity = np.mean(target_len / source_len)

    return continuity

def coverage(p2p, A):
    """
    Computes coverage of a vertex to vertex map. The map goes from
    the target shape to the source shape.

    Parameters
    ----------------------
    p2p :
            (n2,) - vertex to vertex map giving the index of the matched vertex on the source shape
                 for each vertex on the target shape (from a functional map point of view)
    A   :
             (n1,n1) or (n1,) - area matrix on the source shape or array of per-vertex areas.

    Returns
    -----------------------
    coverage : float
            coverage of the vertex to vertex map
    """
    if len(A.shape) == 2:
        vert_area = np.asarray(A.sum(1)).flatten()
    else:
        vert_area=A
    coverage = vert_area[np.unique(p2p)].sum() / vert_area.sum()

    return coverage




