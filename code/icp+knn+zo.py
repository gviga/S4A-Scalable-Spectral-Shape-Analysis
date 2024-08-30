#code to perform ICP+knn


# import libraries

import numpy as np
from pyFM.mesh import TriMesh
import os
import matplotlib.pyplot as plt

from tqdm import tqdm as tqdm
import pyFM.spectral as spectral
from pyFM.FMN import FMN
import pyFM
import sys

import pyFM.spectral as spectral
from pyFM.spectral.nn_utils import knn_query
from pyFM.refine.zoomout import zoomout_refine

import trimesh
from sklearn.neighbors import KDTree, NearestNeighbors
import scipy.spatial
from tqdm.auto import tqdm
from os.path import join

sys.path.append('../')
from utils.visual_utils import *
from utils.icp_utils import *
from evaluation import *


n_sub=2000

#################################LOAD MESHES#############################################

#Trimesh().process Process the LB spectrum and saves it. Additionnaly computes per-face normals, k are the number of eigenvalues to compute. Intrinsic = True skip the normal computation, robust = True, use tufted laplacian, necessary with Non-Manifolds.
n_meshes=2
meshlist=[]
for i in tqdm(range(0,n_meshes)):
    aux=trimesh.load_mesh('../data/giorgio_/00000'+str(i)+'_tumoredbrain/00000'+str(i)+'_tumoredbrain.off')
    print(f"number of vertices mesh{i:02d}:", aux.vertices.shape)
    #split connected components and select the greatest one
    components = aux.split(only_watertight=False)
    largest_component = max(components, key=lambda comp: comp.volume) 
    meshlist.append(TriMesh(largest_component.vertices,largest_component.faces).process(k=200,intrinsic=True))
    print(f"number of vertices mesh{i:02d} After preprocessing:", meshlist[i].vertices.shape)



#################################NEAREST NEIGHBOUR#######################################
p2p_KNN = knn_query(meshlist[0].vertlist, meshlist[1].vertlist)

print('Chamfer Distance Knn: '+str(chamfer_distance(meshlist[0].vertices[p2p_KNN],meshlist[0].vertices)))


#################################ICP##################################################
#subsample some points to perform rigid alignment faster
subsample_list = np.zeros((2, n_sub), dtype=int)
for i in tqdm(range(2)):
    subsample_list[i] = meshlist[i].extract_fps(3000, geodesic=False, verbose=False)

fps1 = subsample_list[0]
fps2 = subsample_list[1]

p2p_21_init_sub = knn_query(meshlist[0].vertlist[fps1], meshlist[1].vertlist[fps2])
_, R, t = icp_align(meshlist[1].vertlist[fps2], meshlist[0].vertlist[fps1],
                            p2p_12=p2p_21_init_sub,
                            return_params=True, n_jobs=20, epsilon=1e-4, verbose=False)

meshlist[1].rotate(R)
meshlist[1].translate(t)

p2p_ICP = knn_query(meshlist[0].vertlist, meshlist[1].vertlist)

print('Chamfer Distance ICP: '+str(chamfer_distance(meshlist[0].vertices[p2p_ICP],meshlist[0].vertices)))



###################################ZO######################################################

# apply zoomout starting from the ICP initialization
FM_12_zo, p2p_ZO = pyFM.refine.zoomout.mesh_zoomout_refine_p2p(p2p_21=p2p_ICP, mesh1=meshlist[0], mesh2=meshlist[1], k_init=20, nit=16, step=5, return_p2p=True, n_jobs=10, verbose=True)

#visualize FMAP
print('Chamfer Distance ICP: '+str(chamfer_distance(meshlist[0].vertices[p2p_ZO],meshlist[0].vertices)))
