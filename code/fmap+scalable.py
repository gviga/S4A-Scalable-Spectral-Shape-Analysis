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


#################################FMAPS###########################################

# compute landmarks: we select 10 random points of a shape and retrieve the corrispondent points using knn

fps1 = meshlist[0].extract_fps(10, geodesic=False, verbose=False)
fps2 = knn_query(meshlist[1].vertlist, meshlist[0].vertlist[fps1])
landmarks=np.array([fps1,fps2]).T

from pyFM.functional import FunctionalMapping

process_params = {
        'n_ev': (20,20),  # Number of eigenvalues on source and Target
        'landmarks': landmarks,
        'subsample_step': 1,  # In order not to use too many descriptors
        'n_descr': 40, #number of descriptors
        'descr_type': 'HKS',  # WKS or HKS
    }
model = FunctionalMapping(meshlist[0],meshlist[1])
model.preprocess(**process_params,verbose=False)

model.fit(w_descr= 1e-1, w_lap= 1e-3, w_dcomm= 1,w_orient= 0, verbose=False)
fmap12=model.FM       #C{XY}, or A_{YX}^T

plt.imshow(fmap12)
#p2p=model.get_p2p(fmap12, model.mesh1.eigenvectors[:,:k],model.mesh2.eigenvectors[:,:k],adj, bijective)
p2p_FM=model.get_p2p()

print('Chamfer Distance FM: '+str(chamfer_distance(meshlist[0].vertices[p2p_FM],meshlist[0].vertices)))

###################################SCALABLE ZO######################################################
import sys
sys.path.append('../Scalable_FM/')
import large_mesh as lmu


# Define parameters for the process
process_params = {
    'dist_ratio': 3, # rho = dist_ratio * average_radius
    'self_limit': .25,  # Minimum value for self weight
    'correct_dist': False,
    'interpolation': 'poly',
    'return_dist': True,
    'adapt_radius': True,
    'n_jobs':10,
    'n_clusters': 100,
    'verbose': False
}

n_samples = 2000

#compute approx spectrum
U1, Ab1, Wb1, sub1, distmat1 = lmu.process_mesh(meshlist[0], n_samples, **process_params)
evals1, evects1 = lmu.get_approx_spectrum(Wb1, Ab1, verbose=True)

U2, Ab2, Wb2, sub2, distmat2 = lmu.process_mesh(meshlist[1], n_samples, **process_params)
evals2, evects2 = lmu.get_approx_spectrum(Wb2, Ab2, verbose=True)
# apply zoomout starting from the  initialization
# Compute an initial approximate functional map

# You can perform ZoomOut like if you had resampled the whole mesh. This gives you a funcitonal map and a point-to-point map between the two samples (not the whole meshes)
FM_12_zo, p2p_ZO = pyFM.refine.zoomout.zoomout_refine(fmap12, evects1, evects2, nit=16, step=5, A2=Ab2, return_p2p=True, n_jobs=10, verbose=True)

# If you need a dense point-to-point map, you can use FM_12_zo as the functional map between the dense shapes. If needed, an accelerated version of this is described in the paper, but I found that its implementation is very machine-dependant.
p2p_SC = spectral.FM_to_p2p(FM_12_zo, U1@evects1, U2@evects2, n_jobs=10)
print('Chamfer Distance ZO: '+str(chamfer_distance(meshlist[0].vertices[p2p_ZO],meshlist[0].vertices)))
