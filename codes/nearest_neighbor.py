import numpy as np
import shutil
import os


__all__ = ['search_NN']


def search_NN(test_emb, train_emb_flat, NN=1, method='kdt'):
    if method == 'ngt':
        return search_NN_ngt(test_emb, train_emb_flat, NN=NN)

    from sklearn.neighbors import KDTree
    kdt = KDTree(train_emb_flat)

    Ntest, I, J, D = test_emb.shape
    closest_inds = np.empty((Ntest, I, J, NN), dtype=np.int32)
    l2_maps = np.empty((Ntest, I, J, NN), dtype=np.float32)
    ######Roger###############
    test_emb_flat=test_emb.reshape(-1,D)
    
    dists, inds = kdt.query(test_emb_flat, return_distance=True, k=NN)#N,1
    closest_inds=inds.reshape(Ntest, I, J, NN)
    l2_maps=dists.reshape(Ntest, I, J, NN)
    ################################
    
    
#     for n in range(Ntest):
#         for i in range(I):
            
#             dists, inds = kdt.query(test_emb[n, i, :, :], return_distance=True, k=NN)
#             if n==0 and n ==0:
#                 print('query shape',test_emb[n, i, :, :].shape)
#                 print('dists shape',dists.shape, 'inds shape',inds.shape)
#             closest_inds[n, i, :, :] = inds[:, :]
#             l2_maps[n, i, :, :] = dists[:, :]

    return l2_maps, closest_inds


def search_NN_ngt(test_emb, train_emb_flat, NN=1):
    import ngtpy

    Ntest, I, J, D = test_emb.shape
    closest_inds = np.empty((Ntest, I, J, NN), dtype=np.int32)
    l2_maps = np.empty((Ntest, I, J, NN), dtype=np.float32)

    # os.makedirs('tmp', exist_ok=True)
    dpath = f'/tmp/{os.getpid()}'
    ngtpy.create(dpath, D)
    index = ngtpy.Index(dpath)
    index.batch_insert(train_emb_flat, num_threads=24)

    for n in range(Ntest):
        for i in range(I):
            for j in range(J):
                query = test_emb[n, i, j, :]
                results = index.search(query, NN)
                inds = [result[0] for result in results]

                closest_inds[n, i, j, :] = inds
                vecs = np.asarray([index.get_object(inds[nn]) for nn in range(NN)])
                dists = np.linalg.norm(query - vecs, axis=-1)
                l2_maps[n, i, j, :] = dists
    shutil.rmtree(dpath)

    return l2_maps, closest_inds
