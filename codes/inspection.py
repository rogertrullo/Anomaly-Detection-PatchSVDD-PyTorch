from codes import mvtecad
import numpy as np
import torch
from torch.utils.data import DataLoader
from .utils import PatchDataset_NCHW, NHWC2NCHW, distribute_scores


__all__ = ['eval_encoder_NN_multiK', 'eval_embeddings_NN_multiK']


def infer(x_, enc, K, S):
    x = NHWC2NCHW(x_)
    print('computing patch dataset and dataloader')
    dataset = PatchDataset_NCHW(x, K=K, S=S)
    loader = DataLoader(dataset, batch_size=2048, shuffle=False, pin_memory=True, num_workers=8)
    embs = np.empty((dataset.N, dataset.row_num, dataset.col_num, enc.D), dtype=np.float32)  # [-1, I, J, D]
    enc = enc.eval()
    print(f'computing actual embs for {len(dataset)} patches')
    with torch.no_grad():
        for xs, ns, iis, js in loader:
            xs = xs.cuda()
            embedding = enc(xs)
            embedding = embedding.detach().cpu().numpy()

            for embed, n, i, j in zip(embedding, ns, iis, js):
                embs[n, i, j] = np.squeeze(embed)
    return embs


def assess_anomaly_maps(obj, anomaly_maps):
    print('computing auc seg')
    auroc_seg = mvtecad.segmentation_auroc(obj, anomaly_maps)
    print('computing max for dets')

    anomaly_scores = anomaly_maps.max(axis=-1).max(axis=-1)
    print('computing auc det')
    auroc_det = mvtecad.detection_auroc(obj, anomaly_scores)
    return auroc_det, auroc_seg


#########################

def eval_encoder_NN_multiK(enc, obj, x_tr, x_te):
    print('loading training and testing datasets')
    #x_tr = mvtecad.get_x_standardized(obj, mode='train')
    #x_te = mvtecad.get_x_standardized(obj, mode='test')
    print('computing embeddings 64')
    embs64_tr = infer(x_tr, enc, K=64, S=16)
    embs64_te = infer(x_te, enc, K=64, S=16)

    #x_tr = mvtecad.get_x_standardized(obj, mode='train')
    #x_te = mvtecad.get_x_standardized(obj, mode='test')
    print('computing embeddings 32')
    embs32_tr = infer(x_tr, enc.enc, K=32, S=4)
    embs32_te = infer(x_te, enc.enc, K=32, S=4)

    embs64 = embs64_tr, embs64_te
    embs32 = embs32_tr, embs32_te

    return eval_embeddings_NN_multiK(obj, embs64, embs32)


def eval_embeddings_NN_multiK(obj, embs64, embs32, NN=1):
    emb_tr, emb_te = embs64
    print('nn 64')
    maps_64 = measure_emb_NN(emb_te, emb_tr, method='kdt', NN=NN)
    print('distribute_scores 64')
    maps_64 = distribute_scores(maps_64, (256, 256), K=64, S=16)
    print('assessing maps 64')
    det_64, seg_64 = assess_anomaly_maps(obj, maps_64)
    print('det auc64',det_64,'seg auc64',seg_64)

    emb_tr, emb_te = embs32
    print('nn 32')
    maps_32 = measure_emb_NN(emb_te, emb_tr, method='ngt', NN=NN)
    print('distribute_scores 64')
    maps_32 = distribute_scores(maps_32, (256, 256), K=32, S=4)
    print('assessing maps 32')
    det_32, seg_32 = assess_anomaly_maps(obj, maps_32)
    print('det auc32',det_32,'seg auc32',seg_32)
    
    maps_sum = maps_64 + maps_32
    print('assessing maps 64+32')
    det_sum, seg_sum = assess_anomaly_maps(obj, maps_sum)

    maps_mult = maps_64 * maps_32
    print('assessing maps 64*32')
    det_mult, seg_mult = assess_anomaly_maps(obj, maps_mult)

    return {
        'det_64': det_64,
        'seg_64': seg_64,

        'det_32': det_32,
        'seg_32': seg_32,

        'det_sum': det_sum,
        'seg_sum': seg_sum,

        'det_mult': det_mult,
        'seg_mult': seg_mult,

        'maps_64': maps_64,
        'maps_32': maps_32,
        'maps_sum': maps_sum,
        'maps_mult': maps_mult,
    }


########################

def measure_emb_NN(emb_te, emb_tr, method='kdt', NN=1):
    from .nearest_neighbor import search_NN
    print('emb training',emb_tr.shape, 'emb testing',emb_te.shape)
    D = emb_tr.shape[-1]
    train_emb_all = emb_tr.reshape(-1, D)
    print('train_emb_all',train_emb_all.shape)

    l2_maps, _ = search_NN(emb_te, train_emb_all, method=method, NN=NN)
    anomaly_maps = np.mean(l2_maps, axis=-1)

    return anomaly_maps
