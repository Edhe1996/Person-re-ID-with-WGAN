# This is the python file for evaluation of the model
import torch
import argparse
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import os
from re_ranking import re_ranking

parser = argparse.ArgumentParser(description='Evaluation arguments')
parser.add_argument('--gpu', action='store_true', help='use gpu or not')
parser.add_argument('--re_ranking', action='store_true', help='use re-ranking or not')

ag = parser.parse_args()

# Use gpu or not
if ag.gpu:
    torch.cuda.set_device(0)
    # Check if GPU is available
    use_gpu = torch.cuda.is_available()
else:
    use_gpu = False

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# Compute average precision
def ap_Computing(ranking_list, good_index, bad_index):
    avg_precision = 0
    num = len(ranking_list)
    CMC = torch.IntTensor(num).zero_()

    if good_index.size == 0:
        CMC[0] = -1
        return avg_precision, CMC

    # Remove the identities of bad indices from the ranking list
    mask = np.in1d(ranking_list, bad_index, invert=True)
    ranking_list = ranking_list[mask]

    # Get the indices of good_index in the ranking_list
    num_good = len(good_index)
    mask = np.in1d(ranking_list, good_index)
    rows_good = np.argwhere(mask == True).flatten()

    # Initialize CMC to ones
    CMC[rows_good[0]:] = 1

    # Calculate average precision among all good identities
    for m in range(num_good):
        precision = (m + 1) * 1.0 / (rows_good[m] + 1)
        if rows_good[m] != 0:
            old_precision = m * 1.0 / rows_good[m]
        else:
            old_precision = 1.0
        avg_precision += (old_precision + precision) / 2

    avg_precision = avg_precision / num_good

    return avg_precision, CMC


def evaluating(qf, ql, qc, gf, gl, gc):
    query = qf.view(-1, 1)
    score = torch.mm(gf, query).squeeze(1).cpu()
    score = score.numpy()
    # print(score)

    # Get the indices of the ranked list with QuickSort
    rank_index = np.argsort(score)
    # Reverse the order
    rank_index = rank_index[::-1]

    # good index
    query_index = np.argwhere(gl == ql)
    camera_index = np.argwhere(gc == qc)
    # Same identities from different cameras
    good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
    # Label = -1 (images that have bad bounding boxes)
    bad_index1 = np.argwhere(gl == -1)
    # Same identities from same cameras
    bad_index2 = np.intersect1d(query_index, camera_index)
    bad_index = np.append(bad_index2, bad_index1)

    ap_cmc_current = ap_Computing(rank_index, good_index, bad_index)

    return ap_cmc_current


# Evaluate
######################################################################
# result = scipy.io.loadmat('features.mat')
result = scipy.io.loadmat('features_erasing_60.mat')
query_feature = torch.FloatTensor(result['query_f'])
query_cam = result['query_cam'][0]
query_label = result['query_label'][0]
gallery_feature = torch.FloatTensor(result['gallery_f'])
gallery_cam = result['gallery_cam'][0]
gallery_label = result['gallery_label'][0]

# If using gpu, map the features to cuda device
if use_gpu:
    query_feature = query_feature.to(device)
    gallery_feature = gallery_feature.to(device)
    print('Using GPU for evaluation')
else:
    print('Using CPU for evaluation')

# Multi-query
multi = os.path.isfile('multi_query_features_erasing.mat')

if multi:
    m_result = scipy.io.loadmat('multi_query_features_erasing.mat')
    mquery_feature = torch.FloatTensor(m_result['mquery_f'])
    mquery_cam = m_result['mquery_cam'][0]
    mquery_label = m_result['mquery_label'][0]
    if use_gpu:
        mquery_feature = mquery_feature.to(device)
        print('Using GPU for evaluation')
    else:
        print('Using CPU for evaluation')

# Initialize CMC and average precision to zeros
CMC = torch.IntTensor(len(gallery_label)).zero_()
ap = 0

for q in range(len(query_label)):
    ap_cur, CMC_cur = evaluating(query_feature[q], query_label[q], query_cam[q],
                                 gallery_feature, gallery_label, gallery_cam)

    if CMC_cur[0] == -1:
        continue
    CMC += CMC_cur
    ap += ap_cur

CMC = CMC.float()
CMC = CMC / len(query_label)
CMC_np = CMC.numpy()
mAP = ap / len(query_label)

fig, ax = plt.subplots()
# Plot the first 15 points of CMC and save it to 'CMC.jpg'
ax.plot(np.arange(1, 31, 1), CMC_np[0:30], 'ro-')
ax.set(xlabel='Rank', ylabel='Matching Accuracy (%)', title='CMC Curve on Market1501')
ax.grid(linestyle='-.', color='b', linewidth=1)
fig.savefig(os.path.join('./model', 'CMC_sq_erasing.jpg'))

# Print rank 1, 5, 10 precision and mAP
print('Rank@1:%f Rank@5:%f Rank@10:%f mAP:%f' % (CMC[0], CMC[4], CMC[9], mAP))

# Multiple-query
if multi:
    CMC = torch.IntTensor(len(gallery_label)).zero_()
    ap = 0.0
    for i in range(len(query_label)):
        mquery_index1 = np.argwhere(mquery_label == query_label[i])
        mquery_index2 = np.argwhere(mquery_cam == query_cam[i])
        mquery_index = np.intersect1d(mquery_index1, mquery_index2)
        mq = torch.mean(mquery_feature[mquery_index, :], dim=0)
        ap_cur, CMC_cur = evaluating(mq, query_label[i], query_cam[i],
                                     gallery_feature, gallery_label, gallery_cam)
        if CMC_cur[0] == -1:
            continue
        CMC += CMC_cur
        ap += ap_cur
    CMC = CMC.float()
    CMC = CMC / len(query_label)
    CMC_np = CMC.numpy()
    mAP = ap / len(query_label)
    fig, ax = plt.subplots()
    # Plot the first 15 points of CMC and save it to 'CMC.jpg'
    ax.plot(np.arange(1, 31, 1), CMC_np[0:30], 'ro-')
    ax.set(xlabel='Rank', ylabel='Matching Accuracy (%)', title='CMC Curve on Market1501')
    ax.grid(linestyle='-.', color='b', linewidth=1)
    fig.savefig(os.path.join('./model', 'CMC_mq_erasing.jpg'))
    print('multi Rank@1:%f Rank@5:%f Rank@10:%f mAP:%f' % (CMC[0], CMC[4], CMC[9], mAP))
