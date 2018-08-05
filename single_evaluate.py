# For every query image, output their re-identification results

import torch
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.image as mtimg
import numpy as np
import scipy.io
import os
from torchvision import datasets
# from re_ranking import re_ranking

# Which query image to use
q = 1

# Use ImageFolder to plot the ranked images
data_dir = '/Users/edwar/Dataset/Market1501/pytorch'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x))
                  for x in ['gallery', 'query', 'multi-query']}
gallery_path = image_datasets['gallery'].imgs
query_path = image_datasets['query'].imgs
mquery_path = image_datasets['multi-query'].imgs

# Evaluate
######################################################################
result = scipy.io.loadmat('features.mat')
query_feature = torch.FloatTensor(result['query_f'])
query_cam = result['query_cam'][0]
query_label = result['query_label'][0]
gallery_feature = torch.FloatTensor(result['gallery_f'])
gallery_cam = result['gallery_cam'][0]
gallery_label = result['gallery_label'][0]

# Multi-query
multi = os.path.isfile('multi_query.mat')

if multi:
    m_result = scipy.io.loadmat('multi_query.mat')
    mquery_feature = torch.FloatTensor(m_result['mquery_f'])
    mquery_cam = m_result['mquery_cam'][0]
    mquery_label = m_result['mquery_label'][0]

qf = query_feature[q]
ql = query_label[q]
qc = query_cam[q]

query = qf.view(-1, 1)
score = torch.mm(gallery_feature, query).squeeze(1).cpu()
score = score.numpy()
# print(score)

rank_index = np.argsort(score)
rank_index = rank_index[::-1]

# good index
query_index = np.argwhere(gallery_label == ql)
camera_index = np.argwhere(gallery_cam == qc)
# Same identities from different cameras
good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
# Label = -1 (images that have bad bounding boxes)
bad_index1 = np.argwhere(gallery_label == -1)
# Same identities from same cameras
bad_index2 = np.intersect1d(query_index, camera_index)
bad_index = np.append(bad_index2, bad_index1)

# Remove the identities of bad indices from the ranking list
mask = np.in1d(rank_index, bad_index, invert=True)
rank_index = rank_index[mask]

plot_index = rank_index[0:5]
fig = plt.figure(figsize=(10, 10))
columns = 2
rows = 5
img = mtimg.imread(query_path[q][0])
fig.add_subplot(rows, columns, 1)
plt.imshow(img)
j = 2
for i in plot_index:
    img = mtimg.imread(gallery_path[i][0])
    fig.add_subplot(rows, columns, j)
    plt.imshow(img)
    j += 2
# plt.show()
fig.savefig(os.path.join('./model', 're_id.jpg'))
