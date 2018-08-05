# This is the python file for assigning labels for generated images
# And then add them to the original dataset

from __future__ import print_function
from __future__ import division

# Add this for implementation on Windows
if __name__ == '__main__':

    import torch
    import torch.nn as nn
    from torchvision import datasets, transforms
    from torch.autograd import Variable
    import numpy as np
    import matplotlib
    # Generate images without having a window appear
    matplotlib.use('agg')
    import matplotlib.pyplot as plt
    from model import ResNet50Baseline
    import time
    import os
    import scipy.io
    from shutil import copyfile

    use_gpu = True

    data_dir = '/Users/edwar/WassersteinGAN/fake_resize/pytorch'
    model = ResNet50Baseline(751)
    model.load_state_dict(torch.load('./model/ResNet50/network_last.pth'))
    model.model.fc = nn.Sequential()
    model.classifier = nn.Sequential()
    model = model.eval()
    if use_gpu:
        model = model.cuda()

    # copy folder tree from source to destination
    def copyfolder(src, dst):
        files = os.listdir(src)
        if not os.path.isdir(dst):
            os.mkdir(dst)
        for tt in files:
            copyfile(src + '/' + tt, dst + '/' + tt)

    preprocessing = transforms.Compose([
        # transforms.CenterCrop((64, 32)),
        transforms.Resize((288, 144), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    fake_dataset = datasets.ImageFolder(os.path.join(data_dir), preprocessing)

    data_itr = torch.utils.data.DataLoader(fake_dataset, batch_size=2,
                                           shuffle=False)

    # Extract features for query and gallery images
    # Flip the images horizontally
    def flipimg(img):
        inv_idx = torch.arange(img.size(3) - 1, -1, -1).long()  # N x C x H x W
        img_flip = img.index_select(3, inv_idx)
        return img_flip


    def extract_feature(model, data_itr):
        # initialize the feature tensor
        features = torch.FloatTensor()
        # which image is processing
        count = 0

        for data in data_itr:
            img, label = data
            # [batch_size, channels, height, width]
            n, c, h, w = img.size()
            count += n
            # print(count)
            out = torch.FloatTensor(n, 2048).zero_()

            for i in range(2):
                if i == 1:
                    img = flipimg(img)
                if use_gpu:
                    input_img = Variable(img.cuda())
                else:
                    input_img = Variable(img)
                outputs_of_model = model(input_img)
                out_data = outputs_of_model.data.cpu()
                out = out + out_data

            out_norm = torch.norm(out, p=2, dim=1, keepdim=True)
            out = out.div(out_norm.expand_as(out))

            features = torch.cat((features, out), 0)

        return features


    fake_feature = extract_feature(model, data_itr)
    result = {'fake_f': fake_feature.numpy()}
    scipy.io.savemat('fake_feature.mat', result)

    original_result = scipy.io.loadmat('train_feature.mat')
    train_feature = torch.FloatTensor(original_result['train_f'])
    train_cam = original_result['train_cam'][0]
    train_label = original_result['train_label'][0]

    labels = []
    for f in fake_feature:
        fake = f.view(-1, 1)
        score = torch.mm(train_feature, fake).squeeze(1).cpu()
        score = score.numpy()
        # print(score)
        rank_index = np.argsort(score)
        rank_index = rank_index[::-1]
        labels.append(train_label[rank_index[0]])

    print(labels)
    train_save_path = '/Users/edwar/Dataset/Market1501/pytorch/train_all'
    label_index = 0
    folders = os.listdir(data_dir)

    # Add the generated images to the original dataset
    for foldernames in folders:
        copyfolder(data_dir + '/' + foldernames, train_save_path + '/' + str(labels[label_index]).zfill(4))
        label_index += 1



