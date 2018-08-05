# This is the python file to extract the feature of query and gallery images

from __future__ import print_function
from __future__ import division

if __name__ == '__main__':
    import argparse
    import torch
    import torch.nn as nn
    from torch.autograd import Variable
    from torchvision import datasets, transforms
    import os
    import scipy.io
    from model import ResNet50Baseline

    # Command line interfaces
    parser = argparse.ArgumentParser(description='Test arguments')
    parser.add_argument('--gpu', action='store_true', help='use gpu or not')
    parser.add_argument('--which_epoch', default='last', type=str, help='0,1,2,3...or last')
    parser.add_argument('--data_dir', default='/Users/edwar/Dataset/DukeMTMC/pytorch', type=str,
                        help='test data path')
    parser.add_argument('--model_name', default='ResNet50', type=str, help='model name')
    parser.add_argument('--batchsize', default=32, type=int, help='batch size')
    parser.add_argument('--multi', action='store_true', help='use multiple query')

    ag = parser.parse_args()

    data_dir = ag.data_dir
    model_name = ag.model_name

    # Use gpu or not
    if ag.gpu:
        torch.cuda.set_device(0)
        # Check if GPU is available
        use_gpu = torch.cuda.is_available()
    else:
        use_gpu = False
    if use_gpu:
        print('Use GPU for testing\n')

    # Data pre-processing
    test_preprocessing = transforms.Compose([
        transforms.Resize((288, 144), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load images from data path and apply the transformations
    # DukeMTMC has no multi-query mode
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), test_preprocessing)
                      for x in ['gallery', 'query']}

    # Provide iterators over the gallery, query and multi-query datasets
    data_itr = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=ag.batchsize,
                                               shuffle=False, num_workers=16)
                for x in ['gallery', 'query']}

    # How many classes in query images
    class_names = image_datasets['query'].classes

    # Load trained model
    def load_model(model):
        save_path = os.path.join('./DukeMTMC', model_name, 'network_%s.pth' % ag.which_epoch)
        model.load_state_dict(torch.load(save_path))
        return model

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

    def get_id(img_path):
        camera_id = []
        labels = []

        for path, v in img_path:
            filename = os.path.basename(path)
            label = filename[0:4]
            camera = filename.split('c')[1]
            if label[0:2] == '-1':
                labels.append(-1)
            else:
                labels.append(int(label))
            camera_id.append(int(camera[0]))
        return camera_id, labels


    gallery_path = image_datasets['gallery'].imgs
    query_path = image_datasets['query'].imgs
    # mquery_path = image_datasets['multi-query'].imgs

    gallery_cam, gallery_label = get_id(gallery_path)
    query_cam, query_label = get_id(query_path)
    # mquery_cam, mquery_label = get_id(mquery_path)

    # Test process
    ################################################
    # DukeMTMC-reid has 702 IDs in query images (1110 in gallery)
    print('Start test')
    print('-'*20)
    model_structure = ResNet50Baseline(702)
    model = load_model(model_structure)
    # Remove the final fc and classification layer
    model.model.fc = nn.Sequential()
    model.classifier = nn.Sequential()
    model = model.eval()

    # Map the model to cuda devices
    if use_gpu:
        model = model.cuda()

    # Extract features
    gallery_feature = extract_feature(model, data_itr['gallery'])
    query_feature = extract_feature(model, data_itr['query'])

    # Save the features to .mat file
    result = {'gallery_f': gallery_feature.numpy(), 'gallery_label': gallery_label, 'gallery_cam': gallery_cam,
              'query_f': query_feature.numpy(), 'query_label': query_label, 'query_cam': query_cam}
    scipy.io.savemat('features_duke_60.mat', result)

    print('Test process finished')
