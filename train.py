# This is the python file for training the network

from __future__ import print_function
from __future__ import division

# Add this for implementation on Windows
if __name__ == '__main__':

    import argparse
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.optim import lr_scheduler
    from torch.autograd import Variable
    from torchvision import datasets, transforms
    import matplotlib
    # Generate images without having a window appear
    matplotlib.use('agg')
    import matplotlib.pyplot as plt
    from model import ResNet50Baseline
    from random_erasing import RandomErasing
    import time
    import os
    import json

    # Command line interfaces
    # Default model is ResNet50 and default batch size is 32
    parser = argparse.ArgumentParser(description='Training arguments')
    parser.add_argument('--gpu', action='store_true', help='use gpu or not')
    parser.add_argument('--model_name', default='ResNet50_erasing', type=str, help='model name')
    parser.add_argument('--data_dir', default='/Users/edwar/Dataset/Market1501/pytorch', type=str,
                        help='training data path')
    parser.add_argument('--train_all', action='store_true', help='use all training data')
    parser.add_argument('--color_jitter', action='store_true', help='use color jitter in training')
    parser.add_argument('--batchsize', default=32, type=int, help='batch size')
    parser.add_argument('--epoch', default=60, type=int, help='number of epochs to train')
    parser.add_argument('--erasing_p', default=0.5, type=float, help='Random Erasing probability, in [0,1]')

    ag = parser.parse_args()

    data_dir = ag.data_dir
    model_name = ag.model_name
    num_epoch = ag.epoch

    # Use gpu or not
    if ag.gpu:
        torch.cuda.set_device(0)
        # Check if GPU is available
        use_gpu = torch.cuda.is_available()
    else:
        use_gpu = False

    # Data preprocessing and loading for Market1501
    train_preprocessing = [
        transforms.Resize((288, 144), interpolation=3),
        transforms.RandomCrop((256, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]

    val_preprocessing = [
        transforms.Resize((256, 128), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]

    if ag.erasing_p > 0:
        train_preprocessing = train_preprocessing + [RandomErasing(probability=ag.erasing_p, mean=[0.0, 0.0, 0.0])]

    if ag.color_jitter:
        train_preprocessing = [transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0)]\
                              + train_preprocessing

    # Compose all the transformations
    data_preprocessing = {
        'train': transforms.Compose(train_preprocessing),
        'val': transforms.Compose(val_preprocessing),
    }

    # Use all the training samples or not
    train_all = ''
    if ag.train_all:
        train_all = '_all'

    # Load images from data path and apply the transformations
    image_datasets = {'train': datasets.ImageFolder(os.path.join(data_dir, 'train' + train_all),
                                                    data_preprocessing['train']),
                      'val': datasets.ImageFolder(os.path.join(data_dir, 'val'),
                                                  data_preprocessing['val'])}

    # Provide iterators over the train and val datasets
    data_itr = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=ag.batchsize,
                                               shuffle=True, num_workers=16)
                for x in ['train', 'val']}
    data_size = {x: len(image_datasets[x]) for x in ['train', 'val']}
    print('Data size:')
    print(data_size)
    class_names = image_datasets['train'].classes

    inputs, classes = next(iter(data_itr['train']))

    # Training process
    ######################################
    # Create some dicts to save the loss and error
    y_loss = {'train': [], 'val': []}
    y_err = {'train': [], 'val': []}

    # Training function
    def train_process(model, criterion, optimizer, scheduler, num_epochs):
        # Record the training time
        begin_time = time.time()

        best_weights = model.state_dict()
        best_acc = 0.0

        for epoch in range(num_epochs):
            print('Epoch {} of {}'.format(epoch + 1, num_epochs))
            print('-' * 15)

            # train and validation modes respectively
            for mode in ['train', 'val']:
                if mode == 'train':
                    scheduler.step()
                    model.train(True)
                else:
                    model.train(False)

                loss = 0.0
                num_corrects = 0

                # Iterate over the train and validation data
                for data in data_itr[mode]:
                    inputs, labels = data
                    # Data format is [batch_size, channels, height, width]
                    now_batch_size, c, h, w = inputs.shape
                    if now_batch_size == 1:
                        continue

                    # If GPU is available, map the data to GPU
                    if use_gpu:
                        inputs = Variable(inputs.cuda())
                        labels = Variable(labels.cuda())
                    else:
                        inputs = Variable(inputs)
                        labels = Variable(labels)

                    # Zero the gradients of parameters
                    optimizer.zero_grad()

                    # Forward process
                    outputs = model(inputs)
                    _, predictions = torch.max(outputs.data, 1)
                    entropy_loss = criterion(outputs, labels)

                    # Backward process, (only in train mode)
                    if mode == 'train':
                        entropy_loss.backward()
                        optimizer.step()

                    loss += entropy_loss.item()
                    num_corrects += torch.sum(predictions == labels.data)

                epoch_loss = loss / data_size[mode]
                epoch_acc = num_corrects.item() / data_size[mode]

                print('{} Loss: {:.4f}, Accuracy: {:.4f}'.format(mode, epoch_loss, epoch_acc))

                y_loss[mode].append(epoch_loss)
                y_err[mode].append(1 - epoch_acc)

                if mode == 'val':
                    last_weights = model.state_dict()
                    if epoch%10 == 9:
                        # Save the trained networks
                        save_network(model, epoch + 1)
                    draw_curve(epoch + 1)

            print()

        end_time = time.time()
        used_time = end_time - begin_time
        print('Training process completes in {:.0f}m {:.0f}s'.format(
            used_time // 60, used_time % 60
        ))

        model.load_state_dict(last_weights)
        save_network(model, 'last')

        return model


    # Save networks
    ###############################################
    def save_network(model, epoch):
        save_filename = 'network_%s.pth' % epoch
        save_path = os.path.join('./model', model_name, save_filename)
        torch.save(model.cpu().state_dict(), save_path)
        if torch.cuda.is_available():
            model.cuda(0)


    # Save training curve to a file
    ###############################################
    x_epoch = []
    fig = plt.figure()
    ax0 = fig.add_subplot(121, title="loss")
    ax1 = fig.add_subplot(122, title="error")


    def draw_curve(current_epoch):
        x_epoch.append(current_epoch)
        ax0.plot(x_epoch, y_loss['train'], 'bo-', label='train')
        ax0.plot(x_epoch, y_loss['val'], 'ro-', label='val')
        ax1.plot(x_epoch, y_err['train'], 'bo-', label='train')
        ax1.plot(x_epoch, y_err['val'], 'ro-', label='val')
        if current_epoch == 0:
            ax0.legend()
            ax1.legend()
        fig.savefig(os.path.join('./model', model_name, 'train_result.jpg'))


    model = ResNet50Baseline(len(class_names))
    print(model)
    if use_gpu:
        model = model.cuda()
    # Use cross entropy loss
    criterion = nn.CrossEntropyLoss()

    # Decide how to optimize the loss function (SGD here)
    ignored_params = list(map(id, model.model.fc.parameters())) + list(map(id, model.classifier.parameters()))
    base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
    # Stochastic gradient descent with momentum
    # Set the learning rate of pretrained parameters as 0.01
    # Set the learning rate of the other parameters as 0.1
    optimizer_ft = optim.SGD([
        {'params': base_params, 'lr': 0.01},
        {'params': model.model.fc.parameters(), 'lr': 0.1},
        {'params': model.classifier.parameters(), 'lr': 0.1}
    ], weight_decay=5e-4, momentum=0.9, nesterov=True)

    # Sets the learning rate of each parameter group to the initial lr
    # decayed by gamma every step_size epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=40, gamma=0.1)

    dir_name = os.path.join('./model', model_name)
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)

    # save the arguments to a json file
    with open('%s/args.json' % dir_name, 'w') as fp:
        json.dump(vars(ag), fp, indent=1)

    # Train and validate for 60 epochs
    model = train_process(model, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=num_epoch)











