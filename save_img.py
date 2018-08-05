# This is the python file for saving random erased images to './erasing_img/'
if __name__ == '__main__':
    import torch
    import torch.utils.data
    import os
    from torchvision import datasets, transforms
    import torchvision.utils as vutils
    from random_erasing import RandomErasing

    data_dir = '/Users/edwar/Dataset/Market1501/pytorch/train_all'
    preprocessing = transforms.Compose([
        transforms.Resize((256, 128), interpolation=3),
        # transforms.RandomCrop((256, 128)),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        RandomErasing(probability=0.5, mean=[0.0, 0.0, 0.0])
    ])

    image_datasets = datasets.ImageFolder(os.path.join(data_dir), preprocessing)
    data_itr = torch.utils.data.DataLoader(image_datasets, batch_size=16, shuffle=True, num_workers=16)

    i = 0
    for data in data_itr:
        inputs, _ = data
        vutils.save_image(inputs, '{0}/{1}_erasing.jpg'.format('erasing_img', i))
        i += 1

