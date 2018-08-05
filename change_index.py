# -*- coding: utf-8 -*-
# change name of the folder(e.g.  0002,0007,0010,0011...  to 0,1,2,3)
import os
from shutil import copyfile

original_path = '/Users/edwar/WassersteinGAN/fake_resize/pytorch'


# copy folder tree from source to destination
def copyfolder(src, dst):
    files = os.listdir(src)
    if not os.path.isdir(dst):
        os.mkdir(dst)
    for tt in files:
        copyfile(src+'/'+tt, dst+'/'+tt)


train_save_path = '/Users/edwar/WassersteinGAN/final_fake'
data_path = original_path
if not os.path.isdir(train_save_path):
    os.mkdir(train_save_path)

folders = os.listdir(data_path)
for foldernames in folders:
    copyfolder(data_path+'/'+foldernames, train_save_path+'/'+str(foldernames).zfill(4))


