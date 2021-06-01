import csv
import os
from math import ceil
from random import shuffle

################ TEST DATASET ################

def add_label_zeros(label):
    return ("0" * (5 - len(label))) + label

def parse_to_struct(filepath):
    files = []
    with open(filepath) as fin:
        csv_reader = csv.reader(fin, delimiter=';')
        i = 0
        for row in csv_reader:
            if i != 0:
                file = row[0]
                label = add_label_zeros(row[7])
                files.append((file, label))
            i += 1
    return files

# def move_files(files, from_dir, to_dir):
#     for file, label in files:
#         if not os.path.exists(f'{to_dir}/{label}'):
#             os.makedirs(f'{to_dir}/{label}')
#         os.rename(f'{from_dir}/{file}',f'{to_dir}/{label}/{file}')

def move_files(files, from_dir, to_dir):
    for file, label in files:
        if not os.path.exists(f'{to_dir}/{label}'):
            os.makedirs(f'{to_dir}/{label}')
        os.rename(f'{from_dir}/{label}/{file}',f'{to_dir}/{label}/{file}')

################ TRAIN DATASET ################

def split_dataset(folderpath, train_set_size=0.7):
    files = []
    labels = os.listdir(folderpath)
    for label in labels:
        imgs = os.listdir(f'{folderpath}/{label}')
        imgs_labels = list(zip(imgs,[label]*len(imgs)))
        files += imgs_labels

    tmp = ceil(train_set_size * len(files))
    shuffle(files)
    train_files = files[:tmp]
    test_files = files[tmp:]
    return train_files, test_files


###############################################

if __name__ == '__main__':
    #files = parse_to_struct('data/GTSRB/Final_Test/GT-final_test.csv')
    #move_files(files,'data/GTSRB/Final_Test/Images','data/gtsrb_full/val_images')

    train_files, test_files = split_dataset('data/gtsrb_full/GTSRB/Final_Training/Images')
    move_files(train_files,'data/gtsrb_full/GTSRB/Final_Training/Images','data/gtsrb_full/train_images')
    move_files(test_files,'data/gtsrb_full/GTSRB/Final_Training/Images','data/gtsrb_full/test_images')
    

#data/GTSRB/Final_Test/Images/{file}
#data/gtsrb_full/val_images

#data/gtsrb_full/train_images
#data/gtsrb_full/test_images