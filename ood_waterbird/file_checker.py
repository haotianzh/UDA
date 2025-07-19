import os
import numpy as np 
import pandas as pd

dataset_dir = 'celeba/img_align_celeba'
split = 'celebA/celebA_split.csv'
orig = 'celebA/celebA_original.csv'
ood = 'celebA/celebA_ood.csv'

def check_num_pic(dir):
    files = sorted(list(os.listdir(dir)))
    indices = np.array([int(file[:-4]) for file in files])
    print(len(files))
    print((np.diff(indices, 1) == 1).all())

# def check_example():
    # pass

def check_split_samples(split_file):
    df = pd.read_csv(split_file)
    stats = df.groupby(['split', 'Male', 'Gray_Hair']).count()
    print(stats)


def check_ood_samples(ood_file):
    df = pd.read_csv(ood_file)
    stats = df.groupby([ 'Male', 'Bald']).count()
    print(stats)


def check_overlap_ood_orig(ood, orig):
    df_ood = pd.read_csv(ood)
    df_orig = pd.read_csv(orig)
    img_id_ood = pd.Index(df_ood['image_id'])
    img_id_orig = pd.Index(df_orig['image_id'])
    intersection = img_id_ood.intersection(img_id_orig)
    print(intersection)


if __name__ == "__main__":
    check_num_pic(dataset_dir)
    check_split_samples(split)  
    check_ood_samples(ood)
    check_overlap_ood_orig(ood, orig)