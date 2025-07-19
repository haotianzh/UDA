import os
import pandas as pd 
import numpy as np 
from tqdm import tqdm
from PIL import Image
import torch 
import torch.nn as nn
from torchvision.transforms import v2
from torch.utils.data import Dataset, DataLoader
from model import PretrainedEmbedding

def get_transform_cub(train):
    orig_w = 178
    orig_h = 218
    orig_min_dim = min(orig_w, orig_h)
    target_resolution = (224, 224)

    if not train:
        transform = v2.Compose([
            v2.CenterCrop(orig_min_dim),
            v2.Resize(target_resolution),
            v2.ToTensor(),
            v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        # Orig aspect ratio is 0.81, so we don't squish it in that direction any more
        transform = v2.Compose([
            v2.RandomResizedCrop(
                target_resolution,
                scale=(0.7, 1.0),
                ratio=(1.0, 1.3333333333333333),
                interpolation=2),
            v2.RandomHorizontalFlip(),
            v2.ToTensor(),
            v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    return transform

class celebADatasetOOD(Dataset):
    def __init__(self, data_dir, metadata_file):
        super().__init__()
        self.data_dir = data_dir
        self.metadata = pd.read_csv(metadata_file)
        self.imgs = self.metadata['image_id'].tolist()
        self.y = self.metadata['Male'].tolist()
        self.a = self.metadata['Gray_Hair'].tolist()
        self.transform = get_transform_cub(train=False)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        y = self.y[idx]
        a = self.a[idx]
        img_filename = os.path.join(
            self.data_dir,
            self.imgs[idx])
        img = Image.open(img_filename).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, y, a, img_filename
    

def get_embeddings(data_dir, metadata_filename, backbone='resnet18'):
    batch_size = 64
    dataset = celebADatasetOOD(data_dir, metadata_filename)
    dataloader = DataLoader(dataset, 
                            batch_size=batch_size, 
                            pin_memory=True, 
                            num_workers=8,
                            drop_last=False) # don't drop last
    model = PretrainedEmbedding(backbone=backbone).cuda()
    embeddings = []
    ys = []
    aas = []
    filenames = []
    ctr = 0
    for img, y, a, img_filename in tqdm(dataloader):
        # if ctr == 2:
        #     break
        ctr += 1
        img = img.cuda(non_blocking=True)
        with torch.no_grad():
            output = model(img).to('cpu').numpy()
        embeddings.extend(output.reshape(output.shape[0], output.shape[1]*output.shape[2]*output.shape[3]))
        aas.extend(a.numpy())
        ys.extend(y.numpy())
        filenames.extend(img_filename)
    metadata = pd.DataFrame({'img_filename': filenames,
                       'y': ys,
                       'a': aas})
    embeddings = np.array(embeddings)
    return embeddings, metadata



if __name__ == "__main__":
    embed_dir = 'embeds'
    backbone = 'resnet18'
    e, df = get_embeddings('./celeba/img_align_celeba', './celebA/celebA_original.csv', backbone=backbone)
    print(e.shape)
    # np.save(os.path.join(embed_dir, f'embed_{backbone}.npy'), e)
    # df.to_csv(os.path.join(embed_dir, f'metadata_{backbone}.csv'))    


