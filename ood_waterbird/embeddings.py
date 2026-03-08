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

def get_transform_cub():
    target_resolution = (224, 224)
    transform = v2.Compose([
            v2.CenterCrop(target_resolution),
            v2.RandomHorizontalFlip(),
            v2.ToTensor(),
            v2.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        
    return transform

class WaterbirdDatasetOOD(Dataset):
    def __init__(self, data_dir, metadata_file):
        super().__init__()
        self.data_dir = data_dir
        self.metadata = pd.read_csv(metadata_file)
        self.imgs = self.metadata['img_filename'].tolist()
        self.y = self.metadata['y'].tolist()
        self.a = self.metadata['place'].tolist()
        self.transform = get_transform_cub()

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
    dataset = WaterbirdDatasetOOD(data_dir, metadata_filename)
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
            # print(output.shape)
        if 'vit' in backbone:
            embeddings.extend(output)
        if 'resnet' in backbone:
            embeddings.extend(output.reshape(output.shape[0], -1))
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
    backbone = 'resnet50'


    # dataset = WaterbirdDatasetOOD('../waterbird/datasets/waterbird_complete50_forest2water2', '../waterbird/datasets/datasets/waterbirds/metadata.csv')

    # print(dataset[3])


    e, df = get_embeddings('../waterbird/datasets/waterbird_complete50_forest2water2', '../waterbird/datasets/datasets/waterbirds/metadata.csv', backbone=backbone)
    # print(e.shape)
    np.save(os.path.join(embed_dir, f'embeds_{backbone}.npy'), e)
    df.to_csv(os.path.join(embed_dir, f'metadata_{backbone}.csv'))    


