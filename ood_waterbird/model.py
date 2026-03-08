import torch 
import torch.nn as nn
import torchvision as tv
from torchsummary import summary


class ResNetPartial(nn.Module):
    def __init__(self, n_classes):
        super(ResNetPartial, self).__init__()
        resnet = tv.models.resnet18()
        # # freezing parameters
        # for param in resnet.parameters():
        #     param.requires_grad = False
        # convolutional layers of resnet34
        layers = list(resnet.children())[:8]
        self.top_model = nn.Sequential(*layers)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(512)
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, n_classes)
    
    def forward(self, x):
        x = torch.relu(self.top_model(x))
        x = nn.AdaptiveAvgPool2d((1,1))(x)
        x = x.view(x.shape[0], -1) # flattening 
        x = self.bn1(x)
        x = self.fc1(x)
        self.embedding = x
        x = torch.relu(x)
        x = self.bn2(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x


class ResNet(nn.Module):
    def __init__(self, n_classes, pretrained=None):
        super(ResNet, self).__init__()
        # assert n_layer in [18, 34, 50, 101], "no backbone."
        # self.n_layer = n_layer
        self.n_classes = n_classes
        self.pretrained = pretrained
        self.model = self.get_resnet18_full_model()
        # self.model = self.get_resnet18_part_model()

    def get_resnet18_part_model(self):
        model = ResNetPartial(self.n_classes)
        return model

    def get_resnet18_full_model(self):
        # model = torch.hub.load('pytorch/vision:v0.10.0', f'resnet{n_layer}', weights=pretrained)
        if self.pretrained:
            model = tv.models.resnet18(weights=tv.models.ResNet18_Weights.IMAGENET1K_V1)
            for param in model.parameters():
                param.requires_grad = False
        else:
            model = tv.models.resnet18()
            
        n_last_in = model.fc.in_features
        model.fc = nn.Sequential(
               nn.Linear(n_last_in, 512),
               nn.ReLU(inplace=True),
               nn.Linear(512, self.n_classes),
               nn.Sigmoid()
               )
        return model

    def forward(self, x):
        return self.model(x)


class Model(nn.Module):
    def __init__(self, n_classes, backbone, pretrained=None):
        super(Model, self).__init__()
        # assert n_layer in [18, 34, 50, 101], "no backbone."
        # self.n_layer = n_layer
        backbones = {'resnet18': self.get_resnet18, 
                     'resnet50': self.get_resnet50, 
                     'vgg11': self.get_vgg11, 
                     'vgg19': self.get_vgg19}
        assert backbone in backbones, "choose valid backbone"
        self.n_classes = n_classes
        self.pretrained = pretrained
        self.backbone = backbone
        self.model = backbones[backbone]()

    def get_resnet18_part_model(self):
        model = ResNetPartial(self.n_classes)
        return model

    def get_resnet18(self):
        if self.pretrained:
            model = tv.models.resnet18(weights=tv.models.ResNet18_Weights.IMAGENET1K_V1)
            for param in model.parameters():
                param.requires_grad = False
        else:
            model = tv.models.resnet18()
        n_last_in = model.fc.in_features
        model.fc = nn.Sequential(
               nn.Linear(n_last_in, 512),
               nn.ReLU(inplace=True),
               nn.Linear(512, self.n_classes),
               nn.Sigmoid()
               )
        return model
    
    def get_resnet50(self):
        if self.pretrained:
            model = tv.models.resnet50(weights=tv.models.ResNet50_Weights.IMAGENET1K_V1)
            for param in model.parameters():
                param.requires_grad = False
        else:
            model = tv.models.resnet50()
        n_last_in = model.fc.in_features
        model.fc = nn.Sequential(
               nn.Linear(n_last_in, 512),
               nn.ReLU(inplace=True),
               nn.Linear(512, self.n_classes),
               nn.Sigmoid()
               )
        return model
    
    def get_vgg11(self):
        if self.pretrained:
            model = tv.models.vgg11(weights=tv.models.VGG11_Weights.IMAGENET1K_V1)
            for param in model.parameters():
                param.requires_grad = False
        else:
            model = tv.models.vgg11()
        n_last_in = model.classifier[0].in_features
        model.classifier = nn.Sequential(
               nn.Linear(n_last_in, 512),
               nn.ReLU(inplace=True),
               nn.Linear(512, self.n_classes),
               nn.Sigmoid()
               )
        return model
    
    def get_vgg19(self):
        if self.pretrained:
            model = tv.models.vgg19(weights=tv.models.VGG19_Weights.IMAGENET1K_V1)
            for param in model.parameters():
                param.requires_grad = False
        else:
            model = tv.models.vgg19()
        n_last_in = model.classifier[0].in_features
        model.classifier = nn.Sequential(
               nn.Linear(n_last_in, 512),
               nn.ReLU(inplace=True),
               nn.Linear(512, self.n_classes),
               nn.Sigmoid()
               )
        return model

    def forward(self, x):
        return self.model(x)


class PretrainedEmbedding(nn.Module):
    def __init__(self, backbone):
        super(PretrainedEmbedding, self).__init__()
        # assert n_layer in [18, 34, 50, 101], "no backbone."
        # self.n_layer = n_layer
        backbones = {'resnet18': self.get_resnet18, 
                     'resnet50': self.get_resnet50, 
                     'vgg11': self.get_vgg11, 
                     'vgg19': self.get_vgg19,
                     'vit16': self.get_vit16}
        assert backbone in backbones, "choose valid backbone"
        self.backbone = backbone
        model = backbones[backbone]()
        if 'vit' not in backbone:
            layers = list(model.children())[:-1]
            self.model = nn.Sequential(*layers)
            summary(self.model.to('cuda'), (3, 224, 224))
        else:
            model.heads = torch.nn.Identity()
            self.model = model
            self.model.to('cuda')
            # print(self.model)

    def get_vit16(self):
        model = tv.models.vit_b_16(weights=tv.models.ViT_B_16_Weights.IMAGENET1K_V1)
        return model


    def get_resnet18(self):
        model = tv.models.resnet18(weights=tv.models.ResNet18_Weights.IMAGENET1K_V1)
        return model
    
    def get_resnet50(self):
        model = tv.models.resnet50(weights=tv.models.ResNet50_Weights.IMAGENET1K_V1)
        return model
    

    def get_vgg11(self):
        model = tv.models.vgg11(weights=tv.models.VGG11_Weights.IMAGENET1K_V1)
        return model
    
    def get_vgg19(self):
        model = tv.models.vgg19(weights=tv.models.VGG19_Weights.IMAGENET1K_V1)
        return model 
    
    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    # extractor = PretrainedEmbedding(backbone='resnet18')
    extractor = PretrainedEmbedding(backbone='vit16')
    batch_size = 32
    img_size = (3, 256, 256) # pytorch channel first

    imgs = torch.rand([batch_size, *img_size]).cuda()
    processor = tv.models.ViT_B_16_Weights.IMAGENET1K_V1.transforms()
    imgs = processor(imgs)
    # print(imgs.shape)
    with torch.no_grad():
        features = extractor(imgs)
    print(features.shape)


    