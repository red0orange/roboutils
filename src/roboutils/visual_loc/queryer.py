import os
import sys

from PIL import Image
import torch
import torchvision.transforms as T

from my_utils.visual_loc.pad_resize import pad_resize_image


class Queryer:
    def __init__(self):
        pass

    def transform_img(self, img):
        raise NotImplementedError

    def encoder(self, inp):
        raise NotImplementedError


class DINOv2Queryer(Queryer):
    def __init__(self):
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        self.model.cuda()
        self.model.eval()

        self.dino_transform = T.Compose([
            T.ToTensor(),
            T.Resize((224, 224), antialias=True),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

        self.img_size = (224, 224)
        pass

    def transform_img(self, img):
        img, _ = pad_resize_image(img, self.img_size[0], self.img_size[1])
        img = Image.fromarray(img[:, :, ::-1])
        return self.dino_transform(img)

    def encoder(self, inp):
        return self.model(inp)


import unicom
class UnicomQueryer(Queryer):
    def __init__(self):
        model_name = "ViT-B/16"
        self.model, self.transform_clip = unicom.load(model_name)
        self.model = self.model.cuda()
        self.model.eval()
        self.img_size = (320, 320)
        pass

    def transform_img(self, img):
        img, _ = pad_resize_image(img, self.img_size[0], self.img_size[1])
        img = Image.fromarray(img[:, :, ::-1])
        return self.transform_clip(img)

    def encoder(self, inp):
        return self.model(inp)