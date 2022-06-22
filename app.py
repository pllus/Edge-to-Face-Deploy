#from ctypes.wintypes import RGB
from turtle import width
import streamlit as st
#import base64
import cv2
from streamlit_drawable_canvas import st_canvas
#import json
#import os
#import re
#import time
#import uuid
from io import BytesIO
from pathlib import Path    
import glob
#import PIL
#from svgpathtools import parse_path
import numpy as np
import os
import PIL
from PIL import Image
import torch
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2 
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import torch.optim as optim


# Specify canvas parameters in application
drawing_mode = st.sidebar.selectbox(
    "Drawing tool:", ( "freedraw", "line", "rect", "circle", "transform")
)

stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)    
if drawing_mode == 'point':
    point_display_radius = st.sidebar.slider("Point display radius: ", 1, 25, 3)
stroke_color = st.sidebar.color_picker("Chosse white")
bg_color = hex(000000)
#st.sidebar.color_picker("Background color hex: ", "#eee")
#selected = st.sidebar.selectbox("PredrawnEdges:",("Female","Male"))
#bg_image = 'Predrawnedge/' + selected + '.jpg'
#'Assets/Female.jpg'
#ddj =  st.file_uploader("Background image:", type=["png", "jpg"])
st.sidebar.write('### Save your image as After you finnish Drawing I havent made it work yet')
#Ground = st.sidebar.selectbox("Male or Female:",("Female","Male"))
Ground_Truth = 'Ground_T/Male.jpg'  #Ground_Truth

realtime_update = (True)
#st.sidebar.checkbox("Update in realtime", True)
#Generate = st.button("generate")
option = st.sidebar.radio('', ['Use a Predrawn Edge', 'Use your own Edge'])
valid_images = glob.glob('Predrawnedge/*.jpg')

if option == 'Use a Predrawn Edge':
    st.sidebar.write('### Use a Predrawn Edge')
    fname = st.sidebar.selectbox('',
                                 valid_images)

else:
    st.sidebar.write('### Select an image to upload')
    fname = st.sidebar.file_uploader('',
                                     type=['png', 'jpg', 'jpeg'],
                                     accept_multiple_files=False)

    if fname is None:
        fname = valid_images[0]


fnames = PIL.Image.open(fname)
bg_image = fname
#st.image(fnames,use_column_width=True)
img_array = np.array(fnames)
outimg = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
cv2.imwrite('out.jpg', cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
img = cv2.imread('out.jpg')
imgs = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
img = cv2.imread("out.jpg")


canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color=bg_color,
    background_image=Image.open(bg_image) if bg_image else None,
    update_streamlit=realtime_update,
    height=600,
    width=600,
    drawing_mode=drawing_mode,
    point_display_radius=point_display_radius if drawing_mode == 'point' else 0,
    key="canvas",
)   
st.write("after your done drawing")
st.write("Save it as png or jpg and put it in Use your own edge im trying to figure out how to make it easier)")
Generate = st.button("Generate")
#------------------------------------------------------------------------------------------------#
# image source selection
    # bitwise = st.button("Change form black edge to white edge")
    # if bitwise == True:
    #     img =cv2.bitwise(img)
    
resized = cv2.resize(img,(600,600))
ground_img = cv2.imread(Ground_Truth)
ground_image = cv2.resize(ground_img,(600,600))

torch.backends.cudnn.benchmark = True

class config(): #for configing images and getting model path
  DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
  #LEARNING_RATE = 2e-4
  VAL_DIR = "Input"
  LOAD_MODEL = True
  CHECKPOINT_GEN = "gen.pth.tar" 
  
  both_transform = A.Compose(
    [A.Resize(width=256, height=256),], additional_targets={"image0": "image"},
  )

  transform_only_input = A.Compose(
      [
          A.ColorJitter(p=0.2),
          A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
         ToTensorV2(),
     ]
  )

  transform_only_mask = A.Compose(
      [
          A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
          ToTensorV2(),
      ]
  )

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, act="relu", use_dropout=False):
        super(Block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False, padding_mode="reflect")
            if down
            else nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU() if act == "relu" else nn.LeakyReLU(0.2),
        )

        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)
        self.down = down

    def forward(self, x):
        x = self.conv(x)
        return self.dropout(x) if self.use_dropout else x

class Generator(nn.Module):
    
    def __init__(self, in_channels=3, features=64):
        super().__init__()
        self.initial_down = nn.Sequential(
            nn.Conv2d(in_channels, features, 4, 2, 1, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
        )
        self.down1 = Block(features, features * 2, down=True, act="leaky", use_dropout=False)
        self.down2 = Block(
            features * 2, features * 4, down=True, act="leaky", use_dropout=False
        )
        self.down3 = Block(
            features * 4, features * 8, down=True, act="leaky", use_dropout=False
        )
        self.down4 = Block(
            features * 8, features * 8, down=True, act="leaky", use_dropout=False
        )
        self.down5 = Block(
            features * 8, features * 8, down=True, act="leaky", use_dropout=False
        )
        self.down6 = Block(
            features * 8, features * 8, down=True, act="leaky", use_dropout=False
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features * 8, features * 8, 4, 2, 1), nn.ReLU()
        )

        self.up1 = Block(features * 8, features * 8, down=False, act="relu", use_dropout=True)
        self.up2 = Block(
            features * 8 * 2, features * 8, down=False, act="relu", use_dropout=True
        )
        self.up3 = Block(
            features * 8 * 2, features * 8, down=False, act="relu", use_dropout=True
        )
        self.up4 = Block(
            features * 8 * 2, features * 8, down=False, act="relu", use_dropout=False
        )
        self.up5 = Block(
            features * 8 * 2, features * 4, down=False, act="relu", use_dropout=False
        )
        self.up6 = Block(
            features * 4 * 2, features * 2, down=False, act="relu", use_dropout=False
        )
        self.up7 = Block(features * 2 * 2, features, down=False, act="relu", use_dropout=False)
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(features * 2, in_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        d1 = self.initial_down(x)
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3)
        d5 = self.down4(d4)
        d6 = self.down5(d5)
        d7 = self.down6(d6)
        bottleneck = self.bottleneck(d7)
        up1 = self.up1(bottleneck)
        up2 = self.up2(torch.cat([up1, d7], 1))
        up3 = self.up3(torch.cat([up2, d6], 1))
        up4 = self.up4(torch.cat([up3, d5], 1))
        up5 = self.up5(torch.cat([up4, d4], 1))
        up6 = self.up6(torch.cat([up5, d3], 1))
        up7 = self.up7(torch.cat([up6, d2], 1))
        return self.final_up(torch.cat([up7, d1], 1))


class FaceDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.list_files = os.listdir(self.root_dir)

    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, index):
        img_file = self.list_files[index]
        img_path = os.path.join(self.root_dir, img_file)
        image = np.array(Image.open(img_path))
        input_image = image[:, :600, :]
        target_image = image[:, 600:, :]

        augmentations = config.both_transform(image=input_image, image0=target_image)
        input_image = augmentations["image"]
        target_image = augmentations["image0"]

        input_image = config.transform_only_input(image=input_image)["image"]
        target_image = config.transform_only_mask(image=target_image)["image"]

        return input_image, target_image


gen = Generator(in_channels=3, features=64).to(config.DEVICE)

def load_checkpoint(checkpoint_file, model):#, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
   # optimizer.load_state_dict(checkpoint["optimizer"])

opt_gen = optim.Adam(gen.parameters())# lr=config.LEARNING_RATE, betas=(0.5, 0.999))
if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN, gen, #opt_gen#, config.LEARNING_RATE,
        )

def save_some_examples(gen, val_loader, folder):
    x, y = next(iter(val_loader))
    x, y = x.to(config.DEVICE), y.to(config.DEVICE)
    gen.eval()
    with torch.no_grad():
        y_fake = gen(x)
        y_fake = y_fake * 0.5 + 0.5  # remove normalization#
        save_image(y_fake, folder + f"/y_gen.png")
        save_image(x * 0.5 + 0.5, folder + f"/input.png")

if Generate == True:
    final_input = np.hstack((resized,ground_image))
    cv2.imwrite("Input\"+str("finalinout")+".jpg",final_input)
    val_dataset = FaceDataset(root_dir=config.VAL_DIR)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    save_some_examples(gen, val_loader,folder="Output")
    imageout = Image.open('Output\y_gen.png')
    st.image(imageout, caption='Behold you Masterpiece or nightmare Idk')
