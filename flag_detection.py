# Imports
import os
import glob
import sys
import torch

import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub

# For drawing onto the image.
import numpy as np
from PIL import Image
from PIL import ImageColor
from PIL import ImageDraw
from PIL import ImageFont
from PIL import ImageOps

# For downloading the image.
import matplotlib.pyplot as plt
import tempfile
from six.moves.urllib.request import urlopen
from six import BytesIO
import pickle

# Commented out IPython magic to ensure Python compatibility.
!git clone https://github.com/IDEA-Research/GroundingDINO.git
# %cd GroundingDINO
!git checkout -q 57535c5a79791cb76e36fdb64975271354f10251
!pip install -q -e .

"""**NOTE:** To glue all the elements of our demo together we will use the [`supervision`](https://github.com/roboflow/supervision) pip package, which will help us **process, filter and visualize our detections as well as to save our dataset**. A lower version of the `supervision` was installed with Grounding DINO. However, in this demo we need the functionality introduced in the latest versions. Therefore, we uninstall the current `supervsion` version and install version `0.6.0`.


"""

!pip uninstall -y supervision
!pip install -q supervision==0.6.0

import supervision as sv
print(sv.__version__)

"""### Download Grounding DINO Model Weights

To run Grounding DINO we need two files - configuration and model weights. The configuration file is part of the [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO) repository, which we have already cloned. The weights file, on the other hand, we need to download. We write the paths to both files to the `GROUNDING_DINO_CONFIG_PATH` and `GROUNDING_DINO_CHECKPOINT_PATH` variables and verify if the paths are correct and the files exist on disk.
"""

import os
GROUNDING_DINO_CONFIG_PATH = os.path.join("groundingdino/config/GroundingDINO_SwinT_OGC.py")
print(GROUNDING_DINO_CONFIG_PATH, "; exist:", os.path.isfile(GROUNDING_DINO_CONFIG_PATH))

# Commented out IPython magic to ensure Python compatibility.
!mkdir -p weights
# %cd weights
!wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

# Commented out IPython magic to ensure Python compatibility.
# %cd /content/GroundingDINO

import os

GROUNDING_DINO_CHECKPOINT_PATH = os.path.join("weights/groundingdino_swint_ogc.pth")
print(GROUNDING_DINO_CHECKPOINT_PATH, "; exist:", os.path.isfile(GROUNDING_DINO_CHECKPOINT_PATH))

"""## Load models"""

import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""### Load Grounding DINO Model"""

from groundingdino.util.inference import Model

grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)

from google.colab import drive
drive.mount('/drive')
os.chdir('/drive/MyDrive/StreetView/Downloads')
cwd = os.getcwd()

sv = os.path.dirname("/drive/MyDrive/StreetView/")
big_towns = pd.read_excel(os.path.join(sv, 'Settlements.xlsx'),sheet_name='Sheet1')
lookup = pd.read_excel(os.path.join(sv, 'Lookup_Table.xlsx') )

"""### Zero-Shot Object Detection with Grounding DINO

**NOTE:** To get better Grounding DINO detection we will leverage a bit of prompt engineering using `enhance_class_name` function defined below. 👇 You can learn more from our [Grounding DINO tutorial](https://blog.roboflow.com/grounding-dino-zero-shot-object-detection/).
"""

from typing import List

def enhance_class_name(class_names: List[str]) -> List[str]:
    return [
        f"all {class_name}s"
        for class_name
        in class_names
    ]

import cv2
import supervision as sv

"""## Full Dataset Mask Auto Annotation"""

i = 49
t=big_towns.Town[i]
path = os.path.join(cwd, t + '/')
list = os.path.join(path,t + "list.pickle")
all_images = pd.read_pickle(list)

""" Split into chunks"""

counter = 0
chunk_size = 100
chunks = [all_images[i:i+chunk_size] for i in range(0, len(all_images), chunk_size)]
chunks = chunks[counter:]

"""Output"""

output_directory = os.path.join(cwd, t,"Results_D")
op = os.makedirs(output_directory, exist_ok=True) # new directory

"""Extract labels from images"""

IMAGES_DIRECTORY = os.path.join(cwd, t )
os.chdir(IMAGES_DIRECTORY )
IMAGES_EXTENSIONS = ['jpg', 'jpeg', 'png']

CLASSES = ['flag']
BOX_TRESHOLD = 0.6
TEXT_TRESHOLD = 0.5

import cv2
from tqdm.notebook import tqdm

images = {}
annotations = {}

for chunk in chunks :
    image_paths = chunk
    A = np.empty(shape=[0, 3])

    for image_path in tqdm(image_paths):
        image_name  = basename = os.path.basename(image_path)
        image_path = str(image_path)
        image = cv2.imread(image_path)

        detections = grounding_dino_model.predict_with_classes(
            image=image,
            classes=enhance_class_name(class_names=CLASSES),
            box_threshold=BOX_TRESHOLD,
            text_threshold=TEXT_TRESHOLD
        )
        detections = detections[(detections.class_id != None)  ]
        images[image_name] = image
        annotations[image_name] = detections

        f = os.path.splitext(image_name)[0]     #  remove ext
        pano  = f[:-4]          # lose camera direction
        newrow = [f, pano, sum(1 for i in detections.class_id if i == 0)]
        A = np.vstack([A, newrow])

    fname = os.path.join(output_directory,t  + str(counter) + "results.pickle")
    pickle_out = open(fname,"wb")
    pickle.dump(A, pickle_out)
    pickle_out.close()
    counter += 1

"""### Save labels in Pascal VOC XML"""

ANNOTATIONS_DIR = os.path.join(output_directory,"annotations")
os.makedirs(ANNOTATIONS_DIR , exist_ok=True) # new directory

import locale
locale.getpreferredencoding = lambda: "UTF-8"

MIN_IMAGE_AREA_PERCENTAGE = 0.002
MAX_IMAGE_AREA_PERCENTAGE = 0.80
APPROXIMATION_PERCENTAGE = 0.75

sv.Dataset(
    classes=CLASSES,
    images=images,
    annotations=annotations
).as_pascal_voc(
    annotations_directory_path=ANNOTATIONS_DIR,
    min_image_area_percentage=MIN_IMAGE_AREA_PERCENTAGE,
    max_image_area_percentage=MAX_IMAGE_AREA_PERCENTAGE,
    approximation_percentage=APPROXIMATION_PERCENTAGE
)

"""# Aggregate results"""

results = np.empty(shape=[0, 3])
for i in range(len(chunks)) :
    A = os.path.join(output_directory,t + str(i) + "resultsD.pickle")
    result = pd.read_pickle(A)
    results = np.vstack([results, result])

"""# Store results"""

fname = os.path.join(path,t  + "resultsD.pickle")
pickle_out = open(fname,"wb")
pickle.dump(results, pickle_out)
pickle_out.close()