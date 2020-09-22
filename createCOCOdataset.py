'''
Grant Holtes 2020

Used to make target pixel maps for segmentaion from COCO images

Sources:
https://towardsdatascience.com/master-the-coco-dataset-for-semantic-image-segmentation-part-1-of-2-732712631047

'''

from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import random
import os
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# For visualizing the outputs
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# dataDir='./COCOdataset2017'
# dataType='val'
# annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)

# # Define the classes (out of the 81) which you want to see. Others will not be shown.
# filterClasses = ['laptop', 'tv', 'cell phone']

# # Fetch class IDs only corresponding to the filterClasses
# catIds = coco.getCatIds(catNms=filterClasses) 
# # Get all images containing the above Category IDs
# imgIds = coco.getImgIds(catIds=catIds)
# print("Number of images containing all the  classes:", len(imgIds))

# # load and display a random image
# img = coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]
# I = io.imread('{}/images/{}/{}'.format(dataDir,dataType,img['file_name']))/255.0

# plt.axis('off')
# plt.imshow(I)
# plt.show()


# #### GENERATE A SEGMENTATION MASK ####
# filterClasses = ['laptop', 'tv', 'cell phone']
# mask = np.zeros((img['height'],img['width']))
# for i in range(len(anns)):
#     className = getClassName(anns[i]['category_id'], cats)
#     pixel_value = filterClasses.index(className)+1
#     mask = np.maximum(coco.annToMask(anns[i])*pixel_value, mask)
# plt.imshow(mask)

# #### GENERATE A BINARY MASK ####
# mask = np.zeros((img['height'],img['width']))
# for i in range(len(anns)):
#     mask = np.maximum(coco.annToMask(anns[i]), mask)
# plt.imshow(mask)