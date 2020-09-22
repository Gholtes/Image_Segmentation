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

print("Packages Loaded...")

def getClassName(classID, cats):
    for i in range(len(cats)):
        if cats[i]['id']==classID:
            return cats[i]['name']
    return "None"


def save_matrix_img(array, filename):
	io.imsave(filename, array)


def make_mask_COCO(img, anns, save_path, cats, coco, filterClasses, binary = False):
	if binary:
		mask = np.zeros((img['height'],img['width']))
		
		for i in range(len(anns)):
			mask = np.maximum(coco.annToMask(anns[i]), mask)
	
	else:
		mask = np.zeros((img['height'],img['width']))
		
		for i in range(len(anns)):
		    className = getClassName(anns[i]['category_id'], cats)
		    pixel_value = filterClasses.index(className)+1
		    mask = np.maximum(coco.annToMask(anns[i])*pixel_value, mask)

	save_matrix_img(mask, save_path)

def load_img_COCO(img_Id, 
				  coco, dataDir, dataType, catIds, cats):
	'''Returns the image, image meta data, image file name, annotations, list of categories present'''
	img = coco.loadImgs(img_Id)[0]

	img_file_name = img['file_name']
	
	im = io.imread('{}/images/{}/{}'.format(dataDir,dataType,img['file_name']))/255.0
	
	annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
	
	anns = coco.loadAnns(annIds)
	
	categories_present = [getClassName(x["category_id"],cats) for x in anns] #get the english name of the cats present

	return im, img, img_file_name, anns, categories_present

def make_all_masks(filterClasses, union = False, binary = False, dataDir='./COCOdata', dataType='val2017'):
	'''makes masks for all COCO images with annotations containing filterClasses. if union is set to True, then
	only images where all classes are present are used'''
	#Init
	annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)
	coco=COCO(annFile)
	# Load the categories in a variable
	catIDs = coco.getCatIds()
	cats = coco.loadCats(catIDs)
	#get annotation files
	annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)

	if union:
		# Fetch class IDs only corresponding to the filterClasses
		catIds = coco.getCatIds(catNms=filterClasses) 
		# # Get all images containing the above Category IDs
		imgIds = coco.getImgIds(catIds=catIds)

		print("Number of images containing all the  classes:", len(imgIds))
	else:
		catIds = []
		for class_name in filterClasses:
			catIds_i = coco.getCatIds(catNms=[class_name]) 
			#add new images (unique) to all_cat_ids
			for c in catIds_i:
				if c not in catIds:
					catIds.append(c)
		# # Get all images containing the above Category IDs
		imgIds = coco.getImgIds(catIds=catIds)

		print("Number of images containing all the  classes:", len(imgIds))


	for img_Id in imgIds:
		print("Processing: {0}".format(img_Id))
		im, img_data, img_file_name, anns, categories_present = load_img_COCO(img_Id, coco, dataDir, dataType, catIds, cats)
		save_path = dataDir + "/images/" + dataType + "Masks/" + img_file_name
		make_mask_COCO(img_data, anns, save_path, cats, coco, filterClasses, binary = binary)

make_all_masks(["car", "bus", "bicycle"], union = True, binary = False)

# dataDir='./COCOdata'
# dataType='val2017'
# annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)

# coco=COCO(annFile)

# # Load the categories in a variable
# catIDs = coco.getCatIds()
# cats = coco.loadCats(catIDs)

# # Define the classes (out of the 81) which you want to see. Others will not be shown.
# filterClasses = ['car', 'bus']

# # Fetch class IDs only corresponding to the filterClasses
# catIds = coco.getCatIds(catNms=filterClasses) 
# print(len(catIDs))

# catIds = []
# for class_name in filterClasses:
# 	catIds_i = coco.getCatIds(catNms=[class_name]) 
# 	#add new images (unique) to all_cat_ids
# 	for c in catIds_i:
# 		if c not in catIds:
# 			catIds.append(c)

# # # Get all images containing the above Category IDs
# imgIds = coco.getImgIds(catIds=catIds)
# print("Number of images containing all the  classes:", len(imgIds))

# # load and display a random image
# img = coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]
# I = io.imread('{}/images/{}/{}'.format(dataDir,dataType,img['file_name']))/255.0

# plt.axis('off')
# plt.imshow(I)
# plt.show()
# plt.axis('off')

# #get 
# annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
# anns = coco.loadAnns(annIds)
# categoriesPresent = [getClassName(x["category_id"],cats) for x in anns] #get the english name of the cats present
# print(categoriesPresent)
# # coco.showAnns(anns)
# # plt.show()

# # # #### GENERATE A SEGMENTATION MASK ####
# mask = np.zeros((img['height'],img['width']))
# for i in range(len(anns)):
#     className = getClassName(anns[i]['category_id'], cats)
#     pixel_value = filterClasses.index(className)+1
#     mask = np.maximum(coco.annToMask(anns[i])*pixel_value, mask)
# plt.imshow(mask)
# plt.show()
# save_matrix_img(mask, "test.jpg")

# # #### GENERATE A BINARY MASK ####
# mask = np.zeros((img['height'],img['width']))
# for i in range(len(anns)):
#     mask = np.maximum(coco.annToMask(anns[i]), mask)
# plt.imshow(mask)
# plt.show()