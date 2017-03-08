
# coding: utf-8

# In[88]:

"""                                                                                                 
Separate file to keep segmentation paramters in it. E.g,                                            
stopping_criterion = 0.47                                                                           
vessel_probability_threshold = .68                                                                  
dilation_size = 3                                                                                   
minimum_size = 4000
"""
from segmentation_param import *
import numpy as np
import scipy.io as sio
from PIL import Image
import ndparse as ndp


# In[145]:

#segmentation entry point functions
from read_tiff_files import read_tiff_files
from classify_pixel import classify_pixel
import detect_cells
from segment_vessels import segment_vessels
import create_synth_dict


# In[39]:

# Following parameters to be passed in as parameters to this script. 
tiff_files_location = '/home/aaron/data/xray/cb2/cb2_images'
classifier_file = '/home/aaron/data/xray/cb2/lee_harvard_cb_sample.ilp'


# In[42]:

# Read tiff stack files - output of tomopy. 
input_data = read_tiff_files(tiff_files_location)


# In[43]:

print("input_data shape", input_data.shape)


# In[44]:

get_ipython().magic(u'matplotlib inline')


# In[45]:

ndp.plot(input_data, slice=50)


# In[46]:

# Compute cell and vessel probability map.
probability_maps = classify_pixel(input_data, classifier_file, threads=no_of_threads, ram=ram_size) 


# In[50]:

cell_prob_map = probability_maps[:, :, :, 2]


# In[51]:

print("cell_prob_map shape", cell_prob_map.shape)


# In[52]:

ndp.plot(cell_prob_map, slice=50, cmap1='jet')


# In[53]:

vessel_prob_map = probability_maps[:, :, :, 1]


# In[54]:

print("vessel_prob_map shape", vessel_prob_map.shape)


# In[55]:

ndp.plot(vessel_prob_map, slice=50, cmap1='jet')


# In[58]:

# Parallel operation to detect cells to be designed. Hard coded for now - cut 400x400x40 volume.
crop_probability_maps = probability_maps[100:500, 100:500, 20:40]


# In[59]:

print("crop_probability_maps shape", crop_probability_maps.shape)


# In[60]:

crop_cell_prob__map = crop_probability_maps[:, :, :, 2]


# In[61]:

print("crop_cell_prob__map shape", crop_cell_prob__map.shape)


# In[62]:

crop_vessel_prob_map = crop_probability_maps[:, :, :, 1]


# In[63]:

print("crop_vessel_prob_map", crop_vessel_prob_map.shape)


# In[65]:

crop_input_data = input_data[100:500, 100:500, 20:40]


# In[90]:

print("crop_input_data shape", crop_input_data.shape)


# In[146]:

centroids, cell_map = detect_cells.detect_cells(crop_cell_prob__map, cell_probability_threshold, stopping_criterion, initial_template_size, dilation_size, max_no_cells) 


# In[26]:

print(centroids.shape)


# In[27]:

print(cell_map.shape)


# In[28]:

vessel_map = segment_vessels(crop_vessel_prob_map, vessel_probability_threshold, dilation_size, minimum_size) 


# In[29]:

print(vessel_map.shape)


# In[30]:

print("Raw Image Slice")
ndp.plot(crop_input_data, slice=50)


# In[31]:

print("Cell Segmentation")
ndp.plot(crop_input_data, cell_map, slice = 50, alpha = 0.5)


# In[32]:

print("Vessel Segmentation")
ndp.plot(crop_input_data, vessel_map, slice = 50, alpha = 0.5)


# In[33]:

print("Cell Probabilities")
ndp.plot(crop_input_data, crop_cell_prob__map, slice = 50, alpha = 0.5)


# In[34]:

print("Vessel Probabilities")
ndp.plot(crop_input_data, crop_vessel_prob_map, slice = 50, alpha = 0.5)


# In[131]:

print(box_radius)

