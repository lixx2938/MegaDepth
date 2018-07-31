# set CUDA_VISIBLE_DEVICES before import torch
import torch
import sys
import os
from torch.autograd import Variable
import numpy as np
from options.train_options import TrainOptions
#from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from skimage import io
from skimage.transform import resize
import pickle

# Example use
# python demo.py --images '/Users/Hallee/Desktop/newdata/'

# required dimensions for the provided model
INPUT_WIDTH = 384
INPUT_HEIGHT = 512

def predict_depth(model, img_path):

    total_loss = 0
    toal_count = 0
    name = os.path.basename(os.path.normpath(img_path)).split('.')[0]

    with torch.no_grad():

        model.switch_to_eval()

        print("Loading image from..." + img_path)
        img = np.float32(io.imread(img_path))/255.0

        orig_height = img.shape[0]
        orig_width = img.shape[1]
        print("Original Image Dimensions")
        print("Height: ", orig_height)
        print("Width: ", orig_width)

        img = resize(img, (INPUT_HEIGHT, INPUT_WIDTH), order = 1)
        input_img =  torch.from_numpy( np.transpose(img, (2,0,1)) ).contiguous().float()
        input_img = input_img.unsqueeze(0)

        input_images = Variable(input_img)
        #input_images = Variable(input_img.cuda() )
        pred_log_depth = model.netG.forward(input_images)
        pred_log_depth = torch.squeeze(pred_log_depth)

        # convert from log relative depth to relative depth
        pred_depth = torch.exp(pred_log_depth)
        # convert from tensor to numpy array
        pred_depth = pred_depth.data.cpu().numpy()

        #saving predicted depth as a pickle
        pickle_path = "./output/" + name + "_depth.p"
        print("Saving depth map to... " +  pickle_path)
        pickle.dump(pred_depth, open(pickle_path, "wb" ) )

        # resize the array to be the original image size
        resized_depth = resize_depth(pred_depth, orig_width, orig_height)

        ## Making inverse depth image

        # Note from MegaDepth authors:
        # visualize prediction using inverse depth, so that we don't need sky
        # segmentation (if you want to use RGB map for visualization, \
        # you have to run semantic segmentation to mask the sky first since the
        # depth of sky is random from CNN)
        pred_inv_depth = 1/resized_depth
        # you might also use percentile for better visualization
        pred_inv_depth = pred_inv_depth/np.amax(pred_inv_depth)

        # saving interse depth image
        io.imsave("./output/" + name + "_inv_depth.png", pred_inv_depth)

        ## Saving depth map
        resized_pred_depth = resize_depth(pred_depth, orig_width, orig_height)
        points = convert_array_to_points(resized_pred_depth)

        # saving point cloud
        save_points(points, name)

def rescale_depth(arr, new_min_depth, new_max_depth):
    """
    summary: rescales depths to be from new_min_depth to new_max_depth
    """
    max_depth = np.amax(arr)
    min_depth = np.amin(arr)
    new_arr = new_min_depth + (new_max_depth/max_depth)*(arr-min_depth)
    return(new_arr)

def resize_depth(pred_depth, orig_width, orig_height):
    """
    summary: resizes the array of depth values
    if we don't rescale the values to be between 0,1 we get a ValueError from
    skimage's resize function
    """
    min_depth = np.amin(pred_depth)
    max_depth = np.amax(pred_depth)
    scaled_depth = rescale_depth(pred_depth, 0, 1)
    resized_depth = resize(scaled_depth, (orig_height, orig_width),
                            anti_aliasing=True, mode='reflect')
    new_depth = rescale_depth(resized_depth, min_depth, max_depth)
    return(new_depth)

def save_points(points, name, path="./output/", label="depth"):
    """
    summary: saves converted points into textfile of (x,y,z) coordinates
    parameters:
        points - (np.array) array with 3 columns
        name - (str) name of image being
    returns: na
    """
    xyz_path = path + name + "_" + label + ".txt"
    print("Saving coordinates to... " +  xyz_path)
    np.savetxt(xyz_path, points,
        fmt=['%.0d','%.0d','%.10f'],
        delimiter=' ',
        comments='', # gets ride of hashtag in header
        header = str(points.shape[0]))
    return()

def convert_array_to_points(points):
    """
    summary: converts m by n array of values to an array of x,y,z coordinates
    notes: originally this function converted to cartesian x,y
           it has since been changed to save coordinate as image coordinates!
    """
    # putting into (x,y,z) format
    m,n = points.shape
    R,C = np.mgrid[:m,:n]
    points = np.column_stack((C.ravel(), R.ravel(), points.ravel()))
    return(points)

if __name__ == "__main__":

    # Loading the model
    #opt = TestOptions().parse()
    opt = TrainOptions().parse()
    model = create_model(opt)
    print("=========================================================")

    # e.g. '/Users/Hallee/Desktop/newdata/'
    mydir = opt.images
    image_list = [mydir + f for f in os.listdir(mydir) if f[0] != "."]
    for i in range(len(image_list)):
        predict_depth(model, image_list[i])
        print("Done with " + os.path.basename(os.path.normpath(image_list[i])).split('.')[0])

    print("Done!")
