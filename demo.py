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

        img = resize(img, (INPUT_HEIGHT, INPUT_WIDTH), order = 1)
        input_img =  torch.from_numpy( np.transpose(img, (2,0,1)) ).contiguous().float()
        input_img = input_img.unsqueeze(0)

        input_images = Variable(input_img)
        #input_images = Variable(input_img.cuda() )
        pred_log_depth = model.netG.forward(input_images)
        pred_log_depth = torch.squeeze(pred_log_depth)

        pred_depth = torch.exp(pred_log_depth)

        # saving predicted depth as a pickle
        pickle_path = "./images/" + name + "_depth.p"
        print("Saving depth map to... " +  pickle_path)
        pickle.dump(pred_depth, open(pickle_path, "wb" ) )

        ## Making inverse depth image

        # visualize prediction using inverse depth, so that we don't need sky segmentation (if you want to use RGB map for visualization, \
        # you have to run semantic segmentation to mask the sky first since the depth of sky is random from CNN)
        pred_inv_depth = 1/pred_depth
        pred_inv_depth = pred_inv_depth.data.cpu().numpy()

        # you might also use percentile for better visualization
        pred_inv_depth = pred_inv_depth/np.amax(pred_inv_depth) # convert from tensor to array
        resized_inv_pred_depth = resize(pred_inv_depth, (orig_height, orig_width))

        # saving depth image
        io.imsave("./images/" + name + "_depth.png", resized_inv_pred_depth)

        ## Saving depth map
        pred_depth = pred_depth.data.cpu().numpy() # convert from tensor to array
        resized_pred_depth = resize_depth(pred_depth, orig_height, orig_width)
        #resized_pred_depth = rescale_depth(resized_pred_depth, 255.0)

        points = convert_array_to_points(resized_pred_depth)
        # saving point cloud
        save_points(points, name)
        #sys.exit()

def resize_depth(pred_depth, orig_height, orig_width):
    """
    summary: rescales tensor to be size of orig_height by orig_width
    """
    pred_depth = rescale_depth(pred_depth, 1)
    resized_pred_depth = resize(pred_depth, (orig_height, orig_width))
    return(resized_pred_depth)

def rescale_depth(pred_depth, new_max_depth):
    """
    summary: rescales depths to be from 0 to max_depth
    """
    max_depth = np.amax(pred_depth)
    min_depth = np.amin(pred_depth)
    new_depth_map = new_max_depth*(pred_depth-min_depth)/max_depth
    return(new_depth_map)

def save_points(points, name):
    """
    summary: saves converted points into textfile of (x,y,z) coordinates
    parameters:
        points - (np.array) array with 3 columns
        name - (str) name of image being
    returns: na
    """
    xyz_path = "./images/" + name + "_depth.txt"
    print("Saving depth map coordinates to... " +  xyz_path)
    np.savetxt(xyz_path, points,
        fmt=['%.0d','%.0d','%.10f'],
        delimiter=' ',
        comments='', # gets ride of hashtag in header
        header = str(points.shape[0]))


def convert_array_to_points(points):
    """
    summary: converts m by n array of values to an array of x,y,z coordinates
    """
    # putting into (x,y,z) format
    m,n = points.shape
    R,C = np.mgrid[:m,:n]
    points = np.column_stack((n-C.ravel(),m-R.ravel(), points.ravel()))
    return(points)

if __name__ == "__main__":

    # Loading the model
    #opt = TestOptions().parse()
    opt = TrainOptions().parse()
    model = create_model(opt)
    print("=========================================================")

    mydir = '/Users/Hallee/Desktop/newdata/'
    image_list = [mydir + f for f in os.listdir(mydir)]
    for i in range(len(image_list)):
        predict_depth(model, image_list[i])
        print("Done with " + os.path.basename(os.path.normpath(image_list[i])).split('.')[0])

    print("Done!")
