import torch
import sys
from torch.autograd import Variable
import numpy as np
from options.train_options import TrainOptions
opt = TrainOptions().parse()  # set CUDA_VISIBLE_DEVICES before import torch
from data.data_loader import CreateDataLoader
from models.models import create_model
from skimage import io
from skimage.transform import resize


img_path = 'demo.jpg'

model = create_model(opt)

input_height = 384
input_width  = 512


def test_simple(model):
    total_loss =0 
    toal_count = 0
    print("============================= TEST ============================")
    with torch.no_grad():
        model.switch_to_eval()

        img = np.float32(io.imread(img_path))/255.0
        img = resize(img, (input_height, input_width), order = 1)
        input_img =  torch.from_numpy( np.transpose(img, (2,0,1)) ).contiguous().float()
        input_img = input_img.unsqueeze(0)

        input_images = Variable(input_img.cuda() )
        pred_log_depth = model.netG.forward(input_images) 
        pred_log_depth = torch.squeeze(pred_log_depth)

        pred_depth = torch.exp(pred_log_depth)

        # visualize prediction using inverse depth, so that we don't need sky segmentation (if you want to use RGB map for visualization, \
        # you have to run semantic segmentation to mask the sky first since the depth of sky is random from CNN)
        pred_inv_depth = 1/pred_depth
        pred_inv_depth = pred_inv_depth.data.cpu().numpy()
        # you might also use percentile for better visualization
        pred_inv_depth = pred_inv_depth/np.amax(pred_inv_depth)

        io.imsave('demo.png', pred_inv_depth)
        # print(pred_inv_depth.shape)
        sys.exit()



test_simple(model)
print("We are done")
