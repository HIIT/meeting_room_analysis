import scipy
import numpy as np
from scipy import io, misc

class input_gazenet(object):
    def __init__(self, input_image, head_box):
        self.head_box = np.copy(head_box)
        self.net_input_shape = [227, 227]

        # load mean of images
        self.places_mean = io.loadmat('model/places_mean_resize.mat')
        self.imagenet_mean = io.loadmat('model/imagenet_mean_resize.mat')
        self.places_mean = self.places_mean['image_mean']
        self.imagenet_mean = self.imagenet_mean['image_mean']

        # 3 necessary inputs for gaze net
        self.input_image = input_image
        self.input_image_resize = None
        self.eye_image_resize = None
        self.eyes_grid_flat = None

        # preparing intact image
        self.prep_whole_image()
        # preparing image cropped from area around eye's location
        self.prep_eye_image()
        # preparing (flattened) eye's grid map
        self.prep_eye_grids()

        self.fit_shape_of_inputs()

    def fit_shape_of_inputs(self):
        # orginal shape: (height, width, channel), new shape: (n_batch, channel, height, width)
        self.input_image_resize = self.input_image_resize.reshape([self.input_image_resize.shape[0], \
                                                                   self.input_image_resize.shape[1], \
                                                                   self.input_image_resize.shape[2], \
                                                                   1])
        self.input_image_resize = self.input_image_resize.transpose(3, 2, 0, 1)
        self.eye_image_resize = self.eye_image_resize.reshape([self.eye_image_resize.shape[0], \
                                                               self.eye_image_resize.shape[1], \
                                                               self.eye_image_resize.shape[2], \
                                                               1])
        self.eye_image_resize = self.eye_image_resize.transpose(3, 2, 0, 1)

    def prep_whole_image(self):
        #self.input_image = scipy.misc.imread(self.image_path)
        #self.input_image = self.input_image[120::, 220: 1000, ::]

        self.input_image_resize = scipy.misc.imresize(self.input_image, self.net_input_shape, interp='bilinear')
        self.input_image_resize = self.input_image_resize - self.places_mean

    def prep_eye_image(self):
        top_left_x = int(self.head_box[0, 0])
        top_left_y = int(self.head_box[0, 1])
        bottom_right_x = int(self.head_box[1, 0])
        bottom_right_y = int(self.head_box[1, 1])

        eye_image = self.input_image[top_left_y:bottom_right_y, top_left_x:bottom_right_x, :]
        self.eye_image_resize = scipy.misc.imresize(eye_image, self.net_input_shape, interp='bilinear')
        self.eye_image_resize = self.eye_image_resize - self.imagenet_mean

    def prep_eye_grids(self):
        w = self.input_image.shape[1]
        h = self.input_image.shape[0]

        head_box = np.copy(self.head_box)
        head_box[0:2, 0] = self.head_box[0:2, 0] / w
        head_box[0:2, 1] = self.head_box[0:2, 1] / h
        # take eye location as the center of the head box
        eye_loc = [(head_box[0, 0] + head_box[1, 0]) * 0.5, (head_box[0, 1] + head_box[1, 1]) * 0.5]

        eye_grid_x = np.floor(eye_loc[0] * 12).astype('int')
        eye_grid_y = np.floor(eye_loc[1] * 12).astype('int')
        # original shape [13, 13] -> [169, ] -> right berfore input [1, 169, 1, 1]
        eyes_grid = np.zeros([13, 13]).astype('int')
        eyes_grid[eye_grid_y, eye_grid_x] = 1
        self.eyes_grid_flat = eyes_grid.flatten()
        self.eyes_grid_flat = self.eyes_grid_flat.reshape(1, len(self.eyes_grid_flat), 1, 1)
