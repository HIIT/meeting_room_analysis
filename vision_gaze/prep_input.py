'''
  Copyright (c) 2016 University of Helsinki
  Permission is hereby granted, free of charge, to any person
  obtaining a copy of this software and associated documentation files
  (the "Software"), to deal in the Software without restriction,
  including without limitation the rights to use, copy, modify, merge,
  publish, distribute, sublicense, and/or sell copies of the Software,
  and to permit persons to whom the Software is furnished to do so,
  subject to the following conditions:
  The above copyright notice and this permission notice shall be
  included in all copies or substantial portions of the Software.
  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
  EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
  MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
  NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
  BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
  ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
  CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
  SOFTWARE.
'''

import scipy
import numpy as np
from scipy import io, misc

class input_gazenet(object):
    def __init__(self, input_image, head_boxes):
        self.head_boxes = np.copy(head_boxes)
        self.net_input_shape = [227, 227]

        # load mean of images
        self.places_mean = io.loadmat('model/places_mean_resize.mat')
        self.imagenet_mean = io.loadmat('model/imagenet_mean_resize.mat')
        self.places_mean = self.places_mean['image_mean']
        self.imagenet_mean = self.imagenet_mean['image_mean']

        # 3 necessary inputs for gaze net
        self.input_image = input_image
        self.input_image_resize = None
        self.eye_images_resize = []
        self.eyes_grids_flat = []

        # preparing intact image
        self.prep_whole_image()
        # preparing image cropped from area around eye's location
        self.prep_eyes_image()
        # preparing (flattened) eye's grid map
        self.prep_eyes_grids()

        self.fit_shape_of_inputs()

    def fit_shape_of_inputs(self):
        # orginal shape: (height, width, channel), new shape: (n_batch, channel, height, width)
        self.input_image_resize = self.input_image_resize.reshape([self.input_image_resize.shape[0], \
                                                                   self.input_image_resize.shape[1], \
                                                                   self.input_image_resize.shape[2], \
                                                                   1])
        self.input_image_resize = self.input_image_resize.transpose(3, 2, 0, 1)

        eye_images_reshape = []
        for eye_image_resize in self.eye_images_resize:
            eye_image_resize = eye_image_resize.reshape([eye_image_resize.shape[0], \
                                                    eye_image_resize.shape[1], \
                                                    eye_image_resize.shape[2], 1])
            eye_image_resize = eye_image_resize.transpose(3, 2, 0, 1)
            eye_images_reshape.append(eye_image_resize)

        self.eye_images_resize = np.copy(eye_images_reshape)

    def prep_whole_image(self):
        #self.input_image = scipy.misc.imread(self.image_path)
        #self.input_image = self.input_image[120::, 220: 1000, ::]

        self.input_image_resize = scipy.misc.imresize(self.input_image, self.net_input_shape, interp='bicubic')
        self.input_image_resize = self.input_image_resize - self.places_mean

    def prep_eyes_image(self):

        for head_box in self.head_boxes:
            top_left_x = int(head_box[0])
            top_left_y = int(head_box[1])
            bottom_right_x = int(head_box[2])
            bottom_right_y = int(head_box[3])

            eye_image = self.input_image[top_left_y:bottom_right_y, top_left_x:bottom_right_x, :]
            eye_image_resize = scipy.misc.imresize(eye_image, self.net_input_shape, interp='bicubic')
            eye_image_resize = eye_image_resize - self.imagenet_mean
            self.eye_images_resize.append(eye_image_resize)

    def prep_eyes_grids(self):
        w = self.input_image.shape[1]
        h = self.input_image.shape[0]

        for head_box in self.head_boxes:
            head_box_scaled = np.copy(head_box)
            head_box_scaled[0] = head_box_scaled[0] / w # x-axis
            head_box_scaled[2] = head_box_scaled[2] / w # x-axis
            head_box_scaled[1] = head_box_scaled[1] / h # y-axis
            head_box_scaled[3] = head_box_scaled[3] / h # y-axis

            # take eye location as the center of the head box
            eye_loc = [(head_box_scaled[0] + head_box_scaled[2]) * 0.5, (head_box_scaled[1] + head_box_scaled[3]) * 0.5]

            eye_grid_x = np.floor(eye_loc[0] * 12).astype('int')
            eye_grid_y = np.floor(eye_loc[1] * 12).astype('int')
            # original shape [13, 13] -> [169, ] -> right berfore input [1, 169, 1, 1]
            eyes_grid = np.zeros([13, 13]).astype('int')
            eyes_grid[eye_grid_y, eye_grid_x] = 1
            eyes_grid_flat = eyes_grid.flatten()
            eyes_grid_flat = eyes_grid_flat.reshape(1, len(eyes_grid_flat), 1, 1)
            self.eyes_grids_flat.append(eyes_grid_flat)
