import os
import cv2
import numpy as np
from prep_input import input_gazenet
from scipy import misc
import apollocaffe
from apollocaffe import layers
import matplotlib.pyplot as plt
import scipy
from scipy import misc

def alpha_exponentiate(x, alpha=1.2):
    return np.exp(alpha * x) / np.sum(np.exp(alpha*x.flatten()))

def ind2sub(array_shape, ind):
    row = (ind.astype('int') / array_shape[1])
    col = ind % array_shape[1]
    return (row, col)

class GazeFollowNet(object):
    # head_boxes: [num of heads, [x1, y1, x2, y2]]
    def __init__(self, input_image, head_boxes, net=None):
        self.net = apollocaffe.ApolloNet()

        self.head_boxes = np.copy(head_boxes)
        self.output_ready = False

        self.final_maps = []
        self.predictions = []
        # preparing inputs
        self.inputs = input_gazenet(input_image, self.head_boxes)

        # define network structure
        # input with some dummy data first, it doesn't matter the result
        self.forward(img=self.inputs.input_image_resize, \
                     eye_img=self.inputs.eye_images_resize[0], \
                     eye_grid=self.inputs.eyes_grids_flat[0])
        # init the net with parameters trained in binary_w.caffemodel
        self.net.load(self.weights_file())

    def weights_file(self):
        filename = 'model/binary_w.caffemodel'
        if not os.path.exists(filename):
            raise OSError('%s not found!' % filename)
        return filename

    def result_viz(self, interp='bicubic', person_specific=False):
        figs = []
        if person_specific:
            for idx, (head_box, gaze_prediction, final_map) in enumerate(zip(self.head_boxes, self.predictions, self.final_maps), 1):
                if self.inputs.input_image.shape[1] > 1600:
                    fig = plt.figure(idx, figsize=(self.inputs.input_image.shape[1] / 1000., self.inputs.input_image.shape[0] / 1000.), dpi=100)
                else:
                    fig = plt.figure(idx, dpi=100)
                image_with_frame = np.copy(self.inputs.input_image)

                top_left_x = int(head_box[0])
                top_left_y = int(head_box[1])
                bottom_right_x = int(head_box[2])
                bottom_right_y = int(head_box[3])

                thickness, radius = 3, 10
                if self.inputs.input_image.shape[1] > 1600:
                    thickness, radius = 10, 30

                # draw head box
                cv2.rectangle(image_with_frame, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), color=(20, 200, 20), thickness=3)
                # draw predicted gaze coordinate
                cv2.circle(image_with_frame, center=(gaze_prediction[1], gaze_prediction[0]), color=(200, 20, 20), radius=10, thickness=-1)
                # draw line connecting center of the head box and the predicted gaze coordinate
                cv2.line(image_with_frame, pt1=((top_left_x+bottom_right_x+1)>>1, (top_left_y+bottom_right_y)>>1), \
                            pt2=(gaze_prediction[1], gaze_prediction[0]), color=(20, 200, 20), thickness=3)


                plt.imshow(image_with_frame)
                plt.hold(True)
                plt.imshow(final_map, alpha=0.35)
                plt.axis('off')
                for axis in fig.axes:
                    axis.get_xaxis().set_visible(False)
                    axis.get_yaxis().set_visible(False)
                plt.tight_layout(pad=0.)
                figs.append(fig)

        else:
            image_with_frame = np.copy(self.inputs.input_image)
            if self.inputs.input_image.shape[1] > 1600:
                fig = plt.figure(1, figsize=(self.inputs.input_image.shape[1] / 1000., self.inputs.input_image.shape[0] / 1000.), dpi=100)
            else:
                fig = plt.figure(1, dpi=100)

            # element-wise maximum computation for fusing the result from those 5 gaze maps
            final_map = self.final_maps[0]
            for map in self.final_maps[1::]:
                final_map = np.fmax(map, final_map)

            for head_box, gaze_prediction in zip(self.head_boxes, self.predictions):
                top_left_x = int(head_box[0])
                top_left_y = int(head_box[1])
                bottom_right_x = int(head_box[2])
                bottom_right_y = int(head_box[3])

                thickness, radius = 3, 10
                if self.inputs.input_image.shape[1] > 1600:
                    thickness, radius = 10, 30

                # draw head box
                cv2.rectangle(image_with_frame, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), color=(20, 200, 20), thickness=thickness)
                # draw predicted gaze coordinate
                cv2.circle(image_with_frame, center=(gaze_prediction[1], gaze_prediction[0]), color=(220, 220, 255), radius=radius, thickness=-1)
                # draw line connecting center of the head box and the predicted gaze coordinate
                cv2.line(image_with_frame, pt1=((top_left_x+bottom_right_x+1)>>1, (top_left_y+bottom_right_y)>>1), \
                            pt2=(gaze_prediction[1], gaze_prediction[0]), color=(20, 200, 20), thickness=thickness)

            plt.imshow(image_with_frame)
            plt.hold(True)
            plt.imshow(final_map, alpha=0.35)
            plt.axis('off')
            for axis in fig.axes:
                axis.get_xaxis().set_visible(False)
                axis.get_yaxis().set_visible(False)

            plt.tight_layout(pad=0.)
            figs.append(fig)

        return figs

    def run(self):
        for eye_image_resize, eyes_grid_flat in zip(self.inputs.eye_images_resize, self.inputs.eyes_grids_flat):
            self.forward(self.inputs.input_image_resize, eye_image_resize, eyes_grid_flat)
            self.output_ready = True
            final_map = self.process_output()

            flatten_final_map = final_map.flatten()

            max_idx = np.argmax(flatten_final_map)
            flatten_final_map[max_idx] = 0
            self.predictions.append(ind2sub(final_map.shape, max_idx))
            self.final_maps.append(final_map)

        '''return self.net.blobs["fc_0_0_reshape"].data, \
                self.net.blobs["fc_1_0_reshape"].data, \
                self.net.blobs["fc_m1_0_reshape"].data, \
                self.net.blobs["fc_0_1_reshape"].data, \
                self.net.blobs["fc_0_m1_reshape"].data'''

    def shifted_mapping(self, x, delta_x, is_topleft_corner):
        if is_topleft_corner:
            if x == 0:
                return 0
            ix = 0 + 3 * x - delta_x
            return max(ix, 0)
        else:
            if x == 4:
                return 14
            ix = 3 * (x + 1) - 1 - delta_x
            return min(14, ix)

    def process_output(self):
        if not self.output_ready:
            return

        f_0_0 = np.copy(self.net.blobs["fc_0_0_reshape"].data)
        f_1_0 = np.copy(self.net.blobs["fc_1_0_reshape"].data)
        f_m1_0 = np.copy(self.net.blobs["fc_m1_0_reshape"].data)
        f_0_1 = np.copy(self.net.blobs["fc_0_1_reshape"].data)
        f_0_m1 = np.copy(self.net.blobs["fc_0_m1_reshape"].data)

        gaze_grid_list = [alpha_exponentiate(f_0_0), \
                          alpha_exponentiate(f_1_0), \
                          alpha_exponentiate(f_m1_0), \
                          alpha_exponentiate(f_0_1), \
                          alpha_exponentiate(f_0_m1)]

        shifted_x = [0, 1, -1, 0, 0]
        shifted_y = [0, 0, 0, -1, 1]

        count_map = np.ones([15, 15])
        average_map = np.zeros([15, 15])
        for delta_x, delta_y, gaze_grids in zip(shifted_x, shifted_y, gaze_grid_list):
            for x in range(0, 5):
                for y in range(0, 5):
                    ix = self.shifted_mapping(x, delta_x, True)
                    iy = self.shifted_mapping(y, delta_y, True)
                    fx = self.shifted_mapping(x, delta_x, False)
                    fy = self.shifted_mapping(y, delta_y, False)

                    average_map[ix:fx+1, iy:fy+1] += gaze_grids[x, y]
                    count_map[ix:fx+1, iy:fy+1] += 1

        average_map = average_map / count_map
        final_map = misc.imresize(average_map, self.inputs.input_image.shape, interp='bicubic')

        return final_map

    def forward(self, img, eye_img, eye_grid):
        # inputs
        self.net.clear_forward()
        self.net.f(layers.NumpyData("data", data=img))
        self.net.f(layers.NumpyData("face", data=eye_img))
        self.net.f(layers.NumpyData("eyes_grid", data=eye_grid))

        for layer in self.net_proto():
            self.net.f(layer)
            '''print layer.p.name
            if layer.p.name == 'conv5_red':
                print 'conv5 blob shape = ', self.net.blobs["conv5"].data.shape
                print 'conv5 param shape = ', self.net.params['conv5.p0'].data.shape
                print 'conv5_red blob shape = ', self.net.blobs["conv5_red"].data.shape
                print 'conv5_red param shape = ', self.net.params['conv5_red.p0'].data.shape'''

    def net_proto(self):
        conv_weight_filler = layers.Filler("gaussian", 0.01)
        bias_filler0 = layers.Filler("constant", 0.0)
        bias_filler1 = layers.Filler("constant", 0.1)
        bias_filler5 = layers.Filler("constant", 0.5)

        # same deploy structure as in deploy_demo.prototxt
        net_layers = [
            # saliency path
            layers.Convolution("conv1", bottoms=["data"], param_lr_mults=[0, 0],
                param_decay_mults=[1, 0], kernel_dim=(11, 11),
                stride=4, weight_filler=conv_weight_filler, bias_filler=bias_filler0, num_output=96),
            layers.ReLU(name="relu1", bottoms=["conv1"], tops=["conv1"]),
            layers.Pooling(name="pool1", bottoms=["conv1"], kernel_size=3, stride=2),
            layers.LRN(name="norm1", bottoms=["pool1"], tops=["norm1"], local_size=5, alpha=0.0001, beta=0.75),

            layers.Convolution(name="conv2", bottoms=["norm1"], param_lr_mults=[0, 0],
                param_decay_mults=[1, 0], kernel_dim=(5, 5),
                pad=2, group=2, weight_filler=conv_weight_filler, bias_filler=bias_filler1, num_output=256),
            layers.ReLU(name="relu2", bottoms=["conv2"], tops=["conv2"]),
            layers.Pooling(name="pool2", bottoms=["conv2"], kernel_size=3, stride=2),
            layers.LRN(name="norm2", bottoms=["pool2"], tops=["norm2"], local_size=5, alpha=0.0001, beta=0.75),

            layers.Convolution(name="conv3", bottoms=["norm2"], param_lr_mults=[0.1, 0.2],
                param_decay_mults=[1, 0], kernel_dim=(3, 3),
                pad=1, weight_filler=conv_weight_filler, bias_filler=bias_filler0, num_output=384),
            layers.ReLU(name="relu3", bottoms=["conv3"], tops=["conv3"]),

            layers.Convolution(name="conv4", bottoms=["conv3"], param_lr_mults=[0.1, 0.2],
                param_decay_mults=[1, 0], kernel_dim=(3, 3),
                pad=1, group=2, weight_filler=conv_weight_filler, bias_filler=bias_filler1, num_output=384),
            layers.ReLU(name="relu4", bottoms=["conv4"], tops=["conv4"]),

            layers.Convolution(name="conv5", bottoms=["conv4"], param_lr_mults=[0.1, 0.2],
                param_decay_mults=[1, 0], kernel_dim=(3, 3),
                pad=1, group=2, weight_filler=conv_weight_filler, bias_filler=bias_filler1, num_output=256),
            layers.ReLU(name="relu5", bottoms=["conv5"], tops=["conv5"]),

            layers.Convolution(name="conv5_red", bottoms=["conv5"], param_lr_mults=[1.0, 2.0],
                param_decay_mults=[1, 0], kernel_dim=(1, 1),
                weight_filler=conv_weight_filler, bias_filler=bias_filler1, num_output=1),
            layers.ReLU(name="relu5_red", bottoms=["conv5_red"], tops=["conv5_red"]),

            # gaze path
            layers.Convolution("conv1_face", bottoms=["face"], param_lr_mults=[0, 0],
                param_decay_mults=[1, 0], kernel_dim=(11, 11),
                stride=4, weight_filler=conv_weight_filler, bias_filler=bias_filler0, num_output=96),
            layers.ReLU(name="relu1_face", bottoms=["conv1_face"], tops=["conv1_face"]),
            layers.Pooling(name="pool1_face", bottoms=["conv1_face"], kernel_size=3, stride=2),
            layers.LRN(name="norm1_face", bottoms=["pool1_face"], tops=["norm1_face"], local_size=5, alpha=0.0001, beta=0.75),

            layers.Convolution(name="conv2_face", bottoms=["norm1_face"], param_lr_mults=[0, 0],
                param_decay_mults=[1, 0], kernel_dim=(5, 5),
                pad=2, group=2, weight_filler=conv_weight_filler, bias_filler=bias_filler1, num_output=256),
            layers.ReLU(name="relu2_face", bottoms=["conv2_face"], tops=["conv2_face"]),
            layers.Pooling(name="pool2_face", bottoms=["conv2_face"], kernel_size=3, stride=2),
            layers.LRN(name="norm2_face", bottoms=["pool2_face"], tops=["norm2_face"], local_size=5, alpha=0.0001, beta=0.75),

            layers.Convolution(name="conv3_face", bottoms=["norm2_face"], param_lr_mults=[0.2, 0.4],
                param_decay_mults=[1, 0], kernel_dim=(3, 3),
                pad=1, weight_filler=conv_weight_filler, bias_filler=bias_filler0, num_output=384),
            layers.ReLU(name="relu3_face", bottoms=["conv3_face"], tops=["conv3_face"]),

            layers.Convolution(name="conv4_face", bottoms=["conv3_face"], param_lr_mults=[0.2, 0.4],
                param_decay_mults=[1, 0], kernel_dim=(3, 3),
                pad=1, group=2, weight_filler=conv_weight_filler, bias_filler=bias_filler1, num_output=384),
            layers.ReLU(name="relu4_face", bottoms=["conv4_face"], tops=["conv4_face"]),

            layers.Convolution(name="conv5_face", bottoms=["conv4_face"], param_lr_mults=[0.2, 0.4],
                param_decay_mults=[1, 0], kernel_dim=(3, 3),
                pad=1, group=2, weight_filler=conv_weight_filler, bias_filler=bias_filler1, num_output=256),
            layers.ReLU(name="relu5_face", bottoms=["conv5_face"], tops=["conv5_face"]),
            layers.Pooling(name="pool5_face", bottoms=["conv5_face"], kernel_size=3, stride=2),

            layers.InnerProduct(name="fc6_face", bottoms=["pool5_face"], tops=["fc6_face"], param_lr_mults=[1, 2],
                param_decay_mults=[1, 0],
                weight_filler=layers.Filler("gaussian", 0.5),
                bias_filler=bias_filler5, num_output=500),
            layers.ReLU(name="relu6_face", bottoms=["fc6_face"], tops=["fc6_face"]),

            layers.Flatten(name="eyes_grid_flat", bottoms=["eyes_grid"], tops=["eyes_grid_flat"]),
            layers.Power(name="eyes_grid_mult", bottoms=["eyes_grid_flat"], tops=["eyes_grid_mult"],
                         power=1, scale=24, shift=0),
            layers.Concat(name="face_input", bottoms=["fc6_face", "eyes_grid_mult"], tops=["face_input"], axis=1),

            layers.InnerProduct(name="fc7_face", bottoms=["face_input"], tops=["fc7_face"], param_lr_mults=[1, 2],
                                param_decay_mults=[1, 0], weight_filler=layers.Filler("gaussian", 0.01),
                                bias_filler=bias_filler5, num_output=400),
            layers.ReLU(name="relu7_face", bottoms=["fc7_face"], tops=["fc7_face"]),

            layers.InnerProduct(name="fc8_face", bottoms=["fc7_face"], tops=["fc8_face"], param_lr_mults=[1, 2],
                                param_decay_mults=[1, 0], weight_filler=layers.Filler("gaussian", 0.01),
                                bias_filler=bias_filler5, num_output=200),
            layers.ReLU(name="relu8_face", bottoms=["fc8_face"], tops=["fc8_face"]),

            # mismatch with deploy_demo.prototxt!
            layers.InnerProduct(name="importance_no_sigmoid", bottoms=["fc8_face"], tops=["importance_no_sigmoid"], param_lr_mults=[0.2, 0.0],
                                param_decay_mults=[1.0, 0.0],
                                weight_filler=layers.Filler("gaussian", 0.01), num_output=169),

            layers.Sigmoid(name="importance_map_prefilter", bottoms=["importance_no_sigmoid"], tops=["importance_map_prefilter"]),
            layers.Reshape('importance_map_reshape', (1, 1, 13, 13), bottoms=['importance_map_prefilter'], tops=["importance_map_reshape"]),

            layers.Convolution(name="importance_map", bottoms=["importance_map_reshape"], param_lr_mults=[0.0, 0.0],
                param_decay_mults=[1.0, 0.0], kernel_dim=(3, 3),
                pad=1, stride=1, weight_filler=conv_weight_filler, bias_filler=bias_filler1, num_output=1),
            layers.Eltwise(name="fc_7", bottoms=["conv5_red", "importance_map"], tops=["fc_7"], operation="PROD"),

            # shifted grids
            layers.InnerProduct(name="fc_0_0", bottoms=["fc_7"], tops=["fc_0_0"], param_lr_mults=[1, 2],
                                param_decay_mults=[1, 0], weight_filler=layers.Filler("gaussian", 0.01),
                                bias_filler=bias_filler0, num_output=25),
            layers.InnerProduct(name="fc_1_0", bottoms=["fc_7"], tops=["fc_1_0"], param_lr_mults=[1, 2],
                                param_decay_mults=[1, 0], weight_filler=layers.Filler("gaussian", 0.01),
                                bias_filler=bias_filler0, num_output=25),
            layers.InnerProduct(name="fc_m1_0", bottoms=["fc_7"], tops=["fc_m1_0"], param_lr_mults=[1, 2],
                                param_decay_mults=[1, 0], weight_filler=layers.Filler("gaussian", 0.01),
                                bias_filler=bias_filler0, num_output=25),
            layers.InnerProduct(name="fc_0_1", bottoms=["fc_7"], tops=["fc_0_1"], param_lr_mults=[1, 2],
                                param_decay_mults=[1, 0], weight_filler=layers.Filler("gaussian", 0.01),
                                bias_filler=bias_filler0, num_output=25),
            layers.InnerProduct(name="fc_0_m1", bottoms=["fc_7"], tops=["fc_0_m1"], param_lr_mults=[1, 2],
                                param_decay_mults=[1, 0], weight_filler=layers.Filler("gaussian", 0.01),
                                bias_filler=bias_filler0, num_output=25),
            layers.Reshape('fc_0_0_reshape', (5, 5), bottoms=['fc_0_0'], tops=["fc_0_0_reshape"]),
            layers.Reshape('fc_1_0_reshape', (5, 5), bottoms=['fc_1_0'], tops=["fc_1_0_reshape"]),
            layers.Reshape('fc_m1_0_reshape', (5, 5), bottoms=['fc_m1_0'], tops=["fc_m1_0_reshape"]),
            layers.Reshape('fc_0_1_reshape', (5, 5), bottoms=['fc_0_1'], tops=["fc_0_1_reshape"]),
            layers.Reshape('fc_0_m1_reshape', (5, 5), bottoms=['fc_0_m1'], tops=["fc_0_m1_reshape"])
        ]

        return net_layers
