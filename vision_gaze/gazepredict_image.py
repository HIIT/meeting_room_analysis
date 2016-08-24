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

import numpy as np
import network
from network import GazeFollowNet
import matplotlib.pyplot as plt
import cv2
import object_tracker.get_points as get_points

def detect(video_path):
    cam = cv2.VideoCapture(video_path)
    # If Camera Device is not opened, exit the program
    if not cam.isOpened():
        print "Video device or file couldn't be opened"
        exit()

    retval, img = cam.read()
    #img = img[100::, 250:1100, ::].astype('uint8')
    # Co-ordinates of objects to be tracked
    # will be stored in a list named `points`
    points = get_points.run(img)

    if not points:
        print "ERROR: No object to be tracked."
        exit()

    return cam, img, points

def gaze_predict(img, head_loc, net=None):
    if net is None:
        net = GazeFollowNet(img, head_loc)
    else:
        net = GazeFollowNet(img, head_loc, net)
    net.run()

    return net

def gaze_prediction_image(image_path, person_specific=False, is_visualize=False):
    img = cv2.imread(image_path)
    points = get_points.run(img)
    if not points:
        print "ERROR: No object to be tracked."
        exit()

    net = gaze_predict(img, points)
    figs = net.result_viz('bicubic', person_specific=person_specific)

    for i, fig in enumerate(figs, 1):
        if is_visualize:
            plt.figure(i, bbox_inches='tight', pad_inches=0)
            plt.show()
        else:
            if person_specific:
                filename = '../../results_image/linus_conversation4_%d_gazemap.png' % i
            else:
                filename = '../../results_image/linus_conversation4_gazemap.png'

            fig.savefig(filename, bbox_inches='tight', pad_inches=0, dpi=200)

if __name__ == "__main__":
    '''parser = ap.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-p', "--POI", help="Person of Interest")
    args = vars(parser.parse_args())'''

    gaze_prediction_image(image_path='../../test_images/linus_conversation3.jpg', person_specific=False, is_visualize=False)
