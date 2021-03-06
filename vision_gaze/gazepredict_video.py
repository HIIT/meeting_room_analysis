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
import scipy
import network
from network import GazeFollowNet
import matplotlib.pyplot as plt
import os
from os import listdir
from os.path import isfile, join
import cv2
import dlib
import argparse as ap
import object_tracker.get_points as get_points

def detect(video_path):
    cam = cv2.VideoCapture(video_path)
    cam.set(1, 55000)
    # If Camera Device is not opened, exit the program
    if not cam.isOpened():
        print "Video device or file couldn't be opened"
        exit()

    '''print "Press key `p` to pause the video to start tracking"
    count = -1

    while True:
        # Retrieve an image and Display it.
        retval, img = cam.read()
        if not retval:
            print "Cannot capture frame device"
            exit()
        if(cv2.waitKey(1)==ord('p')):
            break
        cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
        cv2.imshow("Image", img)
    cv2.destroyWindow("Image")'''

    retval, img = cam.read()
    #img = img[200::, 180:1050, ::].astype('uint8')
    cv2.imwrite('test.png', img)
    # Co-ordinates of objects to be tracked
    # will be stored in a list named `points`
    points = get_points.run(img)

    if not points:
        print "ERROR: No object to be tracked."
        exit()

    return cam, img, points
    #cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    #cv2.imshow("Image", img)

def track(tracker, img=None):
    assert(img is not None)

    # Update the tracker
    tracker.update(img)
    # Get the position of the object, draw a
    # bounding box around it and display it.
    rect = tracker.get_position()
    pt1 = (int(rect.left()), int(rect.top()))
    pt2 = (int(rect.right()), int(rect.bottom()))

    return pt1, pt2

def gaze_predict(img, head_loc, net=None):
    if net is None:
        net = GazeFollowNet(img, head_loc)
    else:
        net = GazeFollowNet(img, head_loc, net)
    net.run()

    return net

# detect -> track head -> predict gaze
def gaze_prediction_pipeline(video_path, person_specific=False, is_visualize=False):
    video_source, img, init_head_locs = detect(video_path)
    tracker = [dlib.correlation_tracker() for _ in xrange(len(init_head_locs))]
    [tracker[i].start_track(img, dlib.rectangle(*rect)) for i, rect in enumerate(init_head_locs)]

    count = -1
    save_count = -1
    apollonet = None

    while True:
        count += 1
        retval, img = video_source.read()
        #img = img[200::, 180:1050, ::].astype('uint8')
        frame_idx = int(video_source.get(1))

        if not retval:
            print "Cannot capture frame device | CODE TERMINATING :("
            exit()
        #pt1, pt2 = track(tracker, img)
        #cv2.rectangle(img, pt1, pt2, (255, 255, 255), 3)

        head_boxes = []
        for i in xrange(len(tracker)):
            pt1, pt2 = track(tracker[i], img)
            head_boxes.append([pt1[0], pt1[1], pt2[0], pt2[1]])

        if frame_idx % 30 == 0:
            pwd = '../../results/'
            net = gaze_predict(img, head_boxes)
            figs = net.result_viz('bicubic', person_specific=person_specific)

            for i, fig in enumerate(figs, 1):
                if is_visualize:
                    plt.figure(fig.number, bbox_inches='tight', pad_inches=0)
                    plt.pause(0.01)
                else:
                    if person_specific:
                        filename = '../../results_video/2016-04-08-pr-psycho2/%d_p%d_gazemap.png' % (frame_idx, i)
                    else:
                        filename = '../../results_video/2016-04-08-pr-psycho2/%d_all_gazemap.png' % (frame_idx)

                    fig.savefig(filename, bbox_inches='tight', pad_inches=0, dpi='figure')
            save_count += 1
            if save_count > 150:
                break

        #cv2.imshow("Image", img)
        # Continue until the user presses ESC key
        #if cv2.waitKey(1) == 27:
        #    break

    video_source.release()
    return

if __name__ == "__main__":
    #pwd_test = '/home/wangt/Projects/apollocaffe_test/gaze_model/python/test_images/'
    pwd_test = '../../test_videos/2016-04-08-pr-psycho/'
    video_path = pwd_test + 'MVI_0056.MP4'

    '''parser = ap.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-p', "--POI", help="Person of Interest")
    args = vars(parser.parse_args())'''

    #gaze_prediction_pipeline(video_path, args["POI"])
    gaze_prediction_pipeline(video_path, person_specific=True, is_visualize=False)
