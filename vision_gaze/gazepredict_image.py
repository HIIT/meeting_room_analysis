import numpy as np
import network
from network import GazeFollowNet
import matplotlib.pyplot as plt
import cv2
import object_tracker.get_points as get_points

def detect(video_path):
    cam = cv2.VideoCapture(video_path)
    cam.set(1, 21900)
    # If Camera Device is not opened, exit the program
    if not cam.isOpened():
        print "Video device or file couldn't be opened"
        exit()

    retval, img = cam.read()
    print img.shape
    #img = img[100::, 250:1100, ::].astype('uint8')
    print img.shape
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
                filename = '../../results_image/linus_exactum_%d_gazemap.png' % i
            else:
                filename = '../../results_image/linus_exactum_gazemap.png'

            fig.savefig(filename, bbox_inches='tight', pad_inches=0, dpi=1000)

if __name__ == "__main__":
    '''parser = ap.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-p', "--POI", help="Person of Interest")
    args = vars(parser.parse_args())'''

    gaze_prediction_image(image_path='../../test_images/linus_exactum.jpg', person_specific=False, is_visualize=False)
