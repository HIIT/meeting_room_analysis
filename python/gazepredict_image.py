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

def gaze_prediction_image(image_path):
    img = cv2.imread(image_path)
    points = get_points.run(img)
    if not points:
        print "ERROR: No object to be tracked."
        exit()

    pwd = '/home/wangt/Projects/apollocaffe_test/gaze_model/python/'
    for i, point in enumerate(points, 1):
        pt1 = point[0:2]
        pt2 = point[2:4]

        apollonet = None
        filename = pwd + 'results_images/avp_p%d_gazemap.png' % i
        #cv2.rectangle(img, pt1, pt2, (255, 255, 255), 3)
        net = gaze_predict(img, np.array([[pt1[0], pt1[1]], [pt2[0], pt2[1]]]))
        fig = net.result_viz('bicubic')

        fig.savefig(filename)

if __name__ == "__main__":
    pwd_test = '/home/wangt/Projects/apollocaffe_test/gaze_model/python/test_images/'

    '''parser = ap.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-p', "--POI", help="Person of Interest")
    args = vars(parser.parse_args())'''

    gaze_prediction_image(image_path=pwd_test + 'avp.jpg')
