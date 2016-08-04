import cv2
import os

def video_to_frames(video, path_output_dir):
    # extract frames from a video and save to directory as 'x.png' where
    # x is the frame index
    vidcap = cv2.VideoCapture(video)
    count = 0
    while vidcap.isOpened():
        success, image = vidcap.read()
        if count % 100 != 0:
            count += 1
            continue

        if success:
            print count
            cv2.imwrite(os.path.join(path_output_dir, '%d.png') % count, image)
            count += 1
        else:
            break

    cv2.destroyAllWindows()
    vidcap.release()

video_to_frames('/home/wangt/Projects/apollocaffe_test/gaze_model/python/test_images/MVI_0056.MP4', \
                '/home/wangt/Projects/apollocaffe_test/gaze_model/python/test_images/')
