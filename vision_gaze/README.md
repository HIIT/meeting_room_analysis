# Introduction
This module aims at capturing human-human and human-object interactions in the meeting scenario in order to carry out meaningful analyses of an effective meeting.
To achieve it, one should know what are people doing there in the meeting, e.g. watching laptop, presentation, or other people when they speak. This information may allow us, for instance, to make inference on the subject being presented during the meeting would interest the participants by looking at the period of most of the participants staring at the presentation slides.

# Functions
* Predict gaze from single image
* Predict gaze from video files (treat every frame as a single image)

# Input / Output of gaze prediction
* Input: image (or video) and head location of person of interest (this can be done in simple UI drawing a bounding box around person of interest)
* Output: heat map (the possibility of every pixel in the image being looked at by that person) and predicted coordinate of the gaze

# Features
* Multi-object tracking (from https://github.com/bikz05/object-tracker)
* Gaze following algorithm (from http://gazefollow.csail.mit.edu/)

# Requirement
One should install properly apollocaffe in order to run the code, please check https://github.com/Russell91/apollocaffe
