# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 11:41:28 2023

@author: user
"""

import cv2
import argparse
from datetime import datetime
import os



def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', type=str, default="00:00:00",
                               help='hh:mm:ss')
    parser.add_argument('-v', type=str, default=".//video",
                               help='hh:mm:ss')
    parser.add_argument('-o', type=str, default="out.jpg",
                               help='hh:mm:ss')
    opt = parser.parse_args()
    return opt

def screenshot(args):
    # Create a VideoCapture object and read from input file
    # If the input is the camera, pass 0 instead of the video file name
    timeObj = datetime.strptime(args.t, '%H:%M:%S').time()
    time = timeObj.hour*3600 + timeObj.minute*60 + timeObj.second
    
    vid_format = ["mp4", "avi", "dav", "asf"]
    video = args.v
    
    for fname in os.listdir(args.v):
        if fname[-3:] in vid_format:
            # do stuff on the file
            video += f"//{fname}"
            break
    
    cap = cv2.VideoCapture(video)
     
    # Check if camera opened successfully
    if (cap.isOpened()== False): 
      print("Error opening video stream or file")
    else:
      cap.set(cv2.CAP_PROP_POS_MSEC,time*1000)
      ret, frame = cap.read()
      if ret == True:
        cv2.imwrite(args.o, frame) 
        print(f"Successfull!\n{args.v} -> {args.o}")
     
    # When everything done, release the video capture object
    cap.release()
     
    # Closes all the frames
    cv2.destroyAllWindows()

if __name__ == "__main__":
    opt = parse_opt()
    screenshot(opt)

