import cv2
import numpy as np

def show_img(win_name,img,auto_resize=0):
    cv2.namedWindow(win_name,auto_resize)
    cv2.imshow(win_name,img)


def draw_pts(img,pts,color=(255,255,255)):
    for p in pts:
        cv2.circle(img,p,1,color,1)
        # cv2.putText(img,str(p),p,1,1.,color)