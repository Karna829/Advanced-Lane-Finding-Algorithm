import numpy as np
import cv2

from utils import *
from cam_caliberate import  undistort

def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Calculate directional gradient
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if orient == 'x':
        sobel = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=sobel_kernel)
    else:
        sobel = cv2.Sobel(gray, cv2.CV_64F , 0, 1, ksize=sobel_kernel)
    sobel_abs = np.absolute(sobel)
    scaled_sobel = np.uint8(255 * sobel_abs / np.max(sobel_abs))
    binary = np.zeros_like(scaled_sobel)
    # Apply threshold
    binary[(scaled_sobel > thresh[0]) & (scaled_sobel < thresh[1])] = 255
    return binary



def magnitude(image, sobel_kernel=3, mag_thresh=(0, 255)):
    # Calculate gradient magnitude
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dx = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=sobel_kernel)
    dy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    mag = np.sqrt(np.square(dx) + np.square(dy))
    mag_scaled = np.uint8(255*mag/np.max(mag))

    # show_img('mag_scaled',mag_scaled)
    mag_binary = np.zeros_like(mag_scaled)
    # Apply threshold

    mag_binary[(mag_scaled >= mag_thresh[0]) & (mag_scaled <= mag_thresh[1])] = 255
    return mag_binary

def dir_threshold(image, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Calculate gradient direction
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dx = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=sobel_kernel)
    dy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobelx = np.absolute(dx)
    abs_sobely = np.absolute(dy)
    theta = np.arctan2(abs_sobely, abs_sobelx)
    dir_binary = np.zeros_like(theta)
    # Apply threshold
    dir_binary[(theta >= thresh[0]) & (theta <= thresh[1])] = 255
    return dir_binary


# def hls_select(img, thresh=(0, 255)):
#     hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
#     s_channel = hls[:,:,2]
#     binary_output = np.zeros_like(s_channel)
#     binary_output[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1
#     return binary_output



def color_threshold(image,hthresh=(0,179),sthresh=(0,255),vthresh=(0,255)):
    #
    binaryh = np.zeros_like(image[:, :, 0])
    binarys = np.zeros_like(image[:, :, 0])
    binaryv = np.zeros_like(image[:, :, 0])
    binary = np.zeros_like(image[:, :, 0])
    #
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    #
    hchannel = hsv[:, :, 0]
    schannel = hsv[:, :, 1]
    vchannel = hsv[:, :, 2]

    # show_img('hchannel', hchannel)
    # show_img('schannel', schannel)
    binaryh[(hchannel > hthresh[0]) & (hchannel < hthresh[1])] = 255
    # show_img('hbin',binaryh)
    binarys[(schannel >= sthresh[0]) & (schannel <= sthresh[1])] = 255

    binaryv[(vchannel >= vthresh[0]) & (vchannel <= vthresh[1])] = 255
    # show_img('sbin', binarys)
    binary[((binaryh == 255) & (binarys == 255)) | (binaryv >= 150)] = 255    # addition of

    return binary

def get_binary(image,canny_thresh=(50,100),enable_display = False):
    #undistort the image
    image = undistort(image,load_param=True)
    # Apply some smoothening
    image = cv2.GaussianBlur(image,(5,5),1)
    # Apply color thresholding to select yellow color old value is sthresh 80 ,60 also good
    c_binary = color_threshold(image,hthresh=(15,30),sthresh=(60,240),vthresh=(240,255))
    # Apply gradient thresholds to extract sharper edges.
    mag_binary = magnitude(image,5,mag_thresh =(40,150))
    # Final result Image
    res = np.zeros_like(mag_binary)
    res[((mag_binary == 255) | (c_binary == 255))] = 255

    if enable_display == True:
        show_img('get_binary',res)
    return res

# Choose a Sobel kernel size
# ksize = 3 # Choose a larger odd number to smooth gradient measurements

# Apply each of the thresholding functions
# gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=(0, 255))
# grady = abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh=(0, 255))
# mag_binary = mag_thresh(image, sobel_kernel=ksize, mag_thresh=(0, 255))
# dir_binary = dir_threshold(image, sobel_kernel=ksize, thresh=(0, np.pi/2))