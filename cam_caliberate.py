import numpy as np
import cv2
import matplotlib.pyplot as plt

import os
import pickle
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.image as mpimg

# prepare object points
nx = 9 #TODO: enter the number of inside corners in x
ny = 6 #TODO: enter the number of inside corners in y


cam_images_path = './camera_cal/'
corners_out_dir = './output_images/cam_corners/'
undistort_out_dir = './output_images/undistorted/'
test_images = glob.glob('./test_images/*.jpg')

image_world_file = corners_out_dir + 'img_wld.pickle'
cam_param = corners_out_dir + 'cam_param.pickle'


# # # Make a list of calibration images
# # fname = 'calibration_test.png'
# # img = cv2.imread(fname)
# #
# # # Convert to grayscale
# # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# #
# # # Find the chessboard corners
# # ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
#
# # If found, draw corners
# # if ret == True:
# #     # Draw and display the corners
# #     cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
# #     plt.imshow(img)
# # else:
# #     plt.imshow(img)
#
#
# # Read in the saved objpoints and imgpoints
# dist_pickle = pickle.load(open("wide_dist_pickle.p", "rb"))
# objpoints = dist_pickle["objpoints"]
# imgpoints = dist_pickle["imgpoints"]
#
# # Read in an image
# img = cv2.imread('test_image.png')
#
#
# # TODO: Write a function that takes an image, object points, and image points
# # performs the camera calibration, image distortion correction and
# # returns the undistorted image
# def cal_undistort(img, objpoints, imgpoints):
#     # Use cv2.calibrateCamera() and cv2.undistort()
#
#     rows, cols, channel = img.shape
#     img_size = (rows, cols)
#     ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
#     dst = cv2.undistort(img, mtx, dist, None, mtx)
#
#     # undist = np.copy(img)  # Delete this line
#     return dst
#
#
# # undistorted = cal_undistort(img, objpoints, imgpoints)
# #
# # f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
# # f.tight_layout()
# # ax1.imshow(img)
# # ax1.set_title('Original Image', fontsize=50)
# # ax2.imshow(undistorted)
# # ax2.set_title('Undistorted Image', fontsize=50)
# # plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
#
# # Define a function that takes an image, number of x and y points,
# # camera matrix and distortion coefficients
# def corners_unwarp(img, nx, ny, mtx, dist):
#     # Use the OpenCV undistort() function to remove distortion
#     undist = cv2.undistort(img, mtx, dist, None, mtx)
#     # Convert undistorted image to grayscale
#     gray = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)
#     # Search for corners in the grayscaled image
#     ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
#
#     if ret == True:
#         # If we found corners, draw them! (just for fun)
#         cv2.drawChessboardCorners(undist, (nx, ny), corners, ret)
#         # Choose offset from image corners to plot detected corners
#         # This should be chosen to present the result at the proper aspect ratio
#         # My choice of 100 pixels is not exact, but close enough for our purpose here
#         offset = 100 # offset for dst points
#         # Grab the image shape
#         img_size = (gray.shape[1], gray.shape[0])
#
#         # For source points I'm grabbing the outer four detected corners
#         src = np.float32([corners[0], corners[nx-1], corners[-1], corners[-nx]])
#         # For destination points, I'm arbitrarily choosing some points to be
#         # a nice fit for displaying our warped result
#         # again, not exact, but close enough for our purposes
#         dst = np.float32([[offset, offset], [img_size[0]-offset, offset],
#                                      [img_size[0]-offset, img_size[1]-offset],
#                                      [offset, img_size[1]-offset]])
#         # Given src and dst points, calculate the perspective transform matrix
#         M = cv2.getPerspectiveTransform(src, dst)
#         # Warp the image using OpenCV warpPerspective()
#         warped = cv2.warpPerspective(undist, M, img_size)
#
#     # Return the resulting image and matrix
#     return warped, M


from utils import *


img_pts = [] # containes x,y values
obj_pts = [] # containes x,y,z basically ideal chess board grid 1 unit by 1 unit uniformly distributed

# performs the camera calibration, image distortion correction and
# returns the undistorted image
def cal_undistort(img, objpoints, imgpoints):
    # Use cv2.calibrateCamera() and cv2.undistort()
    rows, cols, channel = img.shape
    img_size = (rows, cols)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
    dst = cv2.undistort(img, mtx, dist, None, mtx)

    # undist = np.copy(img)  # Delete this line
    return dst


# with open('filename.pickle', 'wb') as handle:
#     pickle.dump(a, handle, protocol=pickle.HIGHEST_PROTOCOL)
#
# with open('filename.pickle', 'rb') as handle:
#     b = pickle.load(handle)

def caliberate(load_param=False):
    # find chessboard corners
    temp = dict()
    if not load_param:
        for i,img_path in enumerate(os.listdir(cam_images_path)):
            img = cv2.imread(cam_images_path+img_path)
            # print(os.path.basename(img_path),img_path)
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
            if ret == True:
                objp = np.zeros((nx * ny, 3), np.float32)
                objp[:, :2] = np.mgrid[0:nx,0:ny].T.reshape(-1, 2)
                # Draw and display the corners
                cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
                img_pts.append(corners)
                obj_pts.append(objp)
                cv2.imwrite(corners_out_dir+img_path,img)

        temp['objpts'] = obj_pts
        temp['imgpts'] = img_pts
        with open(image_world_file,'wb') as out_file:
            pickle.dump(temp,out_file)
        # show_img('corners', img,1)
    else:
        with open(image_world_file,'rb') as in_file:
            temp = pickle.load(in_file)
    return temp['objpts'], temp['imgpts']


def load_cam_param(img_shape,load_param=False):
    rows, cols, channel = img_shape
    img_size = (rows, cols)
    objpoints, imgpoints = caliberate(load_param)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
    return

def undistort(img,load_param=False):
    # Use cv2.calibrateCamera() and cv2.undistort()
    temp = dict()
    rows, cols, channel = img.shape
    img_size = (rows, cols)
    objpoints, imgpoints = caliberate(load_param)
    if not load_param:
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
        temp['mtx'] = mtx
        temp['dist'] = dist
        temp['rvecs'] = rvecs
        temp['tvecs'] = tvecs
        with open(cam_param,'wb') as out_file:
            pickle.dump(temp, out_file)
    else:
        with open(cam_param,'rb') as in_file:
            temp = pickle.load(in_file)

    dst = cv2.undistort(img, temp['mtx'], temp['dist'], None, temp['mtx'])
    return dst




# perform caliberation by computing the chess_corner points
def save_distortion_correction():
    for i,image in enumerate(test_images):
        # print(image,os.path.basename(image))
        img = cv2.imread(image)
        if i == 0:
            obj_pts,img_pts = caliberate()
        else:
            obj_pts, img_pts = caliberate(True)
        undistorted = cal_undistort(img, obj_pts, img_pts)
        concated = cv2.hconcat([img,undistorted])
        cv2.imwrite(undistort_out_dir+os.path.basename(image),concated)
        # show_img('undistorted',concated)
        # cv2.waitKey(0)



