

import cv2
import numpy as np
import glob

from gradient import *
from cam_caliberate import *

from  utils import *


class rectangle():
    def __init__(self,x,y,width,height):
        self.x = int(x)
        self.y = int(y)
        self.width = int(width)
        self.height = int(height)
        self.center = [int(x+width//2),int(y+height//2)]
        self.non_zeros = 0

    def draw(self,img,color=(255,255,255),thickness= 2):
        cv2.rectangle(img, (self.x, self.y),
                      (self.x + self.width, self.y + self.height), color, thickness)
        cv2.circle(img,tuple(self.center) ,3,color,thickness)
        cv2.putText(img,str(self.get_occupancy()),self.get_center(),1,2,(0,255,0),2)
        cv2.putText(img, "{0:.2f}".format(self.get_occupancy()),self.get_center(), 1, 2, (0, 255, 0), 2)

    def print(self):
        print(self.x,self.y,self.width,self.height)

    def get_center(self):
        return tuple(self.center)

    def get_area(self):
        return self.width*self.height

    def get_occupancy(self):
        if self.get_area() > 0:
            return self.non_zeros /self.get_area()
        else:
            return 0


def get_rectangle(center,width=0,height=0):
    x,y = center
    return rectangle(int(x-width//2),int(y -height//2),width,height)

def draw_all_rect(img,rectangles,color=(255,255,255)):
    for rect in rectangles:
        rect.draw(img,color)

def compute_confidence(rectangels):
    lavg = 0
    for rect in rectangels:
        lavg += rect.get_occupancy()

    if len(rectangels) > 0:
        lavg /= len(rectangels)
    else:
        lavg = 0
    return lavg


def compute_mean_pos(rectangles,occpancy = 0.1):
    xavg = 0
    yavg = 0
    total_count = 0
    for rect in rectangles:
        if rect.get_occupancy() > occpancy:
            xavg += rect.get_center()[0]
            yavg += rect.get_center()[1]

    if total_count > 0:
        xavg /= total_count
        yavg /= total_count
    else:
        lavg = 0
    return (xavg, yavg)


def fit_poly(img_shape, leftx, lefty, rightx, righty):
    ### TO-DO: Fit a second order polynomial to each with np.polyfit() ###
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, img_shape[0] - 1, img_shape[0])
    ### TO-DO: Calc both polynomials using ploty, left_fit and right_fit ###
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    return left_fitx, right_fitx, ploty


class LTracker():

    def __init__(self,nwindows = 15,margin = 100,minpix = 50):
        # was the line detected in the last iteration?
        self.max_history = 10 # Total number of frames to track
        self.detected = False

        # polynomial coefficients averaged over the last n iterations
        self.best_left_fit = [np.array([False])]
        self.best_right_fit = [np.array([False])]
        # polynomial coefficients for the most recent fit
        self.lcurrent_fit = [np.array([False])]
        self.rcurrent_fit = [np.array([False])]
        self.left_fits = []
        self.right_fits = []

        # distance in meters of vehicle center from the line
        self.left_base_pos = []
        self.right_base_pos = []

        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0, 0, 0], dtype='float')
        # x values for detected line pixels
        self.left_allx = None
        self.right_allx = None
        # y values for detected line pixels
        self.left_ally = None
        self.right_ally = None

        self.window_width = 50
        self.nwindows = nwindows
        self.margin = margin
        self.minpix = minpix
        self.lradius_of_curvature = None    # radius of curvature of the line in some units
        self.rradius_of_curvature = None    # radius of curvature of the line in some units
        self.fcount = 0
        self.lost_tracking_count = 0

    def search_around_poly(self,binary_warped,enable_display = False):
        # HYPERPARAMETER
        # Choose the width of the margin around the previous polynomial to search
        # The quiz grader expects 100 here, but feel free to tune on your own!
        # margin = 100

        # Grab activated pixels
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        margin = self.margin

        left_fit = np.mean(self.left_fits,axis=0)
        right_fit = np.mean(self.right_fits,axis=0)

        left_lane_inds = ((nonzerox > ([0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy +
                                       left_fit[2] - margin)) & (nonzerox < (left_fit[0] * (nonzeroy ** 2) +
                                                                             left_fit[1] * nonzeroy + left_fit[
                                                                                 2] + margin)))
        right_lane_inds = ((nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy +
                                        right_fit[2] - margin)) & (nonzerox < (right_fit[0] * (nonzeroy ** 2) +
                                                                               right_fit[1] * nonzeroy + right_fit[
                                                                                   2] + margin)))

        # Again, extract left and right line pixel positions
        self.left_allx = nonzerox[left_lane_inds]
        self.left_ally = nonzeroy[left_lane_inds]
        self.right_allx = nonzerox[right_lane_inds]
        self.right_ally = nonzeroy[right_lane_inds]

        # Create an image to draw on and an image to show the selection window
        if (np.alen(self.left_ally) > 0) and (np.alen(self.left_allx) > 0) and (np.alen(self.right_ally) > 0) and (np.alen(self.right_allx) > 0) :
            if enable_display:
                left_fitx, right_fitx, ploty = fit_poly(binary_warped.shape, self.left_allx, self.left_ally,self. right_allx,self. right_ally)
                out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
                window_img = np.zeros_like(out_img)
                # Color in left and right line pixels
                out_img[self.left_ally,self.left_allx] = [255, 0, 0]
                out_img[self.right_ally, self.right_allx] = [0, 0, 255]

                # Generate a polygon to illustrate the search window area
                left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
                left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin,
                                                                                ploty])))])
                left_line_pts = np.hstack((left_line_window1, left_line_window2))
                right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
                right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin,
                                                                                 ploty])))])
                right_line_pts = np.hstack((right_line_window1, right_line_window2))

                # Draw the lane onto the warped blank image
                cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
                cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
                result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

                show_img('search_area',result)
            #indicating it was a success
            return 100
        else:
            return None



    def find_pixels(self,binary_warped, out_img, enable_display=False, nwindows=9, margin=50, minpix=50):
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
        # Create an output image to draw on and visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0] // 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Set height of windows - based on nwindows above and image shape
        window_height = np.int(binary_warped.shape[0] // nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        left_rects = []
        right_rects = []

        leftx_current = leftx_base
        rightx_current = rightx_base

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            lrect = rectangle(leftx_current, win_y_low, 2 * margin, window_height)
            rrect = rectangle(rightx_current, win_y_low, 2 * margin, window_height)
            # Draw the windows on the visualization image

            # cv2.rectangle(out_img, (win_xleft_low, win_y_low),
            #               (win_xleft_high, win_y_high), (0, 255, 0), 2)
            # cv2.rectangle(out_img, (win_xright_low, win_y_low),
            #               (win_xright_high, win_y_high), (0, 255, 0), 2)

            # Identify the nonzero pixels in x and y within the window #
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                              (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]

            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                               (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)


            # If you found > minpix pixels, recenter next window on their mean position

            lrect.non_zeros = len(good_left_inds)
            rrect.non_zeros = len(good_right_inds)

            if lrect.non_zeros > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
                temp = get_rectangle((leftx_current, win_y_low), width=2 * margin, height=window_height)
                temp.non_zeros = lrect.non_zeros
                lrect = temp

            if rrect.non_zeros > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
                temp = get_rectangle((rightx_current, win_y_low), width=2 * margin, height=window_height)
                temp.non_zeros = rrect.non_zeros
                rrect = temp

            left_rects.append(lrect)
            right_rects.append(rrect)
        # Concatenate the arrays of indices (previously was a list of lists of pixels)
        try:
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)
        except ValueError:
            # Avoids an error if the above is not implemented fully
            pass

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        if enable_display:

            draw_all_rect(out_img, left_rects, color=(255, 0, 0))
            draw_all_rect(out_img, right_rects, color=(0, 255, 0))
            lavg = 0
            ravg = 0

            for rect in left_rects:
                lavg += rect.get_occupancy()
            for rect in right_rects:
                ravg += rect.get_occupancy()

            if len(left_rects) > 0:
                lavg /= len(left_rects)
            else:
                lavg = 0

            if len(right_rects) > 0:
                ravg /= len(right_rects)
            else:
                ravg = 0
            #
            cv2.putText(out_img, "left {0:.3f}".format(lavg), (20, 40), 1, 2, (0, 255, 0))
            cv2.putText(out_img, "right {0:.3f}".format(ravg), (10, 80), 1, 2, (0, 255, 0))
            show_img('out_img', out_img)
            # cv2.imwrite('selected_boxes.jpg', out_img)

        return leftx, lefty, rightx, righty, left_rects, right_rects

    def detect_lane(self,warped_binary):

        threshold = 0.1

        temp_img = np.zeros_like(warped_binary)
        draw_img = np.dstack((temp_img,temp_img,temp_img))

        # show_img('binary_warped',warped_binary)
        # print('warped binary image shape is ',warped_binary.shape)
        if self.detected == False:
            # print(' I m running the detector now .........')
            # if lane is not found in the couurent frame find lane from begining
            self.left_allx, self.left_ally, self.right_allx, self.right_ally, left_rects, right_rects = self.find_pixels(warped_binary,warped_binary,False,12)
            left_confidence = compute_confidence(left_rects)
            right_confidence = compute_confidence(right_rects)

            cv2.putText(draw_img," lcon {0:.2f}".format(left_confidence),(30,40),1,2,(0,255,0))
            cv2.putText(draw_img, " rcon {0:.2f}".format(right_confidence), (30, 60), 1, 2, (0, 255, 0))

            self.left_base_pos.append(compute_mean_pos(left_rects,threshold))
            self.right_base_pos.append(compute_mean_pos(right_rects, threshold))

            cv2.putText(draw_img," lbase {}".format(list(np.mean(self.left_base_pos,axis=0))),(200,30),1,1,(0,255,0))
            cv2.putText(draw_img, " rbase {}".format(list(np.mean(self.right_base_pos,axis=0))), (200, 60), 1, 1, (0, 255, 0))

            # Assuming that we get some pixels we fit the lane curves
            self.lcurrent_fit = np.polyfit(self.left_ally, self.left_allx, 2)
            self.rcurrent_fit = np.polyfit(self.right_ally, self.right_allx, 2)
            if self.fcount > 5:
                self.detected = True
            self.lost_tracking_count = 0
            print('D: best prev curve ', self.lcurrent_fit, '  best mean fit' ,self.best_left_fit)

        else:
            # Track the existing curve if failed
            print('T: best prev curve ',self.lcurrent_fit, '  best mean fit',self.best_left_fit)

            self.lcurrent_fit = self.best_right_fit
            self.rcurrent_fit = self.best_left_fit

            if self.search_around_poly(warped_binary):
                self.lcurrent_fit = np.polyfit(self.left_ally, self.left_allx, 2)
                self.rcurrent_fit = np.polyfit(self.right_ally, self.right_allx, 2)

                left_error = np.sum((self.best_left_fit - self.lcurrent_fit)**2)
                right_error = np.sum((self.best_right_fit - self.rcurrent_fit) ** 2)

                cv2.putText(draw_img, " lerror {}".format(left_error), (40, 300), 1, 2,
                            (0, 255, 0))
                cv2.putText(draw_img, " rerror {}".format(right_error), (40, 300), 1, 2,
                            (0, 255, 0))
                # check error

            else:
                cv2.putText(draw_img, " lost_tracking {}", (30, 300), 1, 2,(0, 255, 0))

                self.lcurrent_fit = self.best_left_fit
                self.rcurrent_fit = self.best_right_fit
                self.lost_tracking_count += 1

                if self.lost_tracking_count > 5:
                    self.detected = False

        cv2.putText(draw_img, " best_left {}".format(list(self.best_left_fit)), (300, 100), 1, 1, (0, 255, 0))
        cv2.putText(draw_img, " best_right {}".format(list(self.best_right_fit)), (300, 150), 1, 1, (0, 255, 0))

        cv2.putText(draw_img, " current_left {}".format(list(self.best_left_fit)), (300, 200), 1, 1, (0, 255, 0))
        cv2.putText(draw_img, " current_right {}".format(list(self.best_right_fit)), (300, 250), 1, 1, (0, 255, 0))

        show_img('draw_img_tracker',draw_img)

        self.left_fits.append(self.lcurrent_fit)
        self.right_fits.append(self.rcurrent_fit)
        self.best_right_fit = np.mean(np.array(self.right_fits), axis=0)
        self.best_left_fit = np.mean(np.array(self.left_fits), axis=0)

        self.fcount += 1

        # keep all the values till the max_history
        if self.fcount % self.max_history == 0:
            self.detected = False # Needs to be tested
            self.best_right_fit = np.mean(np.array(self.right_fits),axis=0)
            self.best_left_fit = np.mean(np.array(self.left_fits), axis=0)
            self.left_fits = []
            self.right_fits = []
            self.left_fits.append(self.best_left_fit)
            self.right_fits.append(self.best_right_fit)

            # reset all
        #
        y_eval = warped_binary.shape[0]
        self.get_radius_of_curvature(y_eval,warped_binary.shape)
        #car center is

        shape = warped_binary.shape
        ym_per_pix = 30 / shape[0]  # meters per pixel in y dimension
        xm_per_pix = 3.7 /shape[1] # meters per pixel in x dimension

        left_bottomx = self.lcurrent_fit[0]* shape[0]**2 + self.lcurrent_fit[1]*shape[0] + self.lcurrent_fit[2]
        right_bottomx = self.rcurrent_fit[0] * shape[0]** 2 + self.rcurrent_fit[1] * shape[0] + self.rcurrent_fit[2]

        car_center = (left_bottomx + right_bottomx) //2
        # print(' The car center is at ',car_center)
        self.offset = (shape[1]//2 - car_center)*xm_per_pix


    def get_center_offset(self):
        return self.offset

    def get_radius_of_curvature(self,y_eval,shape):
        #measure real world curvature
        ym_per_pix = 30 / shape[0]  # meters per pixel in y dimension
        xm_per_pix = 3.7 /shape[1] # meters per pixel in x dimension

        y_eval = y_eval * ym_per_pix

        # Define y-value where we want radius of curvature
        # We'll choose the maximum y-value, corresponding to the bottom of the image
        if np.alen(self.left_ally) > 0 and np.alen(self.left_allx) > 0 :
            left_fit_cr = np.polyfit(np.array(self.left_ally)*ym_per_pix, np.array(self.left_allx)*xm_per_pix, 2)
            self.lradius_of_curvature = ((1 + (2 * left_fit_cr[0] * y_eval + left_fit_cr[1]) ** 2) ** 1.5) / np.abs(
                2 * left_fit_cr[0])
        else:
            self.lradius_of_curvature = 0

        if np.alen(self.right_ally) > 0 and np.alen(self.right_allx) > 0:
            right_fit_cr = np.polyfit(np.array(self.right_ally)*ym_per_pix, np.array(self.right_allx)*xm_per_pix, 2)
            self.rradius_of_curvature = ((1 + (2 * right_fit_cr[0] * y_eval + right_fit_cr[1]) ** 2) ** 1.5) / np.abs(2 * right_fit_cr[0])  ## Implement the calculation of the right line here
        else:
            self.rradius_of_curvature = 0

        return self.lradius_of_curvature,self.rradius_of_curvature



    # return an color image containing the fitted line along with overlay
    def draw_overlay(self,img,lcolor=(255,0,0),rcolor=(255,0,0),overlay_color=(0,255,0)):
        # print('Fitted line curve co-efficients ', left_fit, ' right fit is ', right_fit)
        # Generate x and y values for plotting
        lane = img
        lane = np.dstack((lane, lane, lane))

        lpoints = []
        rpoints = []
        left_fit = self.best_left_fit  # best fit curve for the past N frames
        right_fit = self.best_right_fit  # best fit curve for the past N frames
        for ycap in range(img.shape[0]):
            lpt = (int(left_fit[0] * (ycap ** 2) + left_fit[1] * ycap + left_fit[2]), ycap)
            rpt = (int(right_fit[0] * (ycap ** 2) + right_fit[1] * ycap + right_fit[2]), ycap)
            lpoints.append(lpt)
            rpoints.append(rpt)
            # cv2.circle(lane, lpt, 1, lcolor, 10)
            # cv2.circle(lane, rpt, 1, rcolor, 10)

        larr = np.array([lpoints])
        flipped = np.array([np.flipud(rpoints)])
        pts = np.hstack((larr, flipped))
        cv2.fillPoly(lane, np.int_([pts]),overlay_color)

        for ycap in range(img.shape[0]):
            lpt = (int(left_fit[0] * (ycap ** 2) + left_fit[1] * ycap + left_fit[2]), ycap)
            rpt = (int(right_fit[0] * (ycap ** 2) + right_fit[1] * ycap + right_fit[2]), ycap)
            cv2.circle(lane, lpt, 1, lcolor, 10)
            cv2.circle(lane, rpt, 1, rcolor, 10)

        return lane





