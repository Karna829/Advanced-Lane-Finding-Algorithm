
# All possible imports
import glob
import cv2
from cam_caliberate import *
from gradient import get_binary
from Lane import LTracker


# test images and output directories
test_images = glob.glob('./test_images/*.jpg')
videos = glob.glob('./*.mp4')
binary_out_dir = './output_images/binary_images/'
top_view_out_dir = './output_images/top_view/'
top_view_bin_out_dir = './output_images/top_view_binary/'
# save_distortion_correction()

# for vid_name in videos:
#     cap = cv2.VideoCapture(r'./challenge_video.mp4')   # r'./challenge_video.mp4'  r'./project_video.mp4'
#     ret, frame = cap.read()
#     rows,cols,ch = frame.shape
#     tracker = LTracker()
#     # vid_writer = cv2.VideoWriter('./project_video_output.mp4',cv2.VideoWriter_fourcc('M','J','P','G'),15,(cols,rows),True)
#     fcount = 0
#     while (True):
#         # Capture frame-by-frame
#         ret, frame = cap.read()
#         if ret == False:
#             break
#         fcount += 1
#         # resized to get faster output
#         shape = np.array(frame).shape
#         scale_factor = 1
#         # frame = cv2.resize(frame,
#         # (shape[1]//scale_factor,shape[0]//scale_factor))
#         # actual process
#         # gray  = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
#         undistorted = undistort(frame,True)
#         binary = get_binary(undistorted)
#         # show_img('binary',binary)
#         top_view_bin,M, Minv = get_top_view(binary)
#         # top_view, _,_ = get_top_view(gray)
#         # show_img('top_view',top_view_bin)
#
#         tracker.detect_lane(top_view_bin)
#         lane_img = tracker.draw_overlay(top_view_bin)
#
#         overlay = cv2.warpPerspective(lane_img,Minv,(binary.shape[1],binary.shape[0]))
#         # show_img('overlay',overlay)
#
#         left_curvature,right_curvature = tracker.get_radius_of_curvature(binary.shape[0],binary.shape)
#         # display
#         final_img = cv2.addWeighted(frame,1,overlay,0.3,0)
#         cv2.putText(final_img,
#                     'Radius of Curvature ' + str(int((left_curvature+right_curvature)//2)) + ' m',
#                     (30,40),1,2,(255,255,255))
#         cv2.putText(final_img,
#                     'Center Offset  '+"{0:.2f} m".format(round(tracker.get_center_offset(),2)),(30,80),1,2,(255,255,255))
#
#         # cv2.putText(final_img,os.path.basename(vid_name),(30,100),1,2.,(255,255,255),1)
#         show_img('final',final_img)
#
#         # vid_writer.write(final_img)
#         # show_img('output',frame)
#         cv2.waitKey(1)
#
#     break


def get_top_view(undist,enable_display=False,color_image=None):
    rows,cols = undist.shape  # expects gray image
    offset = 100 # 50
    side_offset = 150
    img_size = (cols, rows)
    btm_center = (cols//2,rows)

    width_near = int(0.35*cols)
    some_offset = 0
    width_far = int(0.32*width_near)  # Good Ratioo 0.35
    top_p1 = (int(btm_center[0] - width_far + some_offset) , int(0.7*rows))          # Good height ratio
    top_p2 = (int(btm_center[0] + width_far + some_offset), int(0.7* rows))

    btm_p1 = (int(btm_center[0] - width_near + some_offset) , int(0.95*rows))
    btm_p2 = (int(btm_center[0] + width_near + some_offset),  int(0.95*rows))

    src = np.float32([top_p1, top_p2, btm_p2, btm_p1])
    dst = np.float32([[offset, offset ], [img_size[0] - side_offset, offset],
                      [img_size[0] - side_offset, img_size[1] - offset],
                      [offset , img_size[1] - offset ]])

    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst,src)

    warped = cv2.warpPerspective(undist, M,(cols,rows))
    if enable_display:
        cv2.line(color_image, btm_p1, top_p1, (0, 255, 0))
        cv2.line(color_image, top_p1, top_p2, (0, 255, 0))
        cv2.line(color_image, top_p2, btm_p2, (0, 255, 0))
        cv2.line(color_image, btm_p2, btm_p1, (0, 255, 0))
        cv2.putText(color_image,str(btm_p1),btm_p1,1,1,(0,0,255))
        cv2.putText(color_image, str(top_p1), top_p1, 1, 1, (0, 0, 255))
        cv2.putText(color_image, str(top_p2), top_p2, 1, 1, (0, 0, 255))
        cv2.putText(color_image, str(btm_p2), btm_p2, 1, 1, (0, 0, 255))
        show_img('undistorted',color_image)
        twarped = cv2.warpPerspective(color_image, M, (cols, rows))
        show_img('get_top_view',twarped)

    return warped,M,Minv



def process_image(frame,tracker):
    # load camera-parametes
    undistorted = undistort(frame, True)
    # get binary image after thresholding
    binary = get_binary(undistorted)
    # get perspective corrected image and matrices
    bin_top_view, M, Minv = get_top_view(binary)
    # detect lane using tracker
    tracker.detect_lane(bin_top_view)
    # get the resulting overlay
    lane_img = tracker.draw_overlay(bin_top_view)
    # apply inverse perspective to get image size same as binary
    overlay = cv2.warpPerspective(lane_img,Minv,(binary.shape[1],binary.shape[0]))
    # compute radius of curvature
    left_curvature,right_curvature = tracker.get_radius_of_curvature(binary.shape[0],binary.shape)
    # display
    # create final image
    final_img = cv2.addWeighted(frame, 1, overlay, 0.3, 0)
    cv2.putText(final_img,
                'Radius of Curvature ' + str(int((left_curvature+right_curvature)//2)) + ' m',
                (30,40),1,2,(255,255,255))
    cv2.putText(final_img,
                'Center Offset  '+"{0:.2f} m".format(round(tracker.get_center_offset(),2)),(30,80),1,2,(255,255,255))

    show_img('final',final_img)

    # vid_writer.write(final_img)
    # show_img('output',frame)
    # cv2.imwrite('Final_Image_output.jpg',final_img)
    # cv2.waitKey(0)



# code to test on images present in the folder
# tracker = LTracker()
# for i,image in enumerate(test_images):
#     img = cv2.imread(image)
#     # show_img('src', img)
#     process_image(img,tracker)
#     cv2.waitKey(0)

# check video by video generate the output if required
for vid_name in videos:
    cap = cv2.VideoCapture(vid_name )   # r'./challenge_video.mp4'  r'./project_video.mp4' r'./harder_challenge_video.mp4'
    ret, frame = cap.read()
    rows,cols,ch = frame.shape
    tracker = LTracker()
    # vid_writer = cv2.VideoWriter('./project_video_output.mp4',cv2.VideoWriter_fourcc('M','J','P','G'),15,(cols,rows),True)
    fcount = 0
    while (True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break
        process_image(frame, tracker)
        cv2.waitKey(1)



