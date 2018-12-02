from utils import *
import  numpy as np

# img_shape = (640,480)
# print(np.power(2,2))
#
# a = np.array([1,4,5,6])
# a = np.random.rand(2,2)
# b = np.flipud(a)
#
# print(b,'\n',a)

nx = 9
ny = 7

# offset = 100
#
# img_size = (480,640)
#
# dst = np.float32([[offset, offset], [img_size[0] - offset, offset],
#                   [img_size[0] - offset, img_size[1] - offset],
#                   [offset, img_size[1] - offset]])
# # x = np.arange(0, nx, 1)
# # y = np.linspace(0, ny,1)
# # k = np.mgrid[0:nx,0:ny].T.reshape(-1, 2)
# # print(dst)
# # p = (100,255)
# # print('after it')
# # k = [tuple(dst[x].astype(int)) for x in range(np.alen(dst))]
#
# print(str(p))

img_shape = (720,1080)

ploty = np.linspace(0, 10, 10)
left_fit =  [ -1.02920471e-04 ,  1.92537910e-02  , 2.32951949e+02]

ploty = np.linspace(0, img_shape[0] - 1, img_shape[0])


x = np.array([(10,100), (10,100), (10,100), (10,100), (10,100)])
y = np.array([30, 45, 40, 20,  40])


print(" {} ".format((list(x))))
#
# pts = []
# for y in range(img_shape[0]):
#     left_fitx = left_fit[0] * (y ** 2) + left_fit[1] * y + left_fit[2]
#     pts.append((left_fitx,y))

# fit = np.polyfit(x,y,2)



# y_cap = fit[0]*x**2 + fit[1]*x +fit[2]
#
# print(y)
# print(y_cap)
