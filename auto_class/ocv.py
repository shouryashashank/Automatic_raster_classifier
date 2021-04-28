
import numpy as np
import cv2
from PIL import Image

from geotiff import GeoTiff
from matplotlib import pyplot as plt
import matplotlib.image as ima


# geoTiff = GeoTiff('test2.tif')
# array = geoTiff.read()
# print (array)

def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)

# testImage = ima.imread('TI.tif')
# plt.imshow(testImage)
img = cv2.imread('TI.tif')
img=ResizeWithAspectRatio(img, width=1280)
print (img.shape)
print (img[400][521])

# img_number = 0
# i=0
# j=0
# k=0
# detail=100
# for i in range(255):
#     for j in range(255):
#         for k in range(255):
#             split_image=img.copy()
#             # for x in range(split_image.shape[1]):
#             #     for y in range(split_image.shape[0]):
#             low=np.array([i,j,k])
#             high=np.array([i+detail,j+detail,k+detail])
#             mask = cv2.inRange(split_image, low, high)
#             split_image = cv2.bitwise_and(split_image,split_image, mask= mask)
#             cv2.imwrite("generated/"+str(img_number)+".jpg",split_image)
#             img_number=img_number+1
#             k=k+detail
            
#         j=j+detail
        
#         print(i)
#     i=i+detail
                    

# Z = np.float32(img.reshape((-1,3)))

# criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
# K = 4
# _,labels,centers = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
# labels = labels.reshape((img.shape[:-1]))
# reduced = np.uint8(centers)[labels]

# result = [np.hstack([img, reduced])]
# for i, c in enumerate(centers):
#     mask = cv2.inRange(labels, i, i)
#     mask = np.dstack([mask]*3) # Make it 3 channel
#     ex_img = cv2.bitwise_and(img, mask)
#     ex_reduced = cv2.bitwise_and(reduced, mask)
#     result.append(np.hstack([ex_img, ex_reduced]))

# cv2.imwrite('watermelon_out.jpg', np.vstack(result))
# hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
hsv=img
# # define ranges for colors in HSV color space you wish to display

## LIGHT AND DARK GREEN
lower_green_light = np.array([20,90,20])
upper_green_light = np.array([50,150,65])

## DARK GREEN
lower_green = np.array([0, 70, 50])
upper_green = np.array([170, 180, 100])

## RED
lower_red = np.array([170, 130, 0])
upper_red = np.array([180, 255, 255])

lw1=np.array([0, 0, 0])
uw1=np.array([2, 2, 2])

lg1=np.array([17, 0, 0])


# Threshold with inRange() get only specific colors
mask_green = cv2.inRange(hsv, lower_green, upper_green)
mask_green_light = cv2.inRange(hsv, lower_green_light, upper_green_light)
mask_red = cv2.inRange(hsv, lower_red, upper_red)

# Perform bitwise operation with the masks and original image
res_green = cv2.bitwise_and(img,img, mask= mask_green)
res_green_light = cv2.bitwise_and(img,img, mask= mask_green_light)
res_red = cv2.bitwise_and(img,img, mask= mask_red)

# Display results
cv2.imshow('red', res_red)
cv2.imshow('green', res_green)
cv2.imshow('light green', res_green_light)

            

#     print(color)
#     split_image = img.copy()
#     split_image[np.where(gray != color)] = 0
#     cv2.imwrite(str(img_number)+".jpg",split_image)
#     img_number+=1
# plt.hist(gray.ravel(),256,[0,256])
# plt.savefig('plt')
# plt.show()
# cv2.imshow('image',img)
cv2.imshow('image',img)
cv2.imwrite('watermelon_out.jpg',img)
# cv2.imshow('image',hist)
cv2.waitKey(0)
cv2.destroyAllWindows()