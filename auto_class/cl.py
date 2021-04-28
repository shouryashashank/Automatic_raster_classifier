import numpy as np
import cv2
import matplotlib.pyplot as plt

original_image = cv2.imread("TI.tif")

img=cv2.cvtColor(original_image,cv2.COLOR_BGR2RGB)

vectorized = img.reshape((-1,3))

vectorized = np.float32(vectorized)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

K = 10
attempts=10
ret,label,center=cv2.kmeans(vectorized,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)

center = np.uint8(center)

res = center[label.flatten()]
res2 = res.reshape((img.shape))
all_rgb_codes = res2.reshape(-1, img.shape[-1])
unique_rgbs = np.unique(all_rgb_codes, axis=0)
print(unique_rgbs)
# figure_size = 15
# plt.figure(figsize=(figure_size,figure_size))
# plt.subplot(1,2,1),plt.imshow(img)
# plt.title('Original Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(1,2,2),plt.imshow(res2)
# plt.title('Segmented Image when K = %i' % K), plt.xticks([]), plt.yticks([])
# plt.show()
mi=0
for i in unique_rgbs:
    l=i
    u=i
    mask1=cv2.inRange(res2, l, u)
    out = cv2.bitwise_and(res2,res2, mask= mask1)
    cv2.imwrite("generated/"+str(mi)+".tif",out)
    mi=mi+1


# lw1=np.array([78,120,55])
# uw1=np.array([78,120,55])

# mask_red = cv2.inRange(res2, lw1, uw1)
# res_red = cv2.bitwise_and(res2,res2, mask= mask_red)
# cv2.imshow('red', res_red)
# cv2.imshow('res2', res2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()