import cv2

print(cv2.__version__)


import cv2
import numpy as np

# Reading an Image of  Shelf Store and Converting it into grayscale
im0 = cv2.imread("data/shelf4.jpg")
print('Original Dimensions : ', im0.shape)
scale_percent = 35  # percent of original size
width = int(im0.shape[1] * scale_percent / 100)
height = int(im0.shape[0] * scale_percent / 100)
dim = (width, height)
# resize image
im0 = cv2.resize(im0, dim, interpolation=cv2.INTER_AREA)
im = cv2.cvtColor(im0, cv2.COLOR_BGR2GRAY)
bilateral = cv2.bilateralFilter(im, 9, 75, 75)  # Using a bilateral filter since it better preserves the edges
edges = cv2.Canny(bilateral, 50, 150, apertureSize=3)
# sobelx = cv2.Sobel(edges, cv2.CV_64F, 1, 0, ksize=5)
laplacian = cv2.Laplacian(edges, cv2.CV_64F)  # Stronger edges through Laplacian

kernel1 = np.ones((17, 5), np.uint8)  # kernel is chosen to be rectangular, erosion is less towards horizontal direction and more towards vertical direction
eroded = cv2.erode(laplacian, kernel=kernel1, iterations=1)

# Normalizing the values of Eroded image
norm_image = cv2.normalize(eroded, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
norm_image_1 = norm_image.astype('uint8')

# FInding and Drawing the contours
im_cont = im0.copy()
im_final = im0.copy()
_, contours, hierarchy = cv2.findContours(norm_image_1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cntsSorted = sorted(contours, key=lambda x: cv2.contourArea(x),
                    reverse=True)  # Contours sorted in descending order by area
cntsSorted = cntsSorted[:3]  # print(cv2.contourArea(cntsSorted[2]))

for c in cntsSorted:
    x, y, w, h = cv2.boundingRect(c)
    print('The Rect coordinates of the', x, y, x + w, y + h)
    cv2.rectangle(im_final, (x, y), (x + w, y + h), (0, 0, 255), -1)

cv2.drawContours(im_cont, cntsSorted, -1, (0, 255, 255), thickness=3)

# Display intermediate and final images
cv2.imshow('laplacian', laplacian)
cv2.imshow('eroded', eroded)
cv2.imshow('norm_image', norm_image)
cv2.imshow('im_cont', im_cont)
cv2.imshow('im_final', im_final)
# cv2.imshow("Result", np.hstack([im0, im_final]))

cv2.waitKey(0)
cv2.destroyAllWindows()




