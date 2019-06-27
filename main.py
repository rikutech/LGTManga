import cv2
import numpy as np

origin_img = cv2.imread('./nino.jpg')
target_img = cv2.imread('./nino.jpg', 0)
target_img = cv2.Canny(target_img, 700, 1000)
contours, hierarchy = cv2.findContours(target_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = filter(lambda n:len(n) > 500, contours)
img = cv2.drawContours(origin_img, contours, -1, (0,255,0), 3)
cv2.imwrite('./result.jpg', img)
