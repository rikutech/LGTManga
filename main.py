import cv2
import numpy as np

origin_img = cv2.imread('./input.jpg')
lined_img  = cv2.imread('./input.jpg')
target_img = cv2.imread('./input.jpg', 0)
target_img = cv2.Canny(target_img, 700, 1000)
contours, hierarchy = cv2.findContours(target_img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
hierarchy = hierarchy[0]
ls = list(filter(lambda n:len(n[0]) > 500, zip(contours, hierarchy)))
contours = list(map(lambda n:n[0], ls))
hierarchy = list(map(lambda n:n[1], ls))

for i in range(len(contours)):
    cv2.drawContours(target_img, contours, i, (255, 255, 255), thickness=5)

cv2.imwrite('./target.jpg', target_img)
contours, hierarchy = cv2.findContours(target_img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
hierarchy = hierarchy[0]
ls = list(filter(lambda n:len(n[0]) > 300, zip(contours, hierarchy)))
contours = list(map(lambda n:n[0], ls))
hierarchy = list(map(lambda n:n[1], ls))

for i in range(len(contours)):
    if hierarchy[i][2] > -1:
        cv2.drawContours(origin_img, contours, i, (255, 255, 255), thickness=-1)

        moment = cv2.moments(contours[i])
        x = int(moment["m10"] / moment["m00"])
        y = int(moment["m01"] / moment["m00"])
        size = cv2.getTextSize("L", cv2.FONT_HERSHEY_PLAIN, 4, 5)
        text_width  = size[0][0]
        text_height = size[0][1]
        line_height = text_height + size[1]
        x -= int(text_width / 2)
        y -= line_height
        for char in ['L', 'G', 'T', 'M']:
            cv2.putText(origin_img, char, (x, y), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 0), 5)
            y += line_height

cv2.imwrite('./result.jpg', origin_img)
