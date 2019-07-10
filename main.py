import cv2
import numpy as np
import itertools

def resize_img(img):
    width = 500
    scale = width / img.shape[1]
    return cv2.resize(img, dsize=None, fx=scale, fy=scale)

def childCount(hierarchy, index):
    target = hierarchy[index]
    childIndex = target[2]
    #next(itertools.ifilter(lambda x:len(x) >= 4, hierarchy), None)

origin_img = cv2.imread('./input.jpg')
origin_img = resize_img(origin_img)
width = origin_img.shape[1]
height = origin_img.shape[0]

target_img = cv2.imread('./input.jpg', 0)
target_img = resize_img(target_img)

blank_img = np.zeros((height, width, 3), np.uint8)

target_img = cv2.Canny(target_img, 200, 500)
cv2.imwrite('./target.jpg', target_img)
contours, hierarchy = cv2.findContours(target_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
hierarchy = hierarchy[0]
ls = list(filter(lambda n:len(n[0]) > 100, zip(contours, hierarchy)))
contours = list(map(lambda n:n[0], ls))
hierarchy = list(map(lambda n:n[1], ls))


for i in range(len(contours)):
    if hierarchy[i][3] <= 0 and cv2.contourArea(contours[i]) > 10000:
        cv2.drawContours(blank_img, contours, i, (255, 255, 255), thickness=2)
cv2.imwrite('./blank.jpg', blank_img)
blank_img = cv2.cvtColor(blank_img, cv2.COLOR_RGB2GRAY)
contours, hierarchy = cv2.findContours(blank_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
hierarchy = hierarchy[0]
ls = list(filter(lambda n:len(n[0]) > 100, zip(contours, hierarchy)))
contours = list(map(lambda n:n[0], ls))
hierarchy = list(map(lambda n:n[1], ls))

for i in range(len(contours)):
    if cv2.contourArea(contours[i]) > 10000:
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
