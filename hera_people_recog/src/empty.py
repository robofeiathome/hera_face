import cv2

img = cv2.imread('not.jpg')
print(img)
h, w, _ = img.shape
print(img.shape)
crop = img[:, :w//2]
cv2.imwrite('notcrop.jpg', crop)

