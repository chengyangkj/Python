import cv2
from PIL import Image
import numpy as np
im = Image.open("../image/0.jpg").convert('L')
print(str(im.width),str(im.height))
#PIL对象转opencv
img = cv2.cvtColor(np.asarray(im),cv2.COLOR_RGB2BGR)
cv2.imshow("out",img)

cv2.waitKey(0)