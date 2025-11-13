# OPENING--AND-CLOSING
## Aim
To implement Opening and Closing using Python and OpenCV.

## Software Required
1. Anaconda - Python 3.7
2. OpenCV
## Algorithm:
### Step1:
Import the necessary packages


### Step2:
Give the input text using cv2.putText()

### Step3:
Perform opening operation and display the result

### Step4:
Similarly, perform closing operation and display the result


 
## Program:

``` Python
import numpy as np
import cv2
import matplotlib.pyplot as plt
blank=np.zeros((600,600))
import numpy as np
import matplotlib.pyplot as plt

image = np.zeros((600, 800, 3), dtype=np.uint8)

plt.imshow(image)
plt.axis('on')
plt.show()

def load_image():
    blank_image = np.zeros((600,800))
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(blank_image,text='ARAVIND',org=(50,300),fontFace=font,fontScale = 5,color=(255,255,255),thickness=25,lineType=cv2.LINE_AA)
    return blank_image
def display_img(img):
    fig = plt.figure(figsize=(12,10))
    ax=fig.add_subplot(111)
    ax.imshow(img,cmap='gray')
    plt.show()
img=load_image()
kernel=np.ones((5,5),dtype=np.uint8)
white_noise=np.random.randint(low=0,high=2,size=(600,800))
white_noise = white_noise*255
noise_img = white_noise+img
display_img(noise_img)
opening = cv2.morphologyEx(noise_img, cv2.MORPH_OPEN, kernel)
display_img(opening)
black_noise = np.random.randint(low=0,high=2,size=(600,800))
black_noise= black_noise * -255
black_noise_img = img + black_noise
black_noise_img[black_noise_img==-255] = 0
display_img(black_noise_img)
closing = cv2.morphologyEx(black_noise_img, cv2.MORPH_CLOSE, kernel)
display_img(closing)
display_img(img)
gradient = cv2.morphologyEx(img,cv2.MORPH_GRADIENT,kernel)
display_img(gradient)

```
## Output:

### Display the input Image
<img width="932" height="696" alt="image" src="https://github.com/user-attachments/assets/357660de-0433-4b6b-9a8f-39224a0fb16e" />


### noise img:

<img width="957" height="682" alt="image" src="https://github.com/user-attachments/assets/736aec0c-442e-43a0-9a44-d0cac173f78b" />


### Display the result of Opening

<img width="652" height="465" alt="image" src="https://github.com/user-attachments/assets/2d1b8b36-516a-4070-95d8-9873520221b2" />



### black noise img:


<img width="617" height="461" alt="image" src="https://github.com/user-attachments/assets/26005841-b4f9-49ab-a3af-9ea608eb496f" />


### Display the result of Closing

<img width="613" height="463" alt="image" src="https://github.com/user-attachments/assets/957cf5ec-fddd-43e5-8488-a33c8f5f11d0" />



## Result
Thus the Opening and Closing operation is used in the image using python and OpenCV.
