from glob import glob
from bci.utils.image_utils import just_image_names
from bci.core.bcimage import BCImage
from bci.utils.bcimageenum import IMAGE
import cv2

dir = '../images/*/*/*'
images = just_image_names(dir)
print(images)

# c = 0
# for image in images:
#     print(image)
#     img = BCImage(image)
#     img.image = cv2.flip(img[IMAGE], 1)
#     img.rename_image(str(c)+'ffffffff')
#     img.save_image()
#     c += 1
#     del img
