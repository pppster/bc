import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage
from rembg import remove
from bci.utils.bcimageenum import BCImageEnum, MASK, IMAGE, NOBACKGROUND
from bci.utils.image_utils import just_image_names


class BCImage:

    """Simple class for image processing for the bottle classification"""

    def __init__(self, path: str):
        if os.path.isfile(path):
            self.image_path = path
            self.load_image()
            self.name = self.get_name()
            self.label = self.get_label()

        else:
            print(f'There is no image file located at: {path}')
            self.image_path = ""
            self.image = np.ndarray([])
            self.name = ''
            self.label = ''

        if os.path.isfile(self.get_filename_mask()):
            self.mask_path = self.get_filename_mask()
            self.load_mask()

        else:
            print(f'There is no mask file located at: {self.get_filename_mask()}')
            self.mask_path = ''
            self.mask = np.ndarray([])

        if os.path.isfile(self.get_filename_nobackground()):
            self.no_background_path = self.get_filename_nobackground()
            self.load_no_background()

        else:
            print(f'There is no nobackground file located at: {self.get_filename_nobackground()}')
            self.no_background_path = ''
            self.no_background = np.ndarray([])

    def __str__(self):
        return f'Image located at {self.image_path}'

    def __getitem__(self, item: BCImageEnum):
        if item == IMAGE:
            return self.image
        elif item == MASK:
            return self.mask
        elif item == NOBACKGROUND:
            return self.no_background
        else:
            assert False, "There is no such BCImageType"

    def get_name(self) -> str:
        start, end = self.image_path.rfind('\\'), self.image_path.rfind('.')
        name = self.image_path[start+1:end]
        return name

    def get_label(self) -> str:
        return self.image_path.split('\\')[-2]

    def get_filename_mask(self) -> str:
        index_datatype = self.image_path.rfind('.')
        filename_mask = self.image_path[:index_datatype] + '_mask' + self.image_path[index_datatype:]
        return filename_mask

    def get_filename_nobackground(self) -> str:
        index_datatype = self.image_path.rfind('.')
        filename_nobackground = self.image_path[:index_datatype] + '_nobackground' + self.image_path[index_datatype:]
        return filename_nobackground

    def load_image(self) -> np.ndarray:
        image = plt.imread(self.image_path)
        if image.shape[0] < image.shape[1]:
            image = np.rot90(image, axes=(1, 0))
        self.image = image

    def load_mask(self) -> np.ndarray:
        mask = plt.imread(self.mask_path)
        if mask.shape[0] < mask.shape[1]:
            mask = np.rot90(mask, axes=(1, 0))
        self.mask = mask

    def load_no_background(self) -> np.ndarray:
        no_background = plt.imread(self.no_background_path)
        if no_background.shape[0] < no_background.shape[1]:
            no_background = np.rot90(no_background, axes=(1, 0))
        self.no_background = no_background

    def rename_image(self, new_name):
        old_name = self.image_path
        start, end = old_name.rfind('\\') + 1, old_name.rfind('.')
        updated_filename = old_name[:start] + new_name + old_name[end:]
        os.rename(self.image_path, updated_filename)
        self.path = updated_filename

    def show_image(self, image_type: BCImageEnum = IMAGE):
        plt.imshow(self[image_type])
        plt.show()

    def resize_image(self, height: int = 300):
        aspect_ratio = self.image.shape[1]/self.image.shape[0]
        width = int(np.round(aspect_ratio * height))
        self.image = cv2.resize(self.image, (width, height))

    def padding_image(self, aspect_ratio: int = 0.75):
        actual_width = self.image.shape[1]
        desired_width = aspect_ratio * self.image.shape[0]
        if actual_width == int(desired_width):
            print(f'No padding needed')
        else:
            difference_width = desired_width - actual_width
            left_border, right_border = int(np.floor(difference_width / 2)), int(np.ceil(difference_width / 2))
            if (left_border + actual_width + right_border) == desired_width:
                self.image = cv2.copyMakeBorder(self.image, 0, 0, left_border, right_border, cv2.BORDER_REPLICATE)
            else:
                print(f'Padding failed, new width would be: {(left_border + actual_width + right_border)}')

    def remove_background(self):
        self.no_background = cv2.cvtColor(remove(self.image), cv2.COLOR_BGRA2BGR)

    def generate_mask(self):
        image_grayscale = cv2.cvtColor(self.no_background, cv2.COLOR_BGR2GRAY)
        _, self.mask = cv2.threshold(image_grayscale,
                                     thresh=1,
                                     maxval=255,
                                     type=cv2.THRESH_BINARY)

    def save_image(self):
        plt.imsave(self.image_path, self.image)
        print(f"Image saved at: {self.image_path}")

    def save_mask(self):
        filename_mask = self.get_filename_mask()
        cv2.imwrite(filename_mask, self.mask)
        self.mask_path = filename_mask
        print(f"Mask saved at: {filename_mask}")

    def save_no_background(self):
        filename_nobackground = self.get_filename_nobackground()
        plt.imsave(filename_nobackground, self.no_background)
        self.no_background_path = filename_nobackground
        print(f"No background image saved at: {filename_nobackground}")




def preprocess_images_in_directory(path: str, height: int = 300, aspect_ratio: float = 0.75):
    image_paths = just_image_names(path)

    for image_path in image_paths:
        print(image_path)
        img = BCImage(image_path)
        if os.path.isfile(img.no_background_path) and os.path.isfile(img.mask_path):
            print('Already preprocessed')
        else:
            img.load_image()
            img.resize_image(height=height)
            img.padding_image(aspect_ratio=aspect_ratio)
            img.remove_background()
            img.generate_mask()
            img.save_mask()
            img.save_no_background()
            img.save_image()
            del img



if __name__ == '__main__':
    print('Just bcimage class')
    path = './../../images/_bottle_damaged/damaged/damaged_00000.jpeg'











