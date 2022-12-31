import os
from glob import glob


def get_datatype(path: str):
    return path[path.rfind('.'):]


def change_filename(old: str, new: str):
    start, end = old.rfind('/')+1, old.rfind('.')
    updated_filename = old[:start] + new + old[end:]
    return updated_filename

def get_filename(image_path: str) -> str:
    start, end = image_path.rfind('/') + 1, image_path.rfind('.')
    return image_path[start:end]

def get_last_image(images: list, label: str) -> int:
    last_image = 0
    for image in images:
        filename = get_filename(image)
        if label in filename:
            if 'mask' not in filename:
                if 'nobackground' not in filename:
                    image_number = int(filename[-5:])
                    if image_number > last_image:
                        last_image = image_number

    return last_image

if __name__ == '__main__':
    ABSOLUTE_PATH = os.path.dirname(__file__)
    CLASSIFICATION_PATH = os.path.join(ABSOLUTE_PATH,'images','*')
    classifications = glob(CLASSIFICATION_PATH)
    for classification in classifications:
        labels = os.listdir(classification)
        for label in labels:
            images = glob(os.path.join(classification, label,'*'))
            image_counter = get_last_image(images, label) + 1
            for image in images:
                if label not in get_filename(image):
                    new_filename = f'{label}_{image_counter:05}'
                    changed_filename = change_filename(image, new_filename)
                    print(image)
                    print(changed_filename)
                    os.rename(image, changed_filename)
                    image_counter += 1


















