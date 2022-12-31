from bci.core.bcimage import preprocess_images_in_directory

if __name__ == '__main__':
    path = '../images/_bottle_damaged/*/*'
    preprocess_images_in_directory(path)