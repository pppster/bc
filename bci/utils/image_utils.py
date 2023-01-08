from glob import glob

def just_image_names(dir: str) -> list:
    all_files = glob(dir)
    filtered = [file for file in all_files if "mask" not in file]
    filtered = [file for file in filtered if "nobackground" not in file]
    return sorted(filtered)



if __name__ == '__main__':
    print('Just bcimage util functions')