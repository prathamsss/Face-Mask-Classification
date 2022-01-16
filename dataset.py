import os
import requests, zipfile, io
import zipfile
import glob
from PIL import Image
import argparse
import shutil
def download_data(link, path_to_store):
    r = requests.get(link)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(path_to_store)
    print("Downloaded sucessfully")

def unzip(zip_file_path, directory_to_extract_to):
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(directory_to_extract_to)

    print("Given files extracted Successfully!")

def convert_to_png(img_dir):
    for img in glob.glob(img_dir+"/*.jpg"):
        if (img.split('/')[-2]) == (img_dir.split('/')[-1]):
            origial_img = Image.open(img)
            destination = os.path.join(
                                    img_dir, ((img.split('/')[-1]).split('.')[0])+'.png'
                                       )
            origial_img.save(destination)
            os.remove(img)



def create_dirtree_without_files(src, dst):
    src = os.path.abspath(src)
    src_prefix = len(src) + len(os.path.sep)
    # os.makedirs(dst)
    for root, dirs, files in os.walk(src):
        for dirname in dirs:
            dirpath = os.path.join(dst, root[src_prefix:], dirname)
            os.mkdir(dirpath)


def create_real_test_set(data_dir, real_test_data_path):
    l = len(glob.glob(data_dir + "/*.png"))
    for count, img in enumerate(glob.glob(data_dir + "/*.png")):
        shutil.copy(img, real_test_data_path)
        if count == round(l * 0.10):
            print("Done")
            break



if __name__ == '__main__':
    print("Welcome to Task-1 !")
    zip_file_url = "https://github.com/TheSSJ2612/Real-Time-Medical-Mask-Detection/releases/download/v0.1/Dataset.zip"


    data_directory = os.path.join(os.getcwd(), r'Dataset')         # Making new directory to store data
    if not os.path.exists(data_directory):
        print("Creating Required Directories..")
        os.makedirs(data_directory)
        download_data(zip_file_url, data_directory)                # Download Zip File and extract in given directory
    else:
        print("Dataset exists!")



    for i in os.listdir(data_directory):                        # Convert All imges to PNG files
        print("Converting images to .png files for:",i)
        convert_to_png(os.path.join(data_directory,i))

    try:
        os.remove(os.path.join(data_directory,".DS_Store"))
    except FileNotFoundError:
        pass

    print("Done Task-1  completed !")