import os
import glob
import random
from PIL import Image

def get_images_from_directory(directory, num_images):
    image_paths = []
    for root, dirs, files in os.walk(directory):
        if "0_real" in dirs and "1_fake" in dirs:
            real_dir = os.path.join(root, "0_real")
            fake_dir = os.path.join(root, "1_fake")
            image_paths.extend(random.sample(glob.glob(os.path.join(real_dir, "*.jpg")), num_images))
            image_paths.extend(random.sample(glob.glob(os.path.join(fake_dir, "*.jpg")), num_images))
        else:
            for d in dirs:
                if d.endswith("_real") or d.endswith("_fake"):
                    real_dir = os.path.join(root, d, "0_real")
                    fake_dir = os.path.join(root, d, "1_fake")
                    image_paths.extend(random.sample(glob.glob(os.path.join(real_dir, "*.jpg")), num_images))
                    image_paths.extend(random.sample(glob.glob(os.path.join(fake_dir, "*.jpg")), num_images))
    
    images = []
    for path in image_paths:
        image = Image.open(path)
        images.append(image)
    
    return images

# Usage example
dataset_dir = "dataset/test"
num_images_per_directory = 3

images = get_images_from_directory(dataset_dir, num_images_per_directory)

# Do something with the images (e.g., display or process them)
for image in images:
    image.show()
