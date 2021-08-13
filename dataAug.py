import os
import warnings
warnings.filterwarnings("ignore")
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

def image_aug(filename, n_of_img=20):

    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    img = load_img(filename)
    x = img_to_array(img)
    x = x.reshape((1,) + x.shape)

    # augumentation
    directory = filename.split("/")[-1].split(".")[0]
    if not os.path.isdir(directory):
        os.mkdir(directory)

    i = 0
    for batch in datagen.flow(x, batch_size=1, save_to_dir=directory,save_prefix=directory, save_format="jpg"):
        i += 1
        if i > n_of_img:
            break

    print("Image Augumentation Done")
        

if __name__ == "__main__":
    image_aug("bike.jpg", 10)

