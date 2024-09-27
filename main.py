
import numpy as np
import glob
import cv2
import keras
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator


images = sorted(glob.glob('./archive/Image/*.jpg'))
masks = sorted(glob.glob('./archive/Mask/*.png'))



class CustomDataGenerator(keras.utils.Sequence):
    def __init__(self, images, masks, batch_size=8, img_size=(512, 512), shuffle=True):
        self.batch_size = batch_size
        self.img_size = img_size
        self.shuffle = shuffle
        self.images = images #os.listdir(image_folder)
        self.masks = masks #os.listdir(mask_folder)

        # on each epoch end, shuffle the dataset
        self.on_epoch_end()

        # datagen function to augment the input image and mask pair
        self.datagen = ImageDataGenerator(
            rotation_range=5,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.05,
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode = 'constant',
            cval=0.0,
        )

        # randomly crop the images to 512x512 size
    def random_crop(self, image, mask, crop_size=512):
            # image width and height calculation
            img_height, img_width = image.shape[0], image.shape[1]
            mask_height, mask_width = mask.shape[0], mask.shape[1]

            # random x and y coordinate for cropping the image
            x = np.random.randint(0, img_width - crop_size)
            y = np.random.randint(0, img_height - crop_size)

            # random crop
            image_crop = image[y:y + crop_size, x:x + crop_size, :]
            mask_crop = mask[y:y + crop_size, x:x + crop_size]

            return image_crop, mask_crop

        # data augmentation using keras ImageDataGenerator function
    def data_augmentation(self, image, mask):
            trans_param = self.datagen.get_random_transform(image.shape)
            image = self.datagen.apply_transform(image, trans_param)
            mask = self.datagen.apply_transform(mask, trans_param)
            return image, mask
            # length of the processing batch

    def __len__(self):
            return int(np.ceil(len(self.images) / self.batch_size))

            # data normalization

    def data_normalization(self, image, mask):
            # reshape mask from 512x512 to 512x512x1
            mask = mask.reshape((*self.img_size, 1))

            # Binary mask
            mask = np.where(mask < 127, 0, 1)

            # data normalization (If you want to normalize another way, change the below line)
            image = image / 255.0

            # return image and mask
            return image, mask

            # data preprocessing, resize, crop image etc

    def data_preprocessing(self, image, mask):
            image, mask = cv2.resize(image, (576, 576)), cv2.resize(mask, (576, 576))
            image, mask = self.random_crop(image, mask)
            return image, mask
 # on each epoch, shuffle the dataset (image and mask index)
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.images))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    # get item is the core function
    # this function will run in each batch/epoch to load the dataset into RAM and pass to DL model
    def __getitem__(self, index):

        # start and end index
        # the last index can be shorter than the number of batches
        start_idx = index * self.batch_size
        end_idx = min((index + 1) * self.batch_size, len(self.images))
        indexes = self.indexes[start_idx:end_idx]

        # initialize the images and mask batches
        batch_images = []
        batch_masks = []
 # iterate over each indexes in batch
        for i in indexes:
            img_path = self.images[i]
            mask_path = self.masks[i]

            # read image using open cv
            img = cv2.imread(img_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            # Skip if image or mask is not loaded properly
            if img is None or mask is None:
                continue

            # image preprocessing; resize, random crop
            img, mask = self.data_preprocessing(img, mask)

            # data normalization
            img, mask = self.data_normalization(img, mask)

            # data augmentation
            img, mask = self.data_augmentation(img, mask)

            # to fix the issue during training process
            mask = mask.astype(np.float32)

            # append each image, mask pair to the batches
            batch_images.append(img)
            batch_masks.append(mask)

        # return batch image and batch mamks as a numpy array (n, tile_x, tile_y, channels)
        return np.array(batch_images), np.array(batch_masks)
    
    # Testing and visualization of image/mask pair
data = CustomDataGenerator(images, masks)
batch_images, batch_masks = data.__getitem__(0)

img = np.random.randint(0, 8)
# Visualize the first image and its mask from the batch
image = batch_images[img]
mask = batch_masks[img]

# Plotting the image and its mask
plt.figure(figsize=(10, 5))

# Display Image
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title('Image')
plt.axis('off')

# Display Mask
plt.subplot(1, 2, 2)
plt.imshow(mask)
plt.title('Mask')
plt.axis('off')

plt.tight_layout()
plt.show()

# Display Mask
plt.subplot(1, 2, 2)
plt.imshow(mask)
plt.title('Mask')
plt.axis('off')

plt.tight_layout()
plt.show()


# Train and test data set splits
# train_img, test_img, train_mask, test_mask = train_test_split(images, masks, test_size=0.2, random_state=42)