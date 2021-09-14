# ---------------------------
# Define global imports
# ---------------------------

import numpy as np
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Conv2D, AveragePooling2D, UpSampling2D, Concatenate, concatenate
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import load_model
import os
import skimage
import skimage.io
import skimage.transform
import glob
import matplotlib.pyplot as plt
from IPython.display import clear_output
import cv2

LABEL_NAMES = np.asarray(['background', 'car'])
    
    

N_CLASSES = 3
CLASSES_TO_KEEP = [0,1]

# ---------------------------
# Define helper functions for inference/visualization.
# ---------------------------

def plot_history(histories, titles):
    plt.figure(dpi=300)
    for history, title in zip(histories, titles):
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        plt.plot(range(1, len(loss)+1), loss, label='Training Loss: ' + title)
        plt.plot(range(1, len(loss)+1), val_loss, label='Validation Loss: ' + title)
        plt.legend()
        plt.ylabel('Cross Entropy')
        plt.title('Training and Validation Loss')
        plt.xlabel('epoch')
        plt.grid(True)
        plt.savefig('foo.png', dpi=300)
    plt.show()




def norm_vis(img, mode='rgb'):
    img_norm = (img - img.min()) / (img.max() - img.min())
    return img_norm if mode == 'rgb' else np.flip(img_norm, axis=2)

def run_patch_predict(model, img, deeplab=False):
    """Runs the segmentation model on a single image patch (224 x 224) with flipping.

    Args:
        img: Input image of shape [B, H=224, W=224, C=3] with intensities within range [0,1].

    Returns:
        Segmentation prediction of shape [B, H=224, W=224, N_CLASSES=6].
    """
    img = img.copy() * 255.  # Renorm to [0, 255].
    if not deeplab:
        img = tf.keras.applications.mobilenet_v2.preprocess_input(img)  # Pre-process for MobileNetv2
    else:
        img /= 255.

    left = model.predict(img)
    flip = np.flip(model.predict(np.flip(img, axis=2)), axis=2)

    if deeplab:
        left = left[:, :, CLASSES_TO_KEEP]
        flip = flip[:, :, CLASSES_TO_KEEP]

    return (left + flip) / 2

def run_predict(model, img, step=3, deeplab=False):
    """Runs the segmentation model on a larger image.

    This specific procedure is quite arbitrary: it resizes the input image
    to 256 x 256 regardless of aspect ratio, and applies the network in a
    sliding-window fashion to combine multiple per-patch results.

    Args:
        img: Input image of shape [B, H, W, C=3] with intensities within range [0,1].
        step: Step size for the sliding window.

    Returns:
        Segmentation prediction of shape [B, H=256, W=256, N_CLASSES=6].
    """
    if img.shape[1] != 256 or img.shape[2] != 256:
        img_new = np.zeros(shape=(img.shape[0], 256, 256, img.shape[3]))
        for i in range(img.shape[0]):
            img_new[i] = cv2.resize(img[i], (256, 256), interpolation=cv2.INTER_LINEAR)  # Resize input image as needed.
        img = img_new

    canvas = np.zeros(shape=list(img.shape[:3]) + [N_CLASSES], dtype=np.float32)
    num_hits = np.zeros_like(canvas, dtype=np.int32)

    cx_probe = np.minimum(np.array(list(range(0, img.shape[2] - 224 + step, step))), img.shape[2] - 224)
    cy_probe = np.minimum(np.array(list(range(0, img.shape[1] - 224 + step, step))), img.shape[1] - 224)

    # Sliding-window patch
    for cx in cx_probe:
        for cy in cy_probe:
            patch = img[:, cy:cy+224, cx:cx+224]
            res = run_patch_predict(model, patch, deeplab=deeplab)

            # Combine results.
            canvas[:, cy:cy+224, cx:cx+224] += res
            num_hits[:, cy:cy+224, cx:cx+224] += 1
    return canvas / num_hits

def create_pascal_label_colormap():
    '''Creates a label colormap used in PASCAL VOC segmentation benchmark.

    Returns:
        A Colormap for visualizing segmentation results.
    '''
    colormap = np.zeros((256, 3), dtype=int)
    ind = np.arange(256, dtype=int)

    for shift in reversed(range(8)):
        for channel in range(3):
            colormap[:, channel] |= ((ind >> channel) & 1) << shift
        ind >>= 3

    return colormap

def label_to_color_image(label):
    """Adds color defined by the dataset colormap to the label.

    Args:
    label: A 2D array with integer type, storing the segmentation label.

    Returns:
    result: A 2D array with floating type. The element of the array
      is the color indexed by the corresponding element in the input label
      to the PASCAL color map.

    Raises:
    ValueError: If label is not of rank 2 or its value is larger than color
      map maximum entry.
    """
    if label.ndim != 2:
        raise ValueError('Expect 2-D input label')

    colormap = create_pascal_label_colormap()

    if np.max(label) >= len(colormap):
        raise ValueError('label value too large.')

    return colormap[label]

def vis_segmentation(image, seg_map):
    """Visualizes input image, segmentation map and overlay view."""
    plt.figure(figsize=(15, 5))
    grid_spec = gridspec.GridSpec(1, 4, width_ratios=[6, 6, 6, 1])

    plt.subplot(grid_spec[0])
    plt.imshow(image)
    plt.axis('off')
    plt.title('input image')

    plt.subplot(grid_spec[1])
    seg_image = label_to_color_image(seg_map).astype(np.uint8)
    plt.imshow(seg_image)
    plt.axis('off')
    plt.title('segmentation map')

    plt.subplot(grid_spec[2])
    plt.imshow(image)
    plt.imshow(seg_image, alpha=0.7)
    plt.axis('off')
    plt.title('segmentation overlay')

    unique_labels = np.unique(seg_map)
    ax = plt.subplot(grid_spec[3])
    plt.imshow(
      FULL_COLOR_MAP[unique_labels].astype(np.uint8), interpolation='nearest')
    ax.yaxis.tick_right()
    plt.yticks(range(len(unique_labels)), LABEL_NAMES[unique_labels])
    plt.xticks([], [])
    ax.tick_params(width=0.0)
    plt.grid('off')
    plt.show()

# ---------------------------
# Define custom data generators.
# ---------------------------

class CustomDataGenerator(tf.keras.utils.Sequence):
    """Custom data generator that yields tuples of (image, mask) for a pre-processed version of the Pascal VOC 2012 dataset."""
    def __init__(self, source_raw, source_mask, filenames, batch_size, target_height, target_width, augmentation=True, full_resolution=False):
        self.source_raw = source_raw
        self.source_mask = source_mask
        self.filenames = filenames
        self.batch_size = batch_size
        self.target_height = target_height
        self.target_width = target_width
        self.augmentation = augmentation
        self.full_resolution = full_resolution
        self.on_epoch_end()

    def on_epoch_end(self):
        '''Shuffle list of files after each epoch.'''
        np.random.shuffle(self.filenames)

    def __getitem__(self, index):
        cur_files = self.filenames[index*self.batch_size:(index+1)*self.batch_size]
        # Generate data
        X, y = self.__data_generation(cur_files)
        return X, y

    def __data_generation(self, cur_files):
        X = np.empty(shape=(self.batch_size, self.target_height, self.target_width, 3))
        Y = np.empty(shape=(self.batch_size, self.target_height, self.target_width, 1), dtype=np.int32)
        for i, file in enumerate(cur_files):
            img_raw = img_to_array(load_img(os.path.join(self.source_raw, file) + '.jpg', interpolation='bilinear', target_size=(256, 256)))

            # The preprocessing function varies by architecture.
            # e.g. for ResNet50, caffe-style preprocessing is used.
            # e.g. for MobileNetV2, tf-style preprocessing is used.
            img_raw = tf.keras.applications.mobilenet_v2.preprocess_input(img_raw)

            # General note: people sometimes accidentally use bilinear interpolation when resizing masks.
            # If you need to resize, make sure to use nearest neighbor interpolation only to avoid invalid class labels.
            img_mask = np.load(os.path.join(self.source_mask, file) + '.npy')
            img_mask= img_mask.astype(np.float32)
            img_mask = cv2.resize(img_mask, (256,256), interpolation=cv2.INTER_NEAREST)
            img_mask = img_mask.reshape(img_mask.shape[0],img_mask.shape[1],1)

            if self.augmentation:
                # Random cropping.
                crop_x = np.random.randint(img_raw.shape[1] - self.target_width)
                crop_y = np.random.randint(img_raw.shape[0] - self.target_height)
            else: # Take center crop instead.
                crop_x = (img_raw.shape[1] - self.target_width) // 2
                crop_y = (img_raw.shape[0] - self.target_height) // 2

            if not self.full_resolution:
                img_raw = img_raw[crop_y:crop_y+self.target_height, crop_x:crop_x+self.target_width]
                img_mask = img_mask[crop_y:crop_y+self.target_height, crop_x:crop_x+self.target_width]

            # Random flipping.
            perform_flip = np.random.rand(1) < 0.5
            if self.augmentation and perform_flip:
                img_raw = np.flip(img_raw, axis=1)
                img_mask = np.flip(img_mask, axis=1)

            X[i] = img_raw
            Y[i] = img_mask

        return X, Y

    def __len__(self):
        return int(np.floor(len(self.filenames) / self.batch_size))


class TeacherDataGenerator(tf.keras.utils.Sequence):
    """data generator that yields tuples of (image, (teacher_labels, true_labels))
    for a pre-processed version of the Pascal VOC 2012 dataset."""
    def __init__(self,
                    source_raw,
                    filenames,
                    batch_size,
                    target_height,
                    target_width,
                    augmentation=True,
                    full_resolution=False,
                    teacher_model=None,
                    source_mask=None,
                    classes_to_keep=None):

        self.source_raw = source_raw
        self.source_mask = source_mask
        self.filenames = filenames
        self.batch_size = batch_size
        self.target_height = target_height
        self.target_width = target_width
        self.augmentation = augmentation
        self.full_resolution = full_resolution
        self.teacher_model = teacher_model
        self.classes_to_keep = np.arange(0,21) if classes_to_keep is None else classes_to_keep

        self.on_epoch_end()


    def on_epoch_end(self):
        '''Shuffle list of files after each epoch.'''
        np.random.shuffle(self.filenames)

    def __getitem__(self, index):
        ''''get the next item of the generator'''
        cur_files = self.filenames[index*self.batch_size:(index+1)*self.batch_size]
        # Generate data
        X, y = self.__data_generation(cur_files)
        return X, y


    def __data_generation(self, cur_files):
        '''generates the input image and the labels: gt and teacher output.'''
        X_student = np.empty(shape=(self.batch_size, self.target_height, self.target_width, 3))
        X_teacher = np.empty(shape=(self.batch_size, self.target_height, self.target_width, 3))
        Y_true = np.empty(shape=(self.batch_size, self.target_height, self.target_width, 1), dtype=np.int32)

        for i, file in enumerate(cur_files):
            img_raw = img_to_array(load_img(os.path.join(self.source_raw, file) + '.jpg', interpolation='bilinear', target_size=(256, 256)))

            img_mobilenet = tf.keras.applications.mobilenet_v2.preprocess_input(img_raw)
            img_deeplab = img_raw/255.0

            # General note: people sometimes accidentally use bilinear interpolation when resizing masks.
            # If you need to resize, make sure to use nearest neighbor interpolation only to avoid invalid class labels.
            img_mask = np.load(os.path.join(self.source_mask, file) + '.npy')

            if self.augmentation:
                # Random cropping.
                crop_x = np.random.randint(img_raw.shape[1] - self.target_width)
                crop_y = np.random.randint(img_raw.shape[0] - self.target_height)
            else: # Take center crop instead.
                crop_x = (img_raw.shape[1] - self.target_width) // 2
                crop_y = (img_raw.shape[0] - self.target_height) // 2

            if not self.full_resolution:
                img_mobilenet = img_mobilenet[crop_y:crop_y+self.target_height, crop_x:crop_x+self.target_width]
                img_deeplab = img_deeplab[crop_y:crop_y+self.target_height, crop_x:crop_x+self.target_width]
                img_mask = img_mask[crop_y:crop_y+self.target_height, crop_x:crop_x+self.target_width]

            # Random flipping.
            perform_flip = np.random.rand(1) < 0.5
            if self.augmentation and perform_flip:
                img_mobilenet = np.flip(img_mobilenet, axis=1)
                img_deeplab = np.flip(img_deeplab, axis=1)
                img_mask = np.flip(img_mask, axis=1)

            X_student[i] = img_mobilenet
            X_teacher[i] = img_deeplab
            Y_true[i] = img_mask

        # We need the teacher's output to supervise the student network.
        # We pass the input X through the teacher network and save the result as our teacher labels.
        Y_teacher = self.teacher_model.predict(X_teacher)
        Y_teacher = Y_teacher[:,:,:, self.classes_to_keep]

        # We return both the true labels and the teacher's output. We use both
        # as supervision for the student model.
        return X_student, [Y_teacher, Y_true]

    def __len__(self):
        return int(np.floor(len(self.filenames) / self.batch_size))

FULL_LABEL_MAP = np.arange(len(LABEL_NAMES)).reshape(len(LABEL_NAMES), 1)
FULL_COLOR_MAP = label_to_color_image(FULL_LABEL_MAP)
