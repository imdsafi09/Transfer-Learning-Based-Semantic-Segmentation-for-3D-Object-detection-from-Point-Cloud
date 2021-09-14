
import sys
sys.path.remove('/opt/ros/melodic/lib/python2.7/dist-packages') # in order to import cv2 under python3
import cv2
sys.path.append('/opt/ros/melodic/lib/python2.7/dist-packages') # append back in order to import rospy
import helpers
from helpers import *

from matplotlib.colors import ListedColormap
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import tensorflow

from tensorflow import keras
#from plot_model import plot_model
cmap_ref = cm.get_cmap('tab20', 7)
cmap_seg = np.zeros((6, 4))
cmap_seg[0] = [0.5, 0.5, 0.5, 0]
for i in range(1, 6):
  cmap_seg[i] = cmap_ref(i)

cmap_seg = ListedColormap(cmap_seg)
print("\nClasses to detect, with corresponding colors:")
plt.imshow([[0, 1]], cmap=cmap_seg)
plt.xticks([0,1], LABEL_NAMES[[0,1]], rotation=45)

INPUT_SPATIAL = 224

source_raw = 'JPEGImages'
source_mask = 'SegmentationClassSubset'

with open('ImageSets/Segmentation/train.txt', 'r') as fp:
    files_train = [line.rstrip() for line in fp.readlines()]

with open('ImageSets/Segmentation/val.txt', 'r') as fp:
    files_val = [line.rstrip() for line in fp.readlines()]

# Filter down to the subset we are using.
files_train = [f for f in files_train if os.path.isfile(os.path.join('SegmentationClassSubset/' + f + '.npy'))]
files_val = [f for f in files_val if os.path.isfile(os.path.join('SegmentationClassSubset/' + f + '.npy'))]

# Split train-validation into 80:20 instead of the original split.
files_all = np.array(sorted(list(set(files_train).union(set(files_val)))))
index = int(len(files_all) * 0.8)
files_train = files_all[:index]
files_val = files_all[index:]
print(len(files_train), 'training', len(files_val), 'validation')
labels = ['background', 'car','obstacle']

gen_train = CustomDataGenerator(source_raw=source_raw,
                                source_mask=source_mask,
                                filenames=files_train.copy(),
                                batch_size=2,
                                target_height=INPUT_SPATIAL,
                                target_width=INPUT_SPATIAL)

gen_val = CustomDataGenerator(source_raw=source_raw,
                              source_mask=source_mask,
                              filenames=files_val.copy(),
                              batch_size=2,
                              target_height=INPUT_SPATIAL,
                              target_width=INPUT_SPATIAL)


# In[53]:


X, Y = gen_train[0]
X = X[0]
Y = Y[0]

plt.figure(figsize=(9, 4))
plt.subplot(1, 2, 1)
plt.imshow(norm_vis(X, mode='rgb'))
plt.subplot(1, 2, 2)
plt.imshow(Y[:, :, 0])
plt.show()

print('X shape', X.shape, 'min-mean-max', X.min(), X.mean(), X.max())
print('Y shape', Y.shape, 'min-mean-max', Y.min(), Y.mean(), Y.max())

K.clear_session()

def get_fcn(pretrained=True, add_activation=True, verbose=False, n_outputs=1):
    def conv_block_simple(prev, num_filters, name):
        return Conv2D(num_filters, activation='relu', kernel_size=(3, 3), padding='same', name=name + '_3x3')(prev)

    selected_encoder = tf.keras.applications.Xception(input_shape=(INPUT_SPATIAL, INPUT_SPATIAL, 3),include_top=False,weights='imagenet')
         

    conv0 = selected_encoder.get_layer("expanded_conv_project").output # 112 x 112
    conv1 = selected_encoder.get_layer("block_2_project").output # 56 x 56
    conv2 = selected_encoder.get_layer("block_5_project").output # 28 x 28
    conv3 = selected_encoder.get_layer("block_12_project").output # 14 x 14
   
    up6 = selected_encoder.output 
    conv7 = up6

    up8 = concatenate([UpSampling2D()(conv7), conv3], axis=-1)
    conv8 = conv_block_simple(up8, 128, "conv8_1")

    up9 = concatenate([UpSampling2D()(conv8), conv2], axis=-1)
    conv9 = conv_block_simple(up9, 64, "conv9_1")

    up10 = concatenate([UpSampling2D()(conv9), conv1], axis=-1)
    conv10 = conv_block_simple(up10, 32, "conv10_1")

    up11 = concatenate([UpSampling2D()(conv10), conv0], axis=-1)
    conv11 = conv_block_simple(up11, 32, "conv11_1")

    up12 = UpSampling2D()(conv11)
    conv12 = conv_block_simple(up12, 32, "conv12_1")

    x = Conv2D(N_CLASSES, (1, 1), activation=None, name="prediction")(conv12)
    
    if add_activation:
      x = tf.keras.layers.Activation("softmax")(x)

    model = tf.keras.Model(selected_encoder.input, [x] * n_outputs)
    if verbose:
        model.summary()
    return model








model_pretrained = get_fcn(pretrained=True, verbose=False)
model_pretrained.compile(optimizer=tf.keras.optimizers.Adam(lr=0.00005), loss=masked_loss)

print('total number of model parameters:', model_pretrained.count_params())


history_pretrained = model_pretrained.fit(gen_train, epochs=30, verbose=1, validation_data=gen_val)


model_pretrained.save('model.h5')







