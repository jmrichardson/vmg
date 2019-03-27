import numpy
import os
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
import PIL

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

files_train = 0
files_validation = 0

# Get training files
folder = 'data/train'
for sub_folder in os.listdir(folder):
    path, dirs, files = next(os.walk(os.path.join(folder, sub_folder)))
    files_train += len(files)

# Get test files
folder = 'data/test'
for sub_folder in os.listdir(folder):
    path, dirs, files = next(os.walk(os.path.join(folder, sub_folder)))
    files_validation += len(files)

# print(files_train,files_validation)

# Set parameters
# img_width, img_height = 210, 380
# img_width, img_height = 54, 95
# img_width, img_height = 32, 32
img_width, img_height = 105, 190
train_data_dir = "data/train"
validation_data_dir = "data/test"
nb_train_samples = files_train
nb_validation_samples = files_validation
batch_size = 32
epochs = 30
num_classes = 2

# Get VGG16 model
model = applications.VGG16(weights = "imagenet", include_top=False, input_shape = (img_height, img_width, 3))
# model = applications.MobileNetV2(weights = "imagenet", include_top=False, input_shape = (img_height, img_width, 3))
# model = applications.ResNet50(weights = "imagenet", include_top=False, input_shape = (img_height, img_width, 3))
# model = applications.InceptionResNetV2(weights = "imagenet", include_top=False, input_shape = (img_height, img_width, 3))
# model = applications.InceptionV3(weights = "imagenet", include_top=False, input_shape = (img_height, img_width, 3))
# model = applications.InceptionResNetV2(weights = "imagenet", include_top=False, input_shape = (img_height, img_width, 3))
# model = applications.Xception(weights = "imagenet", include_top=False, input_shape = (img_height, img_width, 3))
# model = applications.DenseNet121(weights = "imagenet", include_top=False, input_shape = (img_height, img_width, 3))
# model = applications.DenseNet169(weights = "imagenet", include_top=False, input_shape = (img_height, img_width, 3))
# model = applications.DenseNet201(weights = "imagenet", include_top=False, input_shape = (img_height, img_width, 3))

x = model.output
x = Flatten()(x)
predictions = Dense(num_classes, activation="softmax")(x)
# creating the final model
model_final = Model(inputs = model.input, outputs = predictions)

# compile the model
model_final.compile(loss = "categorical_crossentropy",
                    optimizer = optimizers.SGD(lr=0.0001, momentum=0.9),
                    metrics=["accuracy"]) 

# Initiate the train and test generators with data Augumentation
train_datagen = ImageDataGenerator(
rescale = 1./255,
horizontal_flip = True,
fill_mode = "nearest",
zoom_range = 0.1,
width_shift_range = 0.1,
height_shift_range=0.1,
rotation_range=5)

test_datagen = ImageDataGenerator(
rescale = 1./255,
horizontal_flip = True,
fill_mode = "nearest",
zoom_range = 0.1,
width_shift_range = 0.1,
height_shift_range=0.1,
rotation_range=5)

train_generator = train_datagen.flow_from_directory(
train_data_dir,
target_size = (img_height, img_width),
batch_size = batch_size,
class_mode = "categorical")

validation_generator = test_datagen.flow_from_directory(
validation_data_dir,
target_size = (img_height, img_width),
class_mode = "categorical")


checkpoint = ModelCheckpoint("spots.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto')

# Train model
history_object = model_final.fit_generator(
train_generator,
samples_per_epoch = nb_train_samples,
epochs = epochs,
validation_data = validation_generator,
validation_steps = nb_validation_samples,
callbacks = [checkpoint, early])


