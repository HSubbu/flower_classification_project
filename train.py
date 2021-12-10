# tensorflow libraries
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
import tensorflow_hub as hub
#support libraries
import numpy as np
import pandas as pd


#indicate path of train and val folder with individual folder containing respective flower names
train_directory = '/content/flowers/flowers/train'
val_directory = '/content/flowers/flowers/validation'

# create a function to get train , validation 
train_image_generator = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
val_image_generator = ImageDataGenerator(rescale = 1./255)
def get_data(IMAGE_SHAPE,batch_size=32):
  ''' This function takes image shape and batch size(default =32) and creates train and val datagen from directory'''
  #make train data generator
  train_data_gen = train_image_generator.flow_from_directory(directory=train_directory, batch_size=batch_size,class_mode='categorical',
                                                        target_size=(IMAGE_SHAPE[0], IMAGE_SHAPE[1]))
  # make the validation dataset generator
  val_data_gen = val_image_generator.flow_from_directory(directory=val_directory, batch_size=batch_size, class_mode='categorical',
                                                         target_size=(IMAGE_SHAPE[0], IMAGE_SHAPE[1]))
  return train_data_gen,val_data_gen

train,val = get_data(IMAGE_SHAPE=(224,224),batch_size=32) # train and validation dataset

#create mobilenet model
def create_mobilenet_model_with_dropout():
  ''' this function creates a mobilenet model with layers are not trainable'''
  # define a feature extractor from tf hub
  feature_extractor = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
  # using the above feature extractor define a feature_extractor_layer
  feature_extractor_layer = hub.KerasLayer(feature_extractor,input_shape=(224,224,3))
  # assign trainable as false
  feature_extractor_layer.trainable=False
  # define a keras sequential model 
  mobilenet_model = tf.keras.Sequential()
  mobilenet_model.add(feature_extractor_layer) # mobilenet forms the initial layer of sequential model
  mobilenet_model.add(keras.layers.Dropout(0.3)) # dropout f 0.3 is used (again can be iterated for optimum)
  mobilenet_model.add(keras.layers.Dense(5,activation='softmax')) # final layer with 5 outputs
  return mobilenet_model

mobilenet_v2 = create_mobilenet_model_with_dropout() # creating the model
#compile mobilenet model 
mobilenet_v2.compile(
    loss = keras.losses.CategoricalCrossentropy(),
    optimizer = keras.optimizers.Adam(),
    metrics = ['accuracy']
)
# define callbacks
earlyStopping = EarlyStopping(monitor='val_accuracy', patience=10, verbose=0, mode='max')
mcp_save = ModelCheckpoint('mobilenet_v2_{epoch:02d}_{val_accuracy:.3f}.h5', save_best_only=True, monitor='val_accuracy', mode='max')
reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, min_delta=1e-4, mode='min')#Reduce learning rate when a metric has stopped improving
# fit the mobilenet model on train dataset
history = mobilenet_v2.fit(train,epochs=50,validation_data=val,callbacks=[earlyStopping, mcp_save, reduce_lr_loss]) 

mobilenet_v2.save('mobilenet_final') #save final model 