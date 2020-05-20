#!/usr/bin/env python
# coding: utf-8



#import the libraries
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten


# In[2]:


#initialise the model
model=Sequential()


# In[3]:


#add convolution 2D layer
model.add(Convolution2D(32,(3,3),input_shape = (64,64,3),activation = 'relu'))


# In[4]:


#add maxpooling layer
model.add(MaxPooling2D(pool_size = (2,2)))
#if activation fn is not mentioned defultly it is relu


# In[5]:


#add flatten layer
model.add(Flatten())
#output from flatten layer is input to ann


# In[6]:


#add hidden layer
model.add(Dense(output_dim = 128,init='uniform',activation='relu'))
#units or output_dim


# In[7]:


#add output layer
model.add(Dense(output_dim=1,activation = 'sigmoid',init='uniform'))


# In[8]:


from keras.preprocessing.image import ImageDataGenerator#gives augmentation(extra facilities to entities) to dataset
train_datagen = ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale = 1./255)


# In[9]:


#importing dataset
x_train = train_datagen.flow_from_directory(r'C:\Users\LENOVO\Downloads\Project\brain_tumor_dataset\train',target_size=(64,64),batch_size=32,class_mode='binary')
x_test = test_datagen.flow_from_directory(r'C:\Users\LENOVO\Downloads\Project\brain_tumor_dataset\test',target_size=(64,64),batch_size=32,class_mode='binary')


# In[10]:


#class indices
print(x_train.class_indices)
#done in alphabetical order


# In[11]:


#compiling the model
model.compile(loss = 'binary_crossentropy',optimizer = "adam",metrics = ["accuracy"])


# In[17]:


#fitting the model
model.fit_generator(x_train,steps_per_epoch = 10,epochs = 50,validation_data = x_test,validation_steps = 3)
#250 is train imagesdivided by batchsize
#for every 1 epoch 250 images are trained
#training nd testing takes place at a time
#for every epoch based on these 2 values accuracy is calculated
#for every epoch how many images need to be tested (63) i.e,testimages divides by batchsize
#for every 1 epoch 250 images should be trained and 63 images shud be tested


# In[18]:


model.save("brain.h5")
#h5 is used to save keras models


# In[ ]:




