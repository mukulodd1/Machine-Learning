import os
import cv2
import subprocess
import shutil




def clr():
    import os
    import subprocess
    import shutil
    dirs= os.listdir('downloads')
    for i in dirs:
        path= 'downloads\\'+ i
        shutil.rmtree(path)




print(os.getcwd())
func.clr()
print("Please Enter Two strings : ",end='')
x,y= input().split()

subprocess.call('googleimagesdownload -k "%s,%s" -s medium -l 100'%(x,y),shell=True)
print("The process is nowa completed")



dirc = os.path.join(os.getcwd(),'downloads')
lst= os.listdir(dirc)
leng=[]
for folder_name in lst:
    dir = os.path.join(dirc,folder_name)
    loc = os.listdir(dir)
    images=[]
    for image in loc:
            images.append(cv2.imread(os.path.join(dir,image)))
    os.mkdir(r'downloads\new'+folder_name)
    j=0
    for i in range(len(images)):
        try:
            images[i]=cv2.resize(images[i],(200,200))
            cv2.imwrite(r'downloads\new'+folder_name+"\\" + str(j)+".jpg",images[i])
            j=j+1
        except:

            pass

    print(len(images))
    leng.append(len(images))

train_size = int(min(leng[0],leng[1])*.75)
test_size  = int(min(leng[0],leng[1])*.25)


os.mkdir(r'downloads\train')
os.mkdir(r'downloads\test')

os.mkdir('downloads\\train\\'+lst[0])
os.mkdir('downloads\\train\\'+lst[1])


os.mkdir('downloads\\test\\'+lst[0])
os.mkdir('downloads\\test\\'+lst[1])



for filename in lst:
    for i in range(0,train_size+test_size):
        try:
            image = cv2.imread("downloads\\new"+filename+"\\"+str(i)+".jpg")
            if(i <= test_size):
                cv2.imwrite(r'downloads\\test\\'+filename+"\\"+str(i)+".jpg",image)
            else:
                cv2.imwrite(r'downloads\\train\\' + filename + "\\" + str(i) + ".jpg", image)
        except:
            pass







## This is the keras model for image classifier
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.preprocessing import image
import cv2
import numpy as np

img_width, img_height = 200,200
train_data_dir = 'downloads\\train'
validation_data_dir = 'downloads\\test'
nb_train_samples = train_size
nb_validation_samples = test_size
epochs = 10
batch_size = 10


if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)


model = Sequential()
model.add(Conv2D(32, (2, 2), input_shape = input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size =(2, 2)))

model.add(Conv2D(32, (2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size =(2, 2)))

model.add(Conv2D(64, (2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size =(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))


model.compile(loss ='binary_crossentropy',
                     optimizer ='rmsprop',
                   metrics =['accuracy'])




train_datagen = ImageDataGenerator(
                rescale = 1. / 255,
                 shear_range = 0.2,
                  zoom_range = 0.2,
            horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1. / 255)

train_generator = train_datagen.flow_from_directory(train_data_dir,
                              target_size =(img_width, img_height),
                     batch_size = batch_size, class_mode ='binary')

validation_generator = test_datagen.flow_from_directory(
                                    validation_data_dir,
                   target_size =(img_width, img_height),
          batch_size = batch_size, class_mode ='binary')


model.fit_generator(train_generator,
    steps_per_epoch = nb_train_samples // batch_size,
    epochs = epochs, validation_data = validation_generator,
    validation_steps = nb_validation_samples // batch_size)

model.save_weights('model_saved.h5')
#Image test
test_image =  cv2.imread('Lion.jpg')
test_image = cv2.resize(test_image,(200,200))
print(test_image.shape)
test_image=np.expand_dims(test_image,axis=0)
result = model.predict(test_image, batch_size=1)
print(result)
