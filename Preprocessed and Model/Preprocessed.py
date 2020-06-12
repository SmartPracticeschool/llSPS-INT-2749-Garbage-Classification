#Import The ImageDataGenerator Library
from keras.preprocessing.image import ImageDataGenerator

#Configure ImageDataGenerator Class
train_datagen = ImageDataGenerator(rescale =1./255, shear_range = 0.2, zoom_range=0.2, horizontal_flip =True)
test_datagen = ImageDataGenerator(rescale=1./255)

#Apply ImageDataGenerator Functionality To Trainset And Testset
x_train = train_datagen.flow_from_directory(r'E:\Smart bridge Internship\llSPS-INT-2749-Garbage-Classification\Data Collection\Train',target_size = (64,64),batch_size = 16, class_mode ="categorical")
x_test = test_datagen.flow_from_directory(r'E:\Smart bridge Internship\llSPS-INT-2749-Garbage-Classification\Data Collection\Test',target_size = (64,64),batch_size = 16, class_mode ="categorical")
print(x_train.class_indices)