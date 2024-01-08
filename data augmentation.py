
import matplotlib.pyplot as plt
from keras.preprocessing import image
import keras
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
#creates a data generator object that transforms images
data = keras.datasets.cifar10
(train_img, train_label), (test_img, test_label) =data.load_data()
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

#pick an image to transform
test_img = Image.open('images.jpg')
img=image.img_to_array(test_img)
print(img.shape)
img=img.reshape((1,)+img.shape)#this adds 1 beginning for ex:(1,64,64)
print(f"shape of the img is {img.shape}" )

i=0

for batch in datagen.flow(img,save_to_dir="augmented images",save_prefix='test',save_format='jpeg'):
    plt.figure(i)##adds the figures
    plot = plt.imshow(image.img_to_array(batch[0]))# batch[0] refers to the first element of the generated batch of augmented images.
    print(f"batch 0 is {batch[0]}")
    i += 1
    if i > 4:
        break

plt.show()#shows all figures one by one
