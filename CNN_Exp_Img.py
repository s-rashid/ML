'''
This is trial 1
It uses single cnn layer one Max Pool  and singe dense layer
'''
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, utils
from tensorflow.keras.optimizers import Adam
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing import image
import tensorflow.keras.backend as K


(x_train, y_train), (x_test,y_test) = datasets.cifar10.load_data()
y_train = y_train.reshape(-1,)
y_test = y_test.reshape(-1,)
classes = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
x_train = tf.keras.utils.normalize(x_train)
x_test = tf.keras.utils.normalize(x_test)


cnn = models.Sequential([
    layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    
    
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

cnn.compile(optimizer=Adam(learning_rate=0.0001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("\nThe is the model plot: \n")
tf.keras.utils.plot_model(cnn, to_file='./trial1.png', show_shapes=True)
print("\n")
print("\nThe is the history for 15 epochs: \n")
cnn.fit(x_train, y_train, epochs=15)
print("\n")
print("\nThis is the evaluation.")
cnn.evaluate(x_test,y_test)


# In[2]:


layer_names = ['block2_conv1','max_pooling2d_1']
def deprocess_image(x):
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1
    x += 0.5
    x = np.clip(x, 0, 1)
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def generate_pattern(layer_name, filter_index, size=150):

    layer_output = cnn.get_layer(layer_name).output
    loss = K.mean(layer_output[:, :, :, filter_index])
    grads = K.gradients(loss, model.input)[0]
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
    iterate = K.function([model.input], [loss, grads])
    input_img_data = np.random.random((1, size, size, 3)) * 20 + 128.
    step = 1.
    for i in range(40):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step
        
    img = input_img_data[0]
    return deprocess_image(img)

def get_activations(img, model_activations):
    img = image.load_img(img, target_size=(32, 32))
    img = image.img_to_array(img)                    
    img = np.expand_dims(img, axis=0)         
    img /= 255. 
    #plt.imshow(img[0])
    plt.show()
    return model_activations.predict(img)

def show_activations(activations, layer_names):
    
    images_per_row = 9
    for layer_name, layer_activation in zip(layer_names, activations):
        n_features = layer_activation.shape[-1]
        size = layer_activation.shape[1]
        #n_cols = n_features // images_per_row
        n_cols = 1
        display_grid = np.zeros((size * n_cols, images_per_row * size))
        for col in range(n_cols):
            for row in range(images_per_row):
                channel_image = layer_activation[0, :, :, col * images_per_row + row]
                channel_image -= channel_image.mean()
                channel_image /= (channel_image.std()+0.01)
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                display_grid[col * size : (col + 1) * size, row * size : (row + 1) * size] = channel_image

        scale = 1. / size
        plt.figure(figsize=(scale * display_grid.shape[1],
                            scale * display_grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')

    plt.show()
    

layers = [layer.output for layer in cnn.layers[:9]]
activations_output = tf.keras.models.Model(inputs=cnn.input, outputs=layers)

img = "./cat.jpg"
activations = get_activations(img,activations_output)


show_activations(activations, layer_names)


# This is trail 2. In this trial 2 cnn layers and one dense layer have been used. Data was processed by normalization in tensorflow.kears.utils.

# In[3]:


'''
This is trial 2
Using two cnn layer 
'''
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, utils
from tensorflow.keras.optimizers import Adam
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.keras.backend as K
from keras.preprocessing import image

(x_train, y_train), (x_test,y_test) = datasets.cifar10.load_data()
y_train = y_train.reshape(-1,)
y_test = y_test.reshape(-1,)
classes = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
x_train = tf.keras.utils.normalize(x_train)
x_test = tf.keras.utils.normalize(x_test)
#x_train, y_train = shuffle(x_train, y_train)

cnn = models.Sequential([
    layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

cnn.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("\nThe is the model plot: \n")
tf.keras.utils.plot_model(cnn, to_file='./trial2.png', show_shapes=True)
print("\n")
print("\nThe is the history for 15 epochs: \n")
cnn.fit(x_train, y_train, epochs=15)
print("\n")
print("\nThis is the evaluation.")
cnn.evaluate(x_test,y_test)


layer_names = ['block2_conv1', 'max_pooling2d_1', 'conv2d_2', 'max_pooling2d_2']
def deprocess_image(x):
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1
    x += 0.5
    x = np.clip(x, 0, 1)
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def generate_pattern(layer_name, filter_index, size=150):

    layer_output = cnn.get_layer(layer_name).output
    loss = K.mean(layer_output[:, :, :, filter_index])
    grads = K.gradients(loss, model.input)[0]
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
    iterate = K.function([model.input], [loss, grads])
    input_img_data = np.random.random((1, size, size, 3)) * 20 + 128.
    step = 1.
    for i in range(40):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step
        
    img = input_img_data[0]
    return deprocess_image(img)

def get_activations(img, model_activations):
    img = image.load_img(img, target_size=(32, 32))
    img = image.img_to_array(img)                    
    img = np.expand_dims(img, axis=0)         
    img /= 255. 
    #plt.imshow(img[0])
    plt.show()
    return model_activations.predict(img)

def show_activations(activations, layer_names):
    
    images_per_row = 9
    for layer_name, layer_activation in zip(layer_names, activations):
        n_features = layer_activation.shape[-1]
        size = layer_activation.shape[1]
        #n_cols = n_features // images_per_row
        n_cols = 1
        display_grid = np.zeros((size * n_cols, images_per_row * size))
        for col in range(n_cols):
            for row in range(images_per_row):
                channel_image = layer_activation[0, :, :, col * images_per_row + row]
                channel_image -= channel_image.mean()
                channel_image /= (channel_image.std()+0.01)
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                display_grid[col * size : (col + 1) * size, row * size : (row + 1) * size] = channel_image

        scale = 1. / size
        plt.figure(figsize=(scale * display_grid.shape[1],
                            scale * display_grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')

    plt.show()
    

layers = [layer.output for layer in cnn.layers[:9]]
activations_output = tf.keras.models.Model(inputs=cnn.input, outputs=layers)

img = "./cat.jpg"
activations = get_activations(img,activations_output)


show_activations(activations, layer_names)


# This is trail 3. In this trial 2 cnn layers and one dense layer have been used. Data was processed by dividing with 255. Ther filters in cnn also has increased from last trial. Also there is a validation set data last 10% and batch size of 10

# In[4]:


#This is 2
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, utils
from tensorflow.keras.optimizers import Adam
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.keras.backend as K
from keras.preprocessing import image

(x_train, y_train), (x_test,y_test) = datasets.cifar10.load_data()
y_train = y_train.reshape(-1,)
y_test = y_test.reshape(-1,)
classes = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
x_train = x_train / 255.0
x_test = x_test / 255.0


cnn = models.Sequential([
    layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

cnn.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("\nThe is the model plot: \n")
tf.keras.utils.plot_model(cnn, to_file='./trial3.png', show_shapes=True)
print("\n")
print("\nThe is the history for 15 epochs: \n")
cnn.fit(x_train, y_train, validation_split=0.1,
       batch_size=10, verbose=2, epochs=15)
print("\n")
print("\nThis is the evaluation.\n")
cnn.evaluate(x_test,y_test)

layer_names = ['conv2d_1', 'max_pooling2d_1', 'conv2d_2', 'max_pooling2d_2']
def deprocess_image(x):
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1
    x += 0.5
    x = np.clip(x, 0, 1)
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def generate_pattern(layer_name, filter_index, size=150):

    layer_output = cnn.get_layer(layer_name).output
    loss = K.mean(layer_output[:, :, :, filter_index])
    grads = K.gradients(loss, model.input)[0]
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
    iterate = K.function([model.input], [loss, grads])
    input_img_data = np.random.random((1, size, size, 3)) * 20 + 128.
    step = 1.
    for i in range(40):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step
        
    img = input_img_data[0]
    return deprocess_image(img)

def get_activations(img, model_activations):
    img = image.load_img(img, target_size=(32, 32))
    img = image.img_to_array(img)                    
    img = np.expand_dims(img, axis=0)         
    img /= 255. 
    #plt.imshow(img[0])
    plt.show()
    return model_activations.predict(img)

def show_activations(activations, layer_names):
    
    images_per_row = 9
    for layer_name, layer_activation in zip(layer_names, activations):
        n_features = layer_activation.shape[-1]
        size = layer_activation.shape[1]
        #n_cols = n_features // images_per_row
        n_cols = 1
        display_grid = np.zeros((size * n_cols, images_per_row * size))
        for col in range(n_cols):
            for row in range(images_per_row):
                channel_image = layer_activation[0, :, :, col * images_per_row + row]
                channel_image -= channel_image.mean()
                channel_image /= (channel_image.std()+0.01)
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                display_grid[col * size : (col + 1) * size, row * size : (row + 1) * size] = channel_image

        scale = 1. / size
        plt.figure(figsize=(scale * display_grid.shape[1],
                            scale * display_grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')

    plt.show()
    

layers = [layer.output for layer in cnn.layers[:9]]
activations_output = tf.keras.models.Model(inputs=cnn.input, outputs=layers)

img = "./cat.jpg"
activations = get_activations(img,activations_output)


show_activations(activations, layer_names)


# This is trail 4. In this trial 2 cnn layers and one dense layer have been used. Data was processed by dividing with 255. There is a validation set data last 10% and batch size of 5. Learning rate has changed and dropout has been introduced.

# In[5]:


#This is 2.2 (best sofar don't touch)
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, utils
from tensorflow.keras.optimizers import Adam
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.keras.backend as K
from keras.preprocessing import image

(x_train, y_train), (x_test,y_test) = datasets.cifar10.load_data()
y_train = y_train.reshape(-1,)
y_test = y_test.reshape(-1,)
classes = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
x_train = x_train / 255.0
x_test = x_test / 255.0


cnn = models.Sequential([
    layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((3, 3)),
    
    layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(10, activation='softmax')
])

cnn.compile(optimizer=Adam(learning_rate=0.0001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("\nThe is the model plot: \n")
tf.keras.utils.plot_model(cnn, to_file='./trial4.png', show_shapes=True)
print("\n")
print("\nThe is the history for 15 epochs: \n")
cnn.fit(x_train, y_train, validation_split=0.1,
       batch_size=5, verbose=2, epochs=15)

print("\n")
print("\nThis is the evaluation.\n")
cnn.evaluate(x_test,y_test)


layer_names = ['conv2d_1', 'max_pooling2d_1', 'conv2d_2', 'max_pooling2d_2']
def deprocess_image(x):
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1
    x += 0.5
    x = np.clip(x, 0, 1)
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def generate_pattern(layer_name, filter_index, size=150):

    layer_output = cnn.get_layer(layer_name).output
    loss = K.mean(layer_output[:, :, :, filter_index])
    grads = K.gradients(loss, model.input)[0]
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
    iterate = K.function([model.input], [loss, grads])
    input_img_data = np.random.random((1, size, size, 3)) * 20 + 128.
    step = 1.
    for i in range(40):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step
        
    img = input_img_data[0]
    return deprocess_image(img)

def get_activations(img, model_activations):
    img = image.load_img(img, target_size=(32, 32))
    img = image.img_to_array(img)                    
    img = np.expand_dims(img, axis=0)         
    img /= 255. 
    #plt.imshow(img[0])
    plt.show()
    return model_activations.predict(img)

def show_activations(activations, layer_names):
    
    images_per_row = 9
    for layer_name, layer_activation in zip(layer_names, activations):
        n_features = layer_activation.shape[-1]
        size = layer_activation.shape[1]
        #n_cols = n_features // images_per_row
        n_cols = 1
        display_grid = np.zeros((size * n_cols, images_per_row * size))
        for col in range(n_cols):
            for row in range(images_per_row):
                channel_image = layer_activation[0, :, :, col * images_per_row + row]
                channel_image -= channel_image.mean()
                channel_image /= (channel_image.std()+0.01)
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                display_grid[col * size : (col + 1) * size, row * size : (row + 1) * size] = channel_image

        scale = 1. / size
        plt.figure(figsize=(scale * display_grid.shape[1],
                            scale * display_grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')

    plt.show()
    

layers = [layer.output for layer in cnn.layers[:9]]
activations_output = tf.keras.models.Model(inputs=cnn.input, outputs=layers)

img = "./cat.jpg"
activations = get_activations(img,activations_output)


show_activations(activations, layer_names)


# This is trail 5. In this trial 5 cnn layers, 3 maxpooling and two dense layer have been used. Data was processed by dividing with 255. There is a validation set data last 10% and batch size of 5. Learning rate same as last trial and two dropout has been introduced.

# In[3]:


#This is 6.2
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, utils
from tensorflow.keras.optimizers import Adam
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.keras.backend as K
from keras.preprocessing import image

(x_train, y_train), (x_test,y_test) = datasets.cifar10.load_data()
y_train = y_train.reshape(-1,)
y_test = y_test.reshape(-1,)
classes = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
x_train = x_train / 255.0
x_test = x_test / 255.0


cnn = models.Sequential([
    layers.Conv2D(filters=128, kernel_size=3, activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((3, 3)),
    
    layers.Conv2D(filters=256, kernel_size=3, padding='same', activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(filters=256, kernel_size=3, padding='same', activation='relu'),
    layers.Conv2D(filters=512, kernel_size=3, padding='same', activation='relu'),

    layers.Conv2D(filters=256, kernel_size=3, activation='relu'),
    layers.MaxPooling2D((3, 3)),


    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.1),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.1),
    layers.Dense(10, activation='softmax')
])

cnn.compile(optimizer=Adam(learning_rate=0.0001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("\nThe is the model plot: \n")
tf.keras.utils.plot_model(cnn, to_file='./trial5.png', show_shapes=True)
print("\n")
print("\nThe is the history for 15 epochs: \n")

cnn.fit(x_train, y_train, validation_split=0.1,
       batch_size=5, verbose=2, epochs=15)
print("\n")
print("\nThis is the evaluation.\n")
cnn.evaluate(x_test,y_test)


layer_names = ['conv2d_1', 'max_pooling2d_1', 'conv2d_2', 'max_pooling2d_2', 'conv2d_3', 'conv2d_4', 'conv2d_5', 'max_pooling2d_3']
def deprocess_image(x):
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1
    x += 0.5
    x = np.clip(x, 0, 1)
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def generate_pattern(layer_name, filter_index, size=150):

    layer_output = cnn.get_layer(layer_name).output
    loss = K.mean(layer_output[:, :, :, filter_index])
    grads = K.gradients(loss, model.input)[0]
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
    iterate = K.function([model.input], [loss, grads])
    input_img_data = np.random.random((1, size, size, 3)) * 20 + 128.
    step = 1.
    for i in range(40):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step
        
    img = input_img_data[0]
    return deprocess_image(img)

def get_activations(img, model_activations):
    img = image.load_img(img, target_size=(32, 32))
    img = image.img_to_array(img)                    
    img = np.expand_dims(img, axis=0)         
    img /= 255. 
    #plt.imshow(img[0])
    plt.show()
    return model_activations.predict(img)

def show_activations(activations, layer_names):
    
    images_per_row = 9
    for layer_name, layer_activation in zip(layer_names, activations):
        n_features = layer_activation.shape[-1]
        size = layer_activation.shape[1]
        #n_cols = n_features // images_per_row
        n_cols = 1
        display_grid = np.zeros((size * n_cols, images_per_row * size))
        for col in range(n_cols):
            for row in range(images_per_row):
                channel_image = layer_activation[0, :, :, col * images_per_row + row]
                channel_image -= channel_image.mean()
                channel_image /= (channel_image.std()+0.01)
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                display_grid[col * size : (col + 1) * size, row * size : (row + 1) * size] = channel_image

        scale = 1. / size
        plt.figure(figsize=(scale * display_grid.shape[1],
                            scale * display_grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')

    plt.show()
    

layers = [layer.output for layer in cnn.layers[:9]]
activations_output = tf.keras.models.Model(inputs=cnn.input, outputs=layers)

img = "./cat.jpg"
activations = get_activations(img,activations_output)


show_activations(activations, layer_names)


# In[ ]:




