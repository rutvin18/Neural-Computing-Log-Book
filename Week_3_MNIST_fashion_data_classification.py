#!/usr/bin/env python
# coding: utf-8

# # MNIST Fashion Dataset Application with Keras
# 
# This is a sample image classification tutorial with the MNIST fashion image dataset.

# In[1]:


pip install tensorflow


# In[3]:


from __future__ import absolute_import, division, print_function, unicode_literals

# Your code to import tensorflow
from tensorflow import keras
# Your code to import Keras
import numpy as np
# Your code to import numpy
import matplotlib.pyplot as plt

#print(tf.__version__)


# ## Import the Fashion MNIST dataset
# This guide uses the Fashion MNIST dataset which contains 70,000 grayscale images in 10 categories. The images show individual articles of clothing at low resolution (28 by 28 pixels), as seen below.
# 
# Here, 60,000 images are used to train the network and 10,000 images to evaluate how accurately the network learned to classify images. You can access the Fashion MNIST directly from TensorFlow. Import and load the Fashion MNIST data directly from TensorFlow:
# 
# 
# 

# In[4]:


from IPython.display import display, Image
myImage = Image('fashion_mnist_sprite');
myImage


# In[ ]:


fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()


# Loading the dataset returns four NumPy arrays:
# 
# * The `train_images` and `train_labels` arrays are the *training set*—the data the model uses to learn.
# * The model is tested against the *test set*, the `test_images`, and `test_labels` arrays.
# 
# The images are 28x28 NumPy arrays, with pixel values ranging from 0 to 255. The *labels* are an array of integers, ranging from 0 to 9. These correspond to the *class* of clothing the image represents:
# 
# <table>
#   <tr>
#     <th>Label</th>
#     <th>Class</th>
#   </tr>
#   <tr>
#     <td>0</td>
#     <td>T-shirt/top</td>
#   </tr>
#   <tr>
#     <td>1</td>
#     <td>Trouser</td>
#   </tr>
#     <tr>
#     <td>2</td>
#     <td>Pullover</td>
#   </tr>
#     <tr>
#     <td>3</td>
#     <td>Dress</td>
#   </tr>
#     <tr>
#     <td>4</td>
#     <td>Coat</td>
#   </tr>
#     <tr>
#     <td>5</td>
#     <td>Sandal</td>
#   </tr>
#     <tr>
#     <td>6</td>
#     <td>Shirt</td>
#   </tr>
#     <tr>
#     <td>7</td>
#     <td>Sneaker</td>
#   </tr>
#     <tr>
#     <td>8</td>
#     <td>Bag</td>
#   </tr>
#     <tr>
#     <td>9</td>
#     <td>Ankle boot</td>
#   </tr>
# </table>
# 
# Each image is mapped to a single label. Since the *class names* are not included with the dataset, store them here to use later when plotting the images:

# In[ ]:


class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


# ## Explore the data
# 
# Let's explore the format of the dataset before training the model. The following shows there are 60,000 images in the training set, with each image represented as 28 x 28 pixels:

# In[ ]:


# Your code to print the shape of the training images


# Likewise, there are 60,000 labels in the training set:

# In[ ]:


# Your code to print length of train labels


# Each label is an integer between 0 and 9:

# In[ ]:


# Your code to print the training labels


# There are 10,000 images in the test set. Again, each image is represented as 28 x 28 pixels:

# In[ ]:


# Your code to print the shape of the test set images


# And the test set contains 10,000 images labels:

# In[ ]:


# Your code to print the length of the test labels


# ## Preprocess the data
# 
# The data must be preprocessed before training the network. If you inspect the first image in the training set, you will see that the pixel values fall in the range of 0 to 255:

# In[ ]:


plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()


# Scale these values to a range of 0 to 1 before feeding them to the neural network model. To do so, divide the values by 255. It's important that the *training set* and the *testing set* be preprocessed in the same way:

# In[ ]:


train_images = train_images / 255.0

test_images = test_images / 255.0


# To verify that the data is in the correct format and that you're ready to build and train the network, let's display the first 25 images from the *training set* and display the class name below each image.

# In[ ]:


plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()


# ## Build the model
# 
# Building the neural network requires configuring the layers of the model, then compiling the model.

# Develop a a neural network model of your choice i.e. number of layers and activation function. Remember the input is a 2-D so you need to flatten the input.

# In[ ]:


#
# Your code to build the neural network model.
# Use model.add.Flatten() layer as the first layer.
# Chose softmax in the output layer. Chose the number of layers and actication functions in hidden layers of your choice.
# Do not forget to set correct number of neurons in the output layer.
#




# ### Compile the model
# 
# Before the model is ready for training, it needs a few more settings. These are added during the model's *compile* step:
# 
# * *Loss function* —This measures how accurate the model is during training. You want to minimize this function to "steer" the model in the right direction.
# * *Optimizer* —This is how the model is updated based on the data it sees and its loss function.
# * *Metrics* —Used to monitor the training and testing steps. The following example uses *accuracy*, the fraction of the images that are correctly classified.
# 
# 
# We have used 'adam' optimizer. Try other optimizer functions. Use Keras API as a reference  

# In[ ]:


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# ## Train the model
# 
# Training the neural network model requires the following steps:
# 
# 1. Feed the training data to the model. In this example, the training data is in the `train_images` and `train_labels` arrays.
# 2. The model learns to associate images and labels.
# 3. You ask the model to make predictions about a test set—in this example, the `test_images` array.
# 4. Verify that the predictions match the labels from the `test_labels` array.
# 
# 

# ### Feed the model
# 
# To start training,  call the `model.fit` method—so called because it "fits" the model to the training data:

# In[ ]:


model.fit(train_images, train_labels, epochs=100)


# As the model trains, the loss and accuracy metrics are displayed. This model reaches an accuracy of about 0.91 (or 91%) on the training data.

# ### Evaluate accuracy
# 
# Next, compare how the model performs on the test dataset:

# In[ ]:


test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print('\nTest accuracy:', test_acc)


# ### Make predictions
# 
# With the model trained, you can use it to make predictions about some images.

# In[ ]:


predictions = model.predict(test_images)


# Here, the model has predicted the label for each image in the testing set. Let's take a look at the first prediction:

# In[ ]:


predictions[0]


# A prediction is an array of 10 numbers. They represent the model's "confidence" that the image corresponds to each of the 10 different articles of clothing. You can see which label has the highest confidence value:

# In[ ]:


np.argmax(predictions[0])


# So, the model is most confident that this image is an ankle boot, or `class_names[9]`. Examining the test label shows that this classification is correct:

# In[ ]:


test_labels[0]


# Graph this to look at the full set of 10 class predictions.

# In[ ]:


def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array, true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array, true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')


# ### Verify predictions
# 
# With the model trained, you can use it to make predictions about some images.

# Let's look at the 0th image, predictions, and prediction array. Correct prediction labels are blue and incorrect prediction labels are red. The number gives the percentage (out of 100) for the predicted label.

# In[ ]:


i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  test_labels)
plt.show()


# Plot any other image

# In[ ]:


# Your code to plot any other image and confidense score of your choise


# Let's plot several images with their predictions.
# In the next cell, you need to plot first 15 images in the following format.
# 
# Use the sample code of the plots provided in this tutorial. Hint - You need to
# use subplots, for loop, etc..
# 
# ![image.png](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAA1QAAALICAYAAAB4srHRAAAgAElEQVR4nOzdeZwdVZn/8W+Rfd83AiQECGFNEAQBZRFEiAjiOAIzg6L+0J+4Ozo4LqDjhvgat3EAHcSFH4tDQMEF2VQgkBAIpEMgCYQsECD7vm/n98eprj7nyb11b1dub7c/79eLF+e5VV1V3el6upbznCMBAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAQCVJWx8AUMmQIUPc2LFj2/owOoWGBmnXrvLLu3aVJk6MP5s5c+YqScNa9MCAVtLS+abIOYYm5BvUk33JN3m5hDxSO9XmnK6tcCzAPhk7dqyefvrptj6MTiGp8Ihl1y7J/lMkSbKk5Y4IaF0tnW+KnGNoQr5BPdmXfJOXS8gjtVNtztmvpQ8EAAAAAOpVq72hottW+0dXCtST9phz9uzZE8WbN2+O4n79+hXe9pYtW6J4v/2anpf17Nmz8HZbyuLFi7V69Wq6naMutMd8s3Hjxihevnx5FPfu3TuKd+7cmbV79OgRLbO5a/fu3WX3u2PHjig+5JBDKh9sK+AaBy2p1W6o6LbV/tGVAvWkPeYce4EzY8aMKD7rrLMKb/uZZ56J4r59+2bt8ePHF95uSznhhBO0evXqtj4MoCbaKt8456I4CfqBPfzww9Gyn/zkJ1E8adKkKF62bFnWPvTQQ6NlmzZtiuK1a9dGcdeuTZeTixYtipb97ne/K3nsrY1rHLQkuvwBAAAAQEEMSgEA+2Dbtm1R/KMf/SiKb7/99qxtn+quXLkyinv16hXFdv08tltfGIdPjyXptNNOi+Irrrgiis8999yq9wug7eS9obrmmmuiZY8//ngU33vvvWW3279//yi2XYp3meHlwty1devWaNkf//jHKD7//PPL7hfoqHhDBQAAAAAFcUMFAAAAAAXR5Q8AmuGqq66K4p///OdRvGHDhigOR9KyXfoGDRoUxbarTJ8+fbK2HVXLjsJltx12Bdq+fXu07E9/+lMU264/J598ctZ+9NFHBaB9CkfztBoaGqLY5pthw+IB78JRR22+GTx4cBR369YtisN8s2DBgmjZvHnzopguf6hHvKECAAAAgIK4oQIAAACAgrihAgAAAICCqKECgArCOqnrrrsuWjZy5MgoDuuepHgYYzvE8c6dO6M4b+jzcDvS3rUTdhjjvO2Gk/5KUpcuXaI4HF753e9+d7TsD3/4Q9n9AGg/7GS8Q4cOjWJb77lnz56sbWs0w2Wltm3XD7366quVDxbo4HhDBQAAAAAFcUMFAAAAAAVxQwUAAAAABVFDBQAVfO1rX8va/fv3j5bZ2iY7f8uyZcvKbnfgwIFRbGudunZtStG2ZmHbtm1RPGTIkLLHEW5H2nteKlvbNWLEiKxt56FatWpVFNu6DABtZ/ny5WWX2Txgc1fI1mTaeads3WW4LZsjV6xYUXY/QL3gDRUAAAAAFMQNFQAAAAAUxA0VAAAAABREDRUAVLB+/fqsbedbsfVHtmbq4x//eNb+2Mc+Fi1705veFMV2DqulS5dm7X79+kXLxowZE8W2diI8znA7kjR69Oiy60rSxo0bs/bWrVujZQsXLoxiaqiA9mPOnDlll3Xv3j2K7bkd1kXZeis7D5XNe3lzWNm6S6Ae8YYKAAAAAArihgoAAAAACqLLHwBUEA4zboc2t11frO9+97tZe8CAAdEy241my5YtUXzGGWdk7b/97W+5+zniiCOieN68eVl7w4YN0bIf//jHURwOCy9Jw4YNy9p2GPipU6dG8Yknnph7XABaT0NDQ9a2Xfxs7rL5JpyKIezmLO09LYMdcj3Mg3ZaBtuVGahHvKECAAAAgIK4oQIAAACAgrihAgAAAICCqKFqB2yNwn77xfe5tq9yyPZVtsOVvvTSS1n7sMMOK3qIQKeyY8eOssvs+WjPQesDH/hA1r7nnnty1127dm0Uh3VTV199dbSsf//+UXzHHXdE8Zo1a7L2kiVLomUXX3xxFNsaqjAn2eGTZ82aVfLYAbS9p556KmvbawlbM2XP7bBuyk7pYM/7QYMGRXF47WH3c+CBB1Y6bKDD4w0VAAAAABTEDRUAAAAAFMQNFQAAAAAURA1VM4TzLNi5Z2xf5ddeey2Kp02blrXPO++8aNm+zNFga6asu+++O2tfddVVhfcDdCavv/562WX2XN+6dWvutpYuXVr1fu+8886yyy677LIo7tWrVxTbWsyJEydm7TfeeCNa1rdv36qPyQrrMgG0L3Pnzs3a3bp1i5bZ3LVp06YoHjVqVNaePn16tMzWjto59MJ4165d0bLBgwdXOmygw+MNFQAAAAAUxA0VAAAAABTEDRUAAAAAFEQNVUG2L7L12GOPRfGTTz6ZtW19xqc//enCx7FixYoovv/++6O4X79+hbcNdFYrV66sel1bL2DrFsLz3dYdWKeffnrZZe985zujeNGiRVFs6xTuu+++rH3GGWdEy8L6KmnvmqrwOLt06RItW7ZsWdljBNC2wrmk7LlbqYbqve99b9X7sXmvd+/eZdfNm9cPqBe8oQIAAACAgrihAgAAAICC6PLXDOGwxF27xj+6p556KorDoUslacSIEVnbDjt80UUXRfGgQYOieNu2bVl7zJgx0bLVq1dH8YYNG6J49OjRAtA8dtqDkJ0ywbJdX8IucrbLjd3W/Pnzozic6mDhwoW5+z3iiCOieN68eVn7lVdeiZZdf/31UWyHSA5zkJ2aIe9nA6BtLV++PGs3d0qWSy+9tOwymwfWrFkTxUOHDi37tVu2bGnWcQAdEW+oAAAAAKAgbqgAAAAAoCBuqAAAAACgIGqoctghjsO6qc2bN0fLpkyZEsW2v3FYB7Vx48Zoma2jyIuff/75aNkBBxwQxbb+Kqz7AlCdvGHT7VDEdvhgG4dDkn/5y1/OXfeBBx6I4oaGhqxtz31bLxnWTElx/dXFF18cLZs1a5byhLkvSZJo2c6dO3O/FkDb2bp1a9a206ZUuh4488wzyy47+eSTo3jatGlRbHNZaMiQIbn7BeoBb6gAAAAAoCBuqAAAAACgIG6oAAAAAKCgDl9DZeuNbH9/WwcVLrfr2v7FtlYidOONN0ZxOM+UJPXs2TOKlyxZkrXDeqpSX2v7IofHaeeVsLVa69evj+Lt27dnbVv31dw5KoDO4o033ii7rNJcUvb8HTBgQNb+7ne/m7vfcF0pzg0vvPBC7teOHDkyiletWpW1bT6qJG/Ovbx1pfy8CaDt2PpHe27b64nQ2LFjo3jq1KlRnDc/n81rQD3iDRUAAAAAFMQNFQAAAAAUxA0VAAAAABTUIWqo8uqkbB2UZesdQs3t+3/77bdn7WXLlkXLjjvuuCi2dRTr1q3L2oMHD46W2TkawtoHSdq0aVPZ7Vr2Z7Vly5as/dJLL0XLJk2alLstoLPKm4fK6t69exS//e1vj+LHHnssa9t542zOCWsepThHhfNZlWJzQ1h/ZbdrtzVw4MAoDuepsvnKWrx4cRQfcsghuesDaB32+mjHjh1R3Jxz1eYue/1U6VoMqHe8oQIAAACAgrihAgAAAICCOkSXv7xXyXZYdBvbLjXhtip18bv55puj+MUXX8zaBx54YLRs9erVUWy73m3dujVrjx49Olq2cePGsscoSb17987adsj1SsPGh+6///4opssfUFrYRdey56s9ny+//PIovu+++7J2eC6XUimf5bHnftgF0Hb5s8Mlv/e9743isMtfJbaLMl3+gPbBnud26pSjjjqq6m1Nnjw5iq+77roobk6uAuoRb6gAAAAAoCBuqAAAAACgIG6oAAAAAKCgdlFDVanvra0NCOuG7LDoecOkW6+//noU33333VEc1j1J0mGHHZa1w6HMpb1rFGxNVbdu3bK2/X7Coc1LCb+nHj16lF0mSX369InicF+PP/547n4AePb8Ddm8MHz48CgeNGhQ2a8N84C091DnNjc0J5/Zrw2HNbbLbL466aSTym7XHkPPnj2jmNoJoH2yQ5vbmutx48ZVva2JEydGsR2CPW9KF3tdAtQj3lABAAAAQEHcUAEAAABAQdxQAQAAAEBBrVpDFfbnDeeAak6dgJQ/19LKlSujePHixVE8f/78rP3GG29Ey7p37x7F/fv3j+JwbpoNGzZEy3bu3BnFtkYh/H7tMdm+xwMHDix7XJX6RPfq1SuKw/X79u0bLZszZ44A7M3OQxXWDdm54Gx9wNy5c8tu184LY/OGlZfrrLw56ex27PfXnLn+7H7sPFQA2s4BBxyQte28U/Zaa//99696uzZ3WdRQobPjDRUAAAAAFMQNFQAAAAAUxA0VAAAAABTUqjVUYR1RaPny5VG8ZMmSKLb9gMPYzgmzaNGiKLZzPIX9gPv16xcts7UC69evj+JwX7Y/sd2PrWUK54+y8zeMGjUqim19VrhtO8eNnQ9rzZo1URzWTS1btix3XQBec+ZWOvzww6P45ZdfLruurVWy+8mbc68S+7VhTYOdv85u186llXeM9mtt3SqAthOeywsXLoyW2TqnF198sert2hpzK6/GqtJcm0A94A0VAAAAABTEDRUAAAAAFNSqXf5CDz30UNZ+/fXXo2X21bHtUlJu+PVSX2u79YVd5GwXONuVxQ59Hna3s91gbNc7O7x5OGyoHb7cDpPenC40tgugHRY17KZouxpWGgYV6KzscOZ554rt8vfII4+UXTdvaGFp7xwU5plK00vYrw3jct2tG4VDLdu40rDoNvcBaDsnnnhi1rZTONiu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Z7jOH9CQeEIFAAAAADlxQwUAAAAAOXFDBQAAAAA5tegaqkKhkBlnufzyy6N40aJFUXzsscdG8ZNPPpm2hwwZEvXtu+++tT6urc9o0yb7f8If//jHafsvf/lL1PfUU0/V+rhASxbWINk1m956660o3rp1axQPHjw4bYf1VFL1uqfVq1dHcVhjtWXLlqhv48aNUTx06NCS+7Z1BuvXr4/iF154IYrHjBmTtidOnBj12fcLoGmydVHhNdDbb78d9V155ZVR/MUvfjGKp06dmravuOKKqO+ee+6J4r/+9a9RfOedd6btM844I+qztVoVFRVR/PnPfz5t2/xq3x9Ql3hCBQAAAAA5cUMFAAAAADk1uyF/u3fvjuK9GdZnH3kfccQRafvcc8+N+g477LAotkNswkfRl112WdT30EMP1fqcahrid/vtt0fx3XffnbbtUKW5c+fW+rhAS7Zhw4a0bacEtp8rO6Q37B82bFjUZ6dgf/nll6N45cqVaXv06NGZx92xY0cUh0MP7TAZe46///3vo/iaa65J23aooX3/AJqmrOshO4T4kUceydzXAw88kLZPPvnkqM8utbBt27YoDssenn322ajPLiVh1XRNBNQXnlABAAAAQE7cUAEAAABATtxQAQAAAEBOjXLwadZUnrbf9rVqlX2PuH379rS9fPnyqO/QQw+NYjtN6Ne//vW0PW7cuKhvwYIFUWzrDA488MC0PWnSpKivR48eUfyNb3wjij/+8Y+nbTvN8j//+c8ovummm6I43P7ggw+O+gYOHCgANQs/37buydZLfuYzn4niG264IW3bz6/NV2GtlhRPo/7ee+9FfdOnT49im5PatWuXtu1yC3bK9XBqdymuubK1WkxFDDR/dlmV+fPnR/F+++0XxbfddlvaDq93pOp14p06dYri8DrOTpN+zDHHZJ7Ho48+mrY/+9nPRn27du0SUF94QgUAAAAAOXFDBQAAAAA5cUMFAAAAADk1yhqqmtaKyuqfPHly5s9ed911advWENm1WOyaVkuWLEnbdr0YK1wDRorrDj7ykY9Efd26dYvim2++OYpvvfXWtN2lS5eob9WqVVFsxzVPmDAhbb/00ktRn63XAFBcOK6/d+/eUd+6deui2H72R4wYkbZtLZNdCy6s8ZTi3GDrNJcuXRrFH/zgB0v+7MKFC6M+m0fsmnthjZVdB8bW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cLihAgAAAJBpxYq962/OuKECAAAAgJyooQKAemLrIfqb8RE7d+6M4rlz56btf//3f4/67JpWPXr0iOI+ffrkPk8A9ceuNZVVN/XpT386iu+9994o7ty5c9quqKiI+mzd06ZNm6J42bJlJY+7devWKO7YsWMUhzVW27Zti/quueaaKP7xj38cxb/85S/T9tlnnx312Zxo17ACGgueUAEAAABATtxQAQAAAEBO3FABAAAAQE4MRgWABmLrEN59990o7tKlS9ru3r171Ne+ffsotutQderUqRynCKABPf3001H80EMPRfH+++8fxeEaVrb+yNq+fXsUL1iwIG2PHj066rN1UevWrYvisKbT1nfaXGTX47v44ovT9iGHHBL1DR8+PIrD9a6k6nVhQEPhCRUAAAAA5MQNFQAAAADkxJA/AGggdljN/Pnzozhr+mTbZ4f8DRw4sOTPMmwGaDxatSr93fZvfvObKG7dunUU22F94fTl9nNe0/TsYbx06dKozw4xzsohts+eoz1u+P6vuuqqqO/RRx8teRygMeEJFQAAAADkxA0VAAAAAOTEDRXQEPr3lwqF4v/179/QZwcAAIBaooYKaAgrVuTrQ5MT1hPY8f92OuE2beKUnPWz/fr1i+LVq1eX/FkATUf42f3nP/8Z9VVUVESxnYI8q5bJbmvrosL6LFtvtXnz5ii2Sz6Ex6op99iaqq5du6bt5557LuqbOXNmFI8dOzZz30BD4QkVAAAAAOTEDRUAAAAA5MQNFQAAAADkRA0VANShrHVT3nrrrSjOWo9m27ZtUbxx48Yo7tWrVxQvXLgw1zkBaFj33HNP2l6zZk3UF9YbSdVrncLPdrdu3aK+LVu2RLGtqQrXsLL1nfY4Nh916NCh6DlINddUZdVf/fSnP43i2267LXNfQEPhCRUAAAAA5MQNFQAAAADkxA0VAAAAAOREDVXgpptuiuJZs2Zl9mex44CpWQBgPf3001G83377RXHbtm3Ttq1hsGyOmTt37l6eHYCG8Pzzz6ftcG0oqXrdk9WuXbu0vXXr1syfDfOLFK8P1b1798zj2GucsP7K1oLWdD0UHte+38mTJ2eeB9BY8IQKAAAAAHLihgoAAAAAcmqwIX/ho+iOHTvWelspfqRdE/v4OPTYY49F8dKlS6O4b9++UXz++een7R/84AdR37777hvFWUP8wkfjxWSdM4Cma968eVHcp0+fKG7fvn3Jn7VTINscY+Nly5blOUUADWzq1Klpu6bhc/Z6KMwD77//ftQXTm0uxUPt7M/a/GHzS9Z12Pbt2zO3tccN35PNgRUVFSWPAzQmPKECAAAAgJy4oQIAAACAnLihAgAAAICcGqyGKqxHuvTSS6O+448/PoprqrHKy06DfuSRR0axHfc7aNCgtH3PPfdEfbbe6qyzzoriLl26pG1bI2VrquwY6T3B9OxA4xXWRkjVaw3s5zec5thOcWzrI2ytxZIlS3KfJ4CGM3/+/LRtrxfs9YFdTiHMA23axJd4WbVLdnubT+yU63Zfpc6hpm2l+BrInvOmTZsyfxZoLHhCBQAAAAA5cUMFAAAAADlxQwUAAAAAOdVbDdXu3bu1efPmNA7H9z/yyCPRtlu2bIniMWPGRHHPnj3Ttl2jwI4nXrRoURT/4Q9/SNv9+/eP+nr37h3Fjz76aBR/7GMfS9vr1q2L+h5//PEonjt3bhT///buPFyOqszj+LdyE7KHrIQQJSFAgsMWBERGQQOyKsPMuCCILI4ojgvLOPNkHoUBRIijg87go46AGyK4IIKEQUFFMgghBEIWIBAgMEAgJJCEQBIhvvPHOd331Hu7qzud7ntv3/w+z5OHOl3VVaea1Js6dc57asqUKeXlI444Irdu0qRJNMrnXxWNp9b7rUR61ty5c3Nln2tQlE9Z6300Pv9qwoQJ5eVly5bl1u2222511lhEutsLL7xQXvb3JVuTy1Tr3XXpvvy9hN/W7zvd3ud7+jpvSa738uXLc+V169blyiNGjKh7XyKtpB4qERERERGRBqlBJSIiIiIi0iA1qERERERERBrUbTlUGzZsYPHixRXXpblVANdcc02uvM8+++TK6fuh/LuifK7AokWLcuX0vS+HHHJIbp1/R8xRRx2VK6f5Wv64Rx99dK68cuXKXPnRRx8tL9999925dW95y1ty5T333DNXPuCAA8rL48aNy63zeVHKkxLpvZYsWZIr+zwEH1fSd7AU5TtUWp/mLaxevTq3TjlUIr1Xmh/p/02v9e66NA+zVs6Ul+ZB+dwtn9vuy2k9ff6VVyv3u8jSpUtz5QMPPLDu74q0knqoREREREREGqQGlYiIiIiISIO6bcjf5s2bc1ONv/TSS52V6J+vxtq1a3PlG264IVceNWpUedlPxzl8+PBc+eCDD86Vp06dWl72w2389OyrVq3KldMu7nTqdsifD3Sdzn3nnXeuuAxdpwGdM2dOrjxv3ryq+x05cmSu7Kdg32GHHcrLe+yxR27dwIEDEZHu46cA9kP8/DC+tOzjpB/646Xffeyxx3LrDjrooNqVFZFu8eyzz1Zd54fp+dclNFO6bz8Mz8cmf+/l76eK+O+mcbDW+T355JO5sob8SW+hHioREREREZEGqUElIiIiIiLSIDWoREREREREGtRtOVT9+vVj6NCh5XI6jfjpp5+e23by5Mm5ss9P2rhxY3nZ5xANGjSo6rYACxcurFrHYcOG5co+XynNWXj++edz63wuxIgRI6p+1+dM+elJfX5Wyp+Pn579ueeey5XTc7j44otz604++eSqxxGR5nv66adz5WnTpuXKPrcg5XMpfE6Vz3lIcxr86yNEpPfwU4EXKbrOt1Y69bl/1YKfvt3fa6X1qlVHn4+V3j/VmkJ9xYoVhetFeop6qERERERERBqkBpWIiIiIiEiD1KASERERERFpULflUK1Zs4abbrqpXJ4wYUJ52ef9+ByjKVOm5Mrpe5z8WFy/r02bNuXKmzdvLqxjyr8Pa8CAAeXl9P1OUDuHKuVzs8aPH19YxzT/yo9b9mX/26W/h8/BuOyyy6rWUUSaI72efQ6kzy0oereUzy3w17OPdWnOg8/5FJHe44knnqh7W5876d/blMYFH1+KtvX8eyr9RYqaWAAAFtxJREFUfYmPR+m+/X59PXw53b5WDtWLL75YuF6kp6iHSkREREREpEFqUImIiIiIiDRIDSoREREREZEGdVsO1aZNm1i2bFm5vOuuu5aX99prr9y2ixcvzpWfeeaZXDnNC/L5SLXG36brfb6CL/txwOmYYj+O1483Hjx4cK6c5l95q1atqlpHgFdeeaW87PO80nXQ9V1aac7GY489llvn9yUizffUU09VXefj16uvvporp3GjKGehUjnNr/TvvxKR3sO/T7KIvy/xuU3+fVFbIo0hteKNr0da9nXy90c+hyp9/17RvRJ0fS+pSG+hHioREREREZEGqUElIiIiIiLSoG4b8tevX7/c8JZ77rmnvOyH2vmpwP361157rbzspycfO3Zsrrx+/fpcuWjadN9V7qcnTcu+y9pPm+6l3dh+WJ7v7k/PD/JToftpl9Ou8kp1TqeV99+98MIL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)
# 
# 

# In[ ]:


#
# Your code to plot first 15 predictions in the above format.
#


# ## Use the trained model
# 
# Finally, use the trained model to make a prediction about a single image.

# In[ ]:


img = test_images[1]

# Your code to print the shape of the image


# `tf.keras` models are optimized to make predictions on a *batch*, or collection, of examples at once. Accordingly, even though you're using a single image, you need to add it to a list:

# In[ ]:


img = (np.expand_dims(img,0))

# Your code to print the shape of the image.


# Now predict the correct label for this image:

# In[ ]:


predictions_single = model.predict(img)

print(predictions_single)


# In[ ]:


plot_value_array(1, predictions_single[0], test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)


# `model.predict` returns a list of lists—one list for each image in the batch of data. Grab the predictions for our (only) image in the batch:

# In[ ]:


np.argmax(predictions_single[0])


# # **Lab Logbook Requirement:**
# 
# # Please record the model's accuracy and a summary of its development. You can obtain the model summary using the model.summary() method. The API for obtaining the model summary is defined in the following link:
# 
# # https://keras.io/api/models/model/#summary-method
# 
# # Ensure that no code or other information is added to the logbook and that only required information is present.  Marks will not be awarded if anything else is found in the logbook or instructions are not clearly followed.
# 
