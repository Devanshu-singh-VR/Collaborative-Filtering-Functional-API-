import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

book = pd.read_csv('D:\colab_filter\_books.csv')
rating = pd.read_csv('D:\colab_filter\_ratings.csv')
file = np.array(book)
book.info()
rating.info()

x_train, x_test = train_test_split(np.array(rating), test_size = 0.2, random_state=1)

data = np.array(rating)
num_books = max(data[:,0])
num_users = max(data[:,1])

#Book input network
input_books = tf.keras.layers.Input((1))
embed_books = tf.keras.layers.Embedding(num_books + 1,15)(input_books)
flatt_books = tf.keras.layers.Flatten()(embed_books)

#user input network
input_users = tf.keras.layers.Input((1))
embed_users = tf.keras.layers.Embedding(num_users + 1,15)(input_users)
flatt_users = tf.keras.layers.Flatten()(embed_users)

concat = tf.keras.layers.Concatenate()([flatt_books, flatt_users])
dense1 = tf.keras.layers.Dense(128, activation='relu')(concat)
dense2 = tf.keras.layers.Dense(1, activation='relu')(dense1)

model = tf.keras.Model([input_books, input_users], dense2)

opt = tf.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=opt, loss='mean_squared_error', metrics=['accuracy'])

hist = model.fit([x_train[:,0], x_train[:,1]], x_train[:,2],
                 batch_size=500,epochs=10, validation_data=([x_test[:,0], x_test[:,1]], x_test[:,2]))

train_loss = hist.history['loss']
plt.plot(train_loss, color='r', label='Train Loss')
plt.title("Train and Validation Loss Curve")
plt.legend()
plt.show()

#recomment some good books to user 100 :)
item = np.arange(1,10001)
user_100 = np.array([100 for i in range(len(item))])

pred = model.predict([item, user_100])

#movies data
movies = file[:,10]

print('Recommended movies')
print('')
for i in range(len(item)):
    if pred[i]>4.7:
        print(movies[i],"  :",pred[i])

print('')
print('initial rated movies by 100th user ->')
print('')
for i in range(len(x_train[:,0])):
    if x_train[i,1] == 100:
        print(movies[x_train[i,0]], '   :',x_train[i,2])

print('....................')
print('....................')
for i in range(len(x_train[:, 0])):
    if x_train[i, 1] == 100:
        print('initial item newly rated :', movies[x_train[i, 0]], '   :', pred[x_train[i, 0]])


print(embed_books)
