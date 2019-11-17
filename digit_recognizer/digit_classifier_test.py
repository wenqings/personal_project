from keras.models import load_model
import cv2
from keras.datasets import mnist

i = 666
model = load_model('digit_classifier.h5')


(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_test[i])
cv2.imshow('',x_test[i])
cv2.waitKey(0)

y = model.predict_classes(x_test[i].reshape(1,784))

print("Predict:",y)