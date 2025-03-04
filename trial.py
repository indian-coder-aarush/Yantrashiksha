from Tanitra import Tanitra
from Parata import PraveshParata, GuptaParata, NirgamParata, Samasuchaka,ConvLayer2D,MaxPoolingLayer2D
from Pratirup import AnukramikPratirup
import cupy as cp
from tensorflow.keras.datasets import mnist

(x_train_ref, y_train), (x_test_ref, y_test) = mnist.load_data()
x_train_ref, x_test_ref = x_train_ref / 255.0, x_test_ref / 255.0

x_train = cp.zeros((3000, 28,28))
x_test = cp.zeros((200,784))
y_train_revised = cp.zeros((3000,10))
y_test_revised = cp.zeros((200,10))

for i in range(len(x_train[:3000])):
    x_train[i] = cp.array(x_train_ref[i])

for i in range(len(y_train[:3000])):
    y_train_revised[i][cp.array(y_train)[i]] = 1

for i in range(len(y_test[300:500])):
    y_test_revised[i][cp.array(y_test)[i]] = 1

x_train = Tanitra(x_train)
x_test = Tanitra(x_test)
y_train_revised = Tanitra(y_train_revised)
y_test_revised = Tanitra(y_test_revised)

model = AnukramikPratirup()

normalizer = Samasuchaka('min-max')

model.add(PraveshParata((28,28)))
model.add(ConvLayer2D(2,5,4,'relu'))
model.add(MaxPoolingLayer2D(2,3))
model.add(GuptaParata(80, 'relu',))
model.add(GuptaParata(60, 'relu',))
model.add(NirgamParata(10, 'relu',))

model.learn ( x_train, y_train_revised,epochs=100)