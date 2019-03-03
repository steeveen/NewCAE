from keras_contrib.applications.densenet import DenseNetFCN
from keras.utils import plot_model
model=DenseNetFCN(input_shape=(256,256,3),classes=2,)
plot_model(model,'tiramisu_nlpb=6.png',show_shapes=True)