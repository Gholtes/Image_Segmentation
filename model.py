from keras.layers import Input, MaxPooling2D, UpSampling2D
from keras.layers.convolutional import Convolution2D
from keras.layers.core import Activation, Reshape
from keras.layers.normalization import BatchNormalization
from keras.models import Model
# from layers import MaxPoolingWithArgmax2D, MaxUnpooling2D

def segnet(input_shape, binary = True, n_labels = 1, kernel = 3, pool_size = (2,2), output_mode = "softmax"):

	# encoder
    inputs = Input(shape=input_shape)

    conv_1 = Convolution2D(64, (kernel, kernel), padding="same")(inputs)
    conv_1 = BatchNormalization()(conv_1)
    conv_1 = Activation("relu")(conv_1)
    conv_2 = Convolution2D(64, (kernel, kernel), padding="same")(conv_1)
    conv_2 = BatchNormalization()(conv_2)
    conv_2 = Activation("relu")(conv_2)

	pool_1 =  pool_size=pool_size, strides=pool_size, padding="same")(conv_2)

	conv_3 = Convolution2D(128, (kernel, kernel), padding="same")(pool_1)
    conv_3 = BatchNormalization()(conv_3)
    conv_3 = Activation("relu")(conv_3)
    conv_4 = Convolution2D(128, (kernel, kernel), padding="same")(conv_3)
    conv_4 = BatchNormalization()(conv_4)
    conv_4 = Activation("relu")(conv_4)

	pool_2 =  pool_size=pool_size, strides=pool_size, padding="same")(conv_4)

	unpool_1 = MaxUnpooling2D(size = pool_size)(pool_2)

    conv_5 = Convolution2D(128, (kernel, kernel), padding="same")(unpool_1)
    conv_5 = BatchNormalization()(conv_5)
    conv_5 = Activation("relu")(conv_5)
    conv_6 = Convolution2D(128, (kernel, kernel), padding="same")(conv_6)
    conv_6 = BatchNormalization()(conv_6)
    conv_6 = Activation("relu")(conv_6)

	unpool_2 = MaxUnpooling2D(size = pool_size)(conv_6)

    conv_7 = Convolution2D(64, (kernel, kernel), padding="same")(unpool_2)
    conv_7 = BatchNormalization()(conv_7)
    conv_7 = Activation("relu")(conv_7)
    conv_8 = Convolution2D(64, (kernel, kernel), padding="same")(conv_7)
    conv_8 = BatchNormalization()(conv_8)
    conv_8 = Activation("relu")(conv_8)

	conv_9 = Convolution2D(n_labels, (1, 1), padding="valid")(conv_8)
    conv_9 = BatchNormalization()(conv_9)
    conv_9 = Reshape(
        (input_shape[0] * input_shape[1], n_labels),
        input_shape=(input_shape[0], input_shape[1], n_labels),
    )(conv_9)

	outputs = Activation(output_mode)(conv_9)
    
	print("Build model done..")

	model = Model(inputs=inputs, outputs=outputs, name="SegNet")

    return model