import tensorflow as tf

class DenseLayer(tf.keras.layers.Layer):
    def __init__(self, units):
        super(DenseLayer, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                 initializer='random_normal',
                                 trainable=True)
        self.b = self.add_weight(shape=(self.units,),
                                 initializer='zeros',
                                 trainable=True)

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b

class BatchNormalizationLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(BatchNormalizationLayer, self).__init__()

    def build(self, input_shape):
        self.gamma = self.add_weight(shape=(input_shape[-1],),
                                     initializer='ones',
                                     trainable=True)
        self.beta = self.add_weight(shape=(input_shape[-1],),
                                    initializer='zeros',
                                    trainable=True)
        self.moving_mean = self.add_weight(shape=(input_shape[-1],),
                                           initializer='zeros',
                                           trainable=False)
        self.moving_variance = self.add_weight(shape=(input_shape[-1],),
                                               initializer='ones',
                                               trainable=False)

    def call(self, inputs, training=False):
        mean, variance = tf.nn.moments(inputs, axes=[0])
        if training:
            self.moving_mean.assign(mean)
            self.moving_variance.assign(variance)
        else:
            mean = self.moving_mean
            variance = self.moving_variance
        return self.gamma * (inputs - mean) / tf.sqrt(variance + 1e-5) + self.beta

class ReLULayer(tf.keras.layers.Layer):
    def __init__(self):
        super(ReLULayer, self).__init__()

    def call(self, inputs):
        return tf.nn.relu(inputs)

class TransposeConvLayer(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides, padding='same', use_bias=True):
        super(TransposeConvLayer, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.use_bias = use_bias

    def build(self, input_shape):
        self.kernel = self.add_weight(shape=(self.kernel_size, self.kernel_size, self.filters, input_shape[-1]),
                                      initializer='random_normal',
                                      trainable=True)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.filters,),
                                        initializer='zeros',
                                        trainable=True)

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        output_shape = [batch_size,
                        inputs.shape[1] * self.strides,
                        inputs.shape[2] * self.strides,
                        self.filters]
        outputs = tf.nn.conv2d_transpose(inputs, self.kernel, output_shape, strides=[1, self.strides, self.strides, 1], padding=self.padding.upper())
        if self.use_bias:
            outputs = tf.nn.bias_add(outputs, self.bias)
        return outputs


class Generator(tf.keras.Model):
    def __init__(self, noise_dim):
        super(Generator, self).__init__()
        self.noise_dim = noise_dim
        self.dense1 = DenseLayer(7 * 7 * 256)
        self.batch_norm1 = BatchNormalizationLayer()
        self.relu1 = ReLULayer()
        self.reshape = tf.keras.layers.Reshape((7, 7, 256))
        self.trans_conv1 = TransposeConvLayer(128, 5, 1)
        self.batch_norm2 = BatchNormalizationLayer()
        self.relu2 = ReLULayer()
        self.trans_conv2 = TransposeConvLayer(64, 5, 2)
        self.batch_norm3 = BatchNormalizationLayer()
        self.relu3 = ReLULayer()
        self.trans_conv3 = TransposeConvLayer(1, 5, 2)
        
    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.batch_norm1(x, training=True)
        x = self.relu1(x)
        x = self.reshape(x)
        x = self.trans_conv1(x)
        x = self.batch_norm2(x, training=True)
        x = self.relu2(x)
        x = self.trans_conv2(x)
        x = self.batch_norm3(x, training=True)
        x = self.relu3(x)
        x = self.trans_conv3(x)
        return tf.nn.tanh(x)

noise_dim = 100
generator = Generator(noise_dim)
