from abc import abstractmethod
from math import log2
from typing import Tuple

import keras.backend as K
import tensorflow as tf
from keras import Input, Model
from keras.layers import Activation, Dense, Layer
from qkeras import (
    QActivation,
    QDense,
    quantized_bits,
    quantized_relu,
    quantized_sigmoid,
)


class GarNetFactoryBase:
    @abstractmethod
    def init_model(
        self,
        encoder_units: int,
        aggregator_units: int,
        decoder_units: int,
    ):
        raise NotImplementedError

    def create_keras_model(self):
        hits = Input(shape=(128, 4))

        encoded_features = self.dense_encoder(hits)
        aggregated_distances = self.dense_aggregator(hits)
        x = self.garnet([encoded_features, aggregated_distances])
        x = self.dense_decoder(x)
        x = self.activation_decoder(x)
        x = self.dense(x)
        x = self.activation(x)
        energies = self.dense_regression(x)
        classes = self.dense_classification(x)
        classes = self.activation_classification(classes)

        return Model(inputs=hits, outputs=[energies, classes])


class GarNetFactory(GarNetFactoryBase):
    def __init__(
        self,
        encoder_units: int = 16,
        aggregator_units: int = 8,
        decoder_units: int = 16,
    ):
        self.init_model(
            encoder_units,
            aggregator_units,
            decoder_units,
        )

    def init_model(
        self,
        encoder_units: int,
        aggregator_units: int,
        decoder_units: int,
    ):
        self.dense_encoder = Dense(encoder_units)
        self.dense_aggregator = Dense(aggregator_units)
        self.garnet = GarNetLayer(name="garnet")
        self.dense_decoder = Dense(decoder_units)
        self.activation_decoder = Activation("relu")
        self.dense = Dense(8)
        self.activation = Activation("relu")
        self.dense_regression = Dense(1, name="regression")
        self.dense_classification = Dense(1)
        self.activation_classification = Activation("sigmoid", name="classification")


class QGarNetFactory(GarNetFactoryBase):
    def __init__(
        self,
        encoder_units: int = 16,
        aggregator_units: int = 8,
        decoder_units: int = 16,
        precision: Tuple[int, int] = (32, 16),
    ):
        self.init_model(
            encoder_units=encoder_units,
            aggregator_units=aggregator_units,
            decoder_units=decoder_units,
            precision=precision,
        )

    def init_model(
        self,
        encoder_units: int,
        aggregator_units: int,
        decoder_units: int,
        precision: Tuple[int, int] = (32, 16),
    ):
        # Currently QGarNet only supports alpha=1 due to scaling issues in HLS
        quantizer = quantized_bits(*precision, alpha=1)

        self.input_activation = QActivation(quantizer)
        self.dense_encoder = QDense(
            encoder_units, kernel_quantizer=quantizer, bias_quantizer=quantizer
        )
        self.dense_aggregator = QDense(
            aggregator_units, kernel_quantizer=quantizer, bias_quantizer=quantizer
        )
        self.garnet = GarNetLayer(name="garnet")
        self.dense_decoder = QDense(
            decoder_units, kernel_quantizer=quantizer, bias_quantizer=quantizer
        )
        self.activation_decoder = QActivation(quantized_relu(*precision))
        self.dense = QDense(8, kernel_quantizer=quantizer, bias_quantizer=quantizer)
        self.activation = QActivation(quantized_relu(*precision))
        self.dense_regression = QDense(
            1, kernel_quantizer=quantizer, bias_quantizer=quantizer, name="regression"
        )
        self.dense_classification = QDense(
            1, kernel_quantizer=quantizer, bias_quantizer=quantizer
        )
        self.activation_classification = QActivation(
            quantized_sigmoid(*precision), name="classification"
        )


class GarNetLayer(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.V: int = 0  # Number of vertices (hits)
        self.S: int = 0  # Number of aggregators per vertex (hit)
        self.N: int = (
            0  # Number of encoded features per vertex (hit) (coming from the encoder layer)
        )
        self.max_dist_input: tf.Variable = tf.Variable(0, trainable=False)

    def build(self, input_shape):
        super().build(input_shape)

        shape_encoded_features, shape_aggregated_distances = input_shape

        # Ensure that both inputs contain the same number of vertices V
        assert shape_encoded_features[1] == shape_aggregated_distances[1]

        self.V = shape_aggregated_distances[1]
        self.S = shape_aggregated_distances[2]
        self.N = shape_encoded_features[2]

        if self.V > 128:
            raise ValueError("GarNetLayer currently only supports <= 128 vertices")
        if not log2(self.V).is_integer():
            raise ValueError("Number of vertices must be a power of 2")

        self.max_dist_input = self.add_weight(
            "max_dist_input", initializer="zeros", trainable=False
        )

    def call(self, inputs):
        # Unpack inputs: encoded features and aggregated distances
        fi_v, d_av = inputs
        assert self.V == fi_v.shape[1] == d_av.shape[1]
        assert self.S == d_av.shape[2]
        assert self.N == fi_v.shape[2]

        # Update max input on each iteration
        self.max_dist_input.assign(
            tf.maximum(tf.reduce_max(tf.round(tf.abs(d_av) + 1)), self.max_dist_input)
        )

        # Weighted distances
        w_av = K.exp(-K.square(d_av))  # (B, V, S)

        # Aggregation across vertices
        fi_v = K.expand_dims(fi_v, axis=1)
        w_av_T = K.expand_dims(K.permute_dimensions(w_av, (0, 2, 1)), axis=3)
        hi_av = w_av_T * fi_v
        hi_av = K.mean(hi_av, axis=2)

        # Send aggregated features back to vertices using the same weights
        hi_av = K.expand_dims(hi_av, axis=1)
        w_av = K.expand_dims(w_av, axis=3)
        f_av_tilde = w_av * hi_av
        f_av_tilde = K.reshape(f_av_tilde, (-1, self.V, self.S * self.N))

        return tf.reduce_mean(f_av_tilde, axis=1)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "V": self.V,
                "S": self.S,
                "N": self.N,
                "max_dist_input": self.max_dist_input.numpy(),
            }
        )
        return config
