# inbuilt lib imports:
from typing import Dict
import math

# external libs
import tensorflow as tf
from tensorflow.keras import models, layers


# project imports


class CubicActivation(layers.Layer):
    """
    Cubic activation as described in the paper.
    """

    def call(self, vector: tf.Tensor) -> tf.Tensor:
        """
        Parameters
        ----------
        vector : ``tf.Tensor``
            hidden vector of dimension (batch_size, hidden_dim)

        Returns tensor after applying the activation
        """
        # TODO(Students) Start
        # Comment the next line after implementing call.
        return tf.pow(vector, 3)
        # raise NotImplementedError
        # TODO(Students) End


class DependencyParser(models.Model):
    def __init__(self,
                 embedding_dim: int,
                 vocab_size: int,
                 num_tokens: int,
                 hidden_dim: int,
                 num_transitions: int,
                 regularization_lambda: float,
                 trainable_embeddings: bool,
                 activation_name: str = "cubic") -> None:
        """
        This model defines a transition-based dependency parser which makes
        use of a classifier powered by a neural network. The neural network
        accepts distributed representation inputs: dense, continuous
        representations of words, their part of speech tags, and the labels
        which connect words in a partial dependency parse.

        This is an implementation of the method described in

        Danqi Chen and Christopher Manning.
        A Fast and Accurate Dependency Parser Using Neural Networks. In EMNLP 2014.

        Parameters
        ----------
        embedding_dim : ``str``
            Dimension of word embeddings
        vocab_size : ``int``
            Number of words in the vocabulary.
        num_tokens : ``int``
            Number of tokens (words/pos) to be used for features
            for this configuration.
        hidden_dim : ``int``
            Hidden dimension of feedforward network
        num_transitions : ``int``
            Number of transitions to choose from.
        regularization_lambda : ``float``
            Regularization loss fraction lambda as given in paper.
        trainable_embeddings : `bool`
            Is the embedding matrix trainable or not.
        """
        super(DependencyParser, self).__init__()
        self._regularization_lambda = regularization_lambda

        if activation_name == "cubic":
            self._activation = CubicActivation()
        elif activation_name == "sigmoid":
            self._activation = tf.keras.activations.sigmoid
        elif activation_name == "tanh":
            self._activation = tf.keras.activations.tanh
        else:
            raise Exception(f"activation_name: {activation_name} is from the known list.")

        # Trainable Variables
        # TODO(Students) Start

        self._trainable_embeddings = trainable_embeddings
        self._embedding_dim = embedding_dim
        self._hidden_dim = hidden_dim
        self._num_tokens = num_tokens
        self._vocab_size = vocab_size
        self._regularization_lambda = regularization_lambda
        self._num_transitions = num_transitions

        w1_std_dev = tf.Variable(tf.sqrt(2 / (self._hidden_dim + self._embedding_dim * self._num_tokens)),
                                 dtype=tf.float32)
        # weight_1 = [hidden_dim, embedding_dim * num_tokens]
        self._weight1 = tf.Variable(
            tf.random.truncated_normal([self._hidden_dim, self._embedding_dim * self._num_tokens], mean=0.0,
                                       stddev=w1_std_dev,
                                       dtype=tf.dtypes.float32), trainable=True)

        w2_std_dev = tf.Variable(tf.sqrt(2 / (self._num_transitions + self._hidden_dim)), dtype=tf.float32)
        # weight_2 = [num_transitions, hidden_dim]
        self._weight2 = tf.Variable(tf.random.truncated_normal([self._num_transitions, self._hidden_dim], mean=0.0,
                                                               stddev=w2_std_dev,
                                                               dtype=tf.dtypes.float32), trainable=True)

        # bias = [1, hidden_dim]
        self._bias = tf.Variable(tf.zeros([1, self._hidden_dim], dtype=tf.float32), trainable=True)

        # embeddings = [vocab_size, embedding_dim]
        self.embeddings = tf.Variable(tf.random.uniform([self._vocab_size, self._embedding_dim],
                                                        minval=-0.01, maxval=0.01, dtype=tf.dtypes.float32),
                                      trainable=self._trainable_embeddings)
        # TODO(Students) End

    def call(self,
             inputs: tf.Tensor,
             labels: tf.Tensor = None) -> Dict[str, tf.Tensor]:
        """
        Forward pass of Dependency Parser.

        Parameters
        ----------
        inputs : ``tf.Tensor``
            Tensorized version of the batched input text. It is of shape:
            (batch_size, num_tokens) and entries are indices of tokens
            in to the vocabulary. These tokens can be word or pos tag.
            Each row corresponds to input features a configuration.
        labels : ``tf.Tensor``
            Tensor of shape (batch_size, num_transitions)
            Each row corresponds to the correct transition that
            should be made in the given configuration.

        Returns
        -------
        An output dictionary consisting of:
        logits : ``tf.Tensor``
            A tensor of shape ``(batch_size, num_transitions)`` representing
            logits (unnormalized scores) for the labels for every instance in batch.
        loss : ``tf.float32``
            If input has ``labels``, then mean loss for the batch should
            be computed and set to ``loss`` key in the output dictionary.

        """
        # TODO(Students) Start
        # embed_mat = [batch_size, embedding_dim * num_tokens]
        embed_mat = tf.nn.embedding_lookup(self.embeddings, inputs)
        embed_mat = tf.reshape(embed_mat, [inputs.shape[0], -1])

        # h = [batch_size, hidden_dim]
        h = self._activation(tf.math.add(tf.matmul(embed_mat, self._weight1, transpose_b=True), self._bias))

        # logits = [batch_size * num_tokens]
        logits = tf.matmul(h, self._weight2, transpose_b=True)

        # TODO(Students) End
        output_dict = {"logits": logits}

        if labels is not None:
            output_dict["loss"] = self.compute_loss(logits, labels)
        return output_dict

    def compute_loss(self, logits: tf.Tensor, labels: tf.Tensor) -> tf.float32:
        """
        Parameters
        ----------
        logits : ``tf.Tensor``
            A tensor of shape ``(batch_size, num_transitions)`` representing
            logits (unnormalized scores) for the labels for every instance in batch.

        Returns
        -------
        loss : ``tf.float32``
            If input has ``labels``, then mean loss for the batch should
            be computed and set to ``loss`` key in the output dictionary.

        """
        # TODO(Students) Start

        zeros = tf.zeros(logits.shape, dtype=tf.float32)
        # Mask for the correct labels i.e labels=1
        correct_mask = tf.equal(labels, 1)

        # Mask for the feasible labels i.e. labels==0 or 1
        feasible_mask = tf.greater_equal(labels, 0)
        correct_labels_num = tf.where(correct_mask, tf.math.exp(logits), zeros)
        feasible_labels_den = tf.where(feasible_mask, tf.math.exp(logits), zeros)

        softmax = tf.math.divide_no_nan(tf.reduce_sum(correct_labels_num, axis=1),
                                        tf.reduce_sum(feasible_labels_den, axis=1))

        loss = -tf.reduce_mean(tf.math.log(softmax + 1e-10))

        # Add regularization term
        if self._trainable_embeddings:
            regularization = self._regularization_lambda * (
                    tf.nn.l2_loss(self._weight1) + tf.nn.l2_loss(self._bias) + tf.nn.l2_loss(self._weight2) +
                    tf.nn.l2_loss(self.embeddings))
        else:
            regularization = self._regularization_lambda * (
                    tf.nn.l2_loss(self._weight1) + tf.nn.l2_loss(self._bias) + tf.nn.l2_loss(self._weight2))

        # TODO(Students) End
        return loss + regularization
