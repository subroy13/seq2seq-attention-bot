# Making the necessary imports
import tensorflow as tf

def create_encoder(vocab_size, embedding_dim, enc_units, max_sentence_length):
    """
    Creates an encoder with the necessary vocabulary size, embedding dimension and encoding units
    The model is very simple:
    
    Input -> Embedding -> GRU Layer -> Hidden Vector output which is to be passed to decoder
    """
    inputs = tf.keras.layers.Input((max_sentence_length, ))
    embed = tf.keras.layers.Embedding(vocab_size, embedding_dim)(inputs)
    output, state = tf.keras.layers.GRU(enc_units, 
                                        return_sequences=True, 
                                        return_state=True, 
                                        recurrent_initializer='glorot_uniform')(embed)
    encoder = tf.keras.models.Model(inputs, [output, state])
    return encoder


class BahdanauAttention(tf.keras.layers.Layer):
    """
    Create a layer that implements Bahadanau attention mechanism
    More details at https://www.tensorflow.org/tutorials/text/nmt_with_attention
    """
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)
        self.R = tf.keras.layers.Dropout(0.25)

    def call(self, query, values):
        # hidden_size = num_units
        # hidden shape == (batch_size, hidden size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden size)
        # we are doing this to perform addition to calculate the score
        hidden_with_time_axis = tf.expand_dims(query, 1)

        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, max_length, units)
        score = self.V(tf.nn.tanh(self.W1(values) + self.W2(hidden_with_time_axis)))
        
        score = self.R(score)  # perform regularization

        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector

    
def create_decoder(vocab_size, embedding_dim, enc_units, dec_units, max_sentence_length):
    """
    Implements a simple Decoder model. Network is as follows:
    
    Encoder Output                  Hidden State of Encoder                 Decoder input of previous word
              \                         /                                            |                                                      
                  Bahadanau Attetion                                             Embedding
                        |                                                            | 
                    Context Vector                                                  /
                                 \                                                 /
                                            Concatenated Input
                                                 |
                                             GRU Cell
                                                 |
                                            Dense Layer
                                                 |
                                Predicted probabilities on the vocabulary
        
    """
    enc_output = tf.keras.layers.Input((max_sentence_length, enc_units, ))
    hidden = tf.keras.layers.Input((enc_units, ))
    context_vector = BahdanauAttention(dec_units)(hidden, enc_output)
    
    dec_input = tf.keras.layers.Input((1, ))
    dec_embed = tf.keras.layers.Embedding(vocab_size, embedding_dim)(dec_input)
    
    context_expand = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, 1))(context_vector)
    full_context = tf.keras.layers.Concatenate(axis = -1)([context_expand, dec_embed])
    
    output, state = tf.keras.layers.GRU(dec_units, 
                                        return_sequences=True, 
                                        return_state=True, 
                                        recurrent_initializer='glorot_uniform')(full_context)
    flat_output = tf.keras.layers.Flatten()(output)
    final = tf.keras.layers.Dense(vocab_size, activation = 'softmax')(flat_output)
    
    decoder = tf.keras.models.Model([enc_output, hidden, dec_input], [final, state])
    return decoder


def loss_func(target, pred):
    """
    Returns the total loss over all batch, using categorical crossentropy
    """
    return tf.math.reduce_sum(tf.keras.losses.sparse_categorical_crossentropy(target, pred))