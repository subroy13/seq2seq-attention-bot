# Making the necessary imports
import tensorflow as tf
import Chatbot_class
import json
import numpy as np
from keras.preprocessing.text import tokenizer_from_json

# load the tokenziers
with open('./processed_data/inp_lang.json', 'r') as f:
    json_data = json.load(f)
    inp_lang = tokenizer_from_json(json_data)
    f.close()

print('Input Language Loaded...')    

with open('./processed_data/targ_lang.json', 'r') as f:
    json_data = json.load(f)
    targ_lang = tokenizer_from_json(json_data)
    f.close()

print('Target Language Loaded...')    

# load the dataset
npzfile = np.load('./processed_data/data.npz')    


# define hyperparameters
BUFFER_SIZE = len(npzfile['arr_0'])
BATCH_SIZE = 64
steps_per_epoch = len(npzfile['arr_0'])//BATCH_SIZE
embedding_dim = 128
units = 256
vocab_inp_size = len(inp_lang.word_index)+1
vocab_tar_size = len(targ_lang.word_index)+1
max_sentence_length = 15

# create tensorflow dataset pipeline for faster processing
dataset = tf.data.Dataset.from_tensor_slices((npzfile['arr_0'], npzfile['arr_1'])).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
print('Loaded dataset into memory...')


# create encoder from Chatbot class
encoder = Chatbot_class.create_encoder(vocab_inp_size, embedding_dim, units, max_sentence_length)
encoder.summary()

# create decoder from Chatbot class
decoder = Chatbot_class.create_decoder(vocab_tar_size, embedding_dim, units, units, max_sentence_length)
decoder.summary()


# there are lots of parameters, so more training would yield better results

optimizer = tf.keras.optimizers.Adam(1e-2)

# the training step function that performs the optimization
@tf.function
def train_step(inp, targ):
    loss = 0

    with tf.GradientTape() as tape:
        enc_output, enc_hidden = encoder(inp)  # pass the input to the encoder, get encoder_output and state
        dec_hidden = enc_hidden   # set the decoder hidden state same as encoder final state
        dec_input = tf.expand_dims([targ_lang.word_index['<start>']] * BATCH_SIZE, 1)

        # Teacher forcing - feeding the target as the next input
        for t in range(1, targ.shape[1]):
            # passing enc_output to the decoder
            predictions, dec_hidden = decoder([enc_output, dec_hidden, dec_input])

            loss += Chatbot_class.loss_func(targ[:, t], predictions)

            # using teacher forcing
            dec_input = tf.expand_dims(targ[:, t], 1)

    batch_loss = (loss / int(targ.shape[1]))
    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))
    return batch_loss



## Here you have the training step
def training(EPOCHS, First_time = False):
    show_output = int(steps_per_epoch/100)
    
    if not First_time:
        encoder.load_weights('./trained_model/encoder_weights.h5')
        decoder.load_weights('./trained_model/decoder_weights.h5')
    
    for epoch in range(EPOCHS):
        print('=' * 50)
        print('EPOCH: ', epoch+1)
        total_loss = 0

        for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
            batch_loss = train_step(inp, targ)
            total_loss += batch_loss

            if batch % show_output == 0:
                print(str(batch/show_output) + '%\t\t Loss: ' + str(batch_loss))

        print('Epoch {} Loss {:.4f}'.format(epoch + 1, total_loss / steps_per_epoch))
      
    # after training save the weights
    encoder.save_weights('./trained_model/encoder_weights.h5')
    decoder.save_weights('./trained_model/decoder_weights.h5')

    
# when performing training for first time, use First_time = True, else First_time = False
training(1, First_time = False) 




    
