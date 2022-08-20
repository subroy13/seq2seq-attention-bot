# Making the necessary imports
import tensorflow as tf
import Chatbot_class
from preprocess import clean_text
import numpy as np
import json
from keras.preprocessing.text import tokenizer_from_json

# load the tokenziers
with open('./processed_data/inp_lang.json', 'r') as f:
    json_data = json.load(f)
    inp_lang = tokenizer_from_json(json_data)
    f.close()
    
with open('./processed_data/targ_lang.json', 'r') as f:
    json_data = json.load(f)
    targ_lang = tokenizer_from_json(json_data)
    f.close()

# define hyperparameters
embedding_dim = 128
units = 256
vocab_inp_size = len(inp_lang.word_index)+1
vocab_tar_size = len(targ_lang.word_index)+1
max_sentence_length = 15


# create encoder from Chatbot class
encoder = Chatbot_class.create_encoder(vocab_inp_size, embedding_dim, units, max_sentence_length)
print('Encoder model initialized...')
encoder.load_weights('trained_model/encoder_weights.h5')  # load the weights, we shall use them to make inference
print('Encoder model trained weights loaded...')

# create decoder from Chatbot class
decoder = Chatbot_class.create_decoder(vocab_tar_size, embedding_dim, units, units, max_sentence_length)
print('Decoder model initialized...')
decoder.load_weights('trained_model/decoder_weights.h5')
print('Decoder model trained weights loaded...')


def evaluate(sentence, samp_type = 1):
    sentence = clean_text(sentence)
    inputs = []
    # split the sentence and replace unknown words by <unk> token.
    for i in sentence.split(' '):
        try:
            inputs.append(inp_lang.word_index[i])
        except KeyError:
            inputs.append(inp_lang.word_index['<unk>'])
    
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs], maxlen = max_sentence_length, padding='post')
    inputs = tf.convert_to_tensor(inputs)

    result = ''
    enc_output, enc_hidden = encoder(inputs)
    
    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([targ_lang.word_index['<start>']], 0)
    
    for t in range(max_sentence_length):
        predictions, dec_hidden = decoder([enc_output, dec_hidden, dec_input])
        if samp_type == 1:
            # that means simple greedy sampling
            predicted_id = tf.argmax(predictions[0]).numpy()
        elif samp_type == 2:
            predicted_id = np.random.choice(vocab_tar_size, p = predictions[0].numpy())
        elif samp_type == 3:
            _ , indices = tf.math.top_k(predictions[0], k = 3)
            predicted_id = np.random.choice(indices)

        if predicted_id!= 0:
            if targ_lang.index_word[predicted_id] == '<end>':
                return result, sentence
            else:
                result += targ_lang.index_word[predicted_id] + ' '
        
        # the predicted ID is fed back into the model
        dec_input = tf.expand_dims([predicted_id], 0)

    return result, sentence


print('=' * 50)
print('+' * 15 + ' CHATBOT v2.0 ' + '+' * 15)
print('=' * 50)

print('\nThere are different sampling techniques which might result in different text generation.')
print('Input 1 for => Greedy Sampling')
print('Input 2 for => Probability Proportional Sampling')
print('Input 3 for => Top-3 Sampling')
samp_type = int(input('Input your preferred sampling choice: '))

if samp_type != 1 and samp_type != 2 and samp_type != 3:
    raise NotImplementedError

while True:
    inputs = input('User :> ')
    if inputs == 'quit' or inputs == 'Quit':
        break
    result, sentence = evaluate(inputs, samp_type)
    print('Bot :> ' + result)





    
