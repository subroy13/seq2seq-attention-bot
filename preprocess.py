# Making the necessary imports
import tensorflow as tf
import re
import numpy as np
import json


def clean_text(text):
    """
    A function that cleans the text by removing the common abbreviations and unwanted characters or puntuations
    It also ends up adding a <start> tag at the beginning of the text and
    and <end> tag at the last of the text
    """
    text = text.lower().strip()   # lowercase and remove trailing whitespaces
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"there's", "there is", text)
    text = re.sub(r"how's", "how is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"n'", "ng", text)
    text = re.sub(r"'bout", "about", text)
    text = re.sub(r"'til", "until", text)
    text = re.sub(r'[" "]+', " ", text)   # remove extra spaces in between
    text = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", text)
    text = '<start> ' + text + ' <end>'
    return text


def preprocess(dataset_folder_path, len_bound, num_examples = None):
    """
    It reads the required files, creates questions and answers based on the conversations.
    """
    min_sentence_length = len_bound[0]
    max_sentence_length = len_bound[1]
    
    lines = open(str(dataset_folder_path) + '/movie_lines.txt',encoding='utf-8', errors = 'ignore').read().split('\n')
    conv_lines = open(str(dataset_folder_path) + '/movie_conversations.txt', encoding='utf-8', errors='ignore').read().split('\n')


    # Create a dictionary to map each line's id with its text
    id2line = {}
    sent_len = {}   # create a dictionary to contain sentence lengths
    
    for line in lines:
        _line = line.split(' +++$+++ ')
        if len(_line) == 5:
            speech = clean_text(_line[4])
            id2line[_line[0]] = speech
            sent_len[_line[0]] = len(speech.split(' '))


    # Create a list of all of the conversations' lines' ids.
    convs = [ ]
    for line in conv_lines[:-1]:
        _line = line.split(' +++$+++ ')[-1][1:-1].replace("'","").replace(" ","")
        convs.append(_line.split(','))

    # Sort the sentences into questions (inputs) and answers (targets)
    input_lang = []
    output_lang = []
    if num_examples is not None:
        convs = convs[:num_examples]
    for conv in convs:
        for i in range(len(conv)-1):
            if (sent_len[conv[i]] <= max_sentence_length   and 
                sent_len[conv[i+1]] <= max_sentence_length and 
                sent_len[conv[i]] >= min_sentence_length   and 
                sent_len[conv[i+1]] >= min_sentence_length ):
                # we do not use very long sentences
                input_lang.append(id2line[conv[i]])
                output_lang.append(id2line[conv[i+1]])

    assert len(input_lang) == len(output_lang)
    print("Read %s sentence pairs" % len(input_lang))
        
    return (input_lang, output_lang)


def tokenize(lang, oov=True):
    """
    Tokenize sentences into words, and correspondingly create an index based representation for vocabulary
    """
    if oov:
        lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='', oov_token = '<unk>')
    else:
        lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
    lang_tokenizer.fit_on_texts(lang)
    tensor = lang_tokenizer.texts_to_sequences(lang)
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')
    return tensor, lang_tokenizer


def load_dataset(dataset_folder_path, len_bound, num_examples = None):
    # creating cleaned input, output pairs
    targ_lang, inp_lang = preprocess(dataset_folder_path, len_bound, num_examples)

    input_tensor, inp_lang_tokenizer = tokenize(inp_lang, oov = True)   # in the input language, we allow OOV words
    target_tensor, targ_lang_tokenizer = tokenize(targ_lang, oov = False)   # in the output language, we do not allow OOV words

    return input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer


if __name__ == "__main__":
    dataset_folder_path = './cornell movie-dialogs corpus'   # the path to the folder 
    len_bounds = [2, 15]   # minimum and maximum permissible length of a sentence to be considered.
    
    input_tensor, target_tensor, inp_lang, targ_lang = load_dataset(dataset_folder_path, len_bounds, num_examples = None)
    
    inp_lang_json = inp_lang.to_json()
    targ_lang_json = targ_lang.to_json()
    
    with open('processed_data/inp_lang.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(inp_lang_json, ensure_ascii=False))
        f.close()
    print('Input Language Tokenizer saved...')
        
    with open('processed_data/targ_lang.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(targ_lang_json, ensure_ascii=False))
        f.close()
    print('Target Language Tokenizer saved...')
        
    np.savez('processed_data/data.npz', input_tensor, target_tensor)
    print('Final Dataset saved...')
    
    
