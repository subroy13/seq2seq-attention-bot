import re
import torch
from torch.autograd import Variable
import pickle

max_sentence_length = 10
use_cuda = torch.cuda.is_available()
EOS_token = 1
SOS_token = 0

class Lang:

    def __init__(self, name):
        self.name = name
        self.word2index = {}   #vocabtoint dictionary
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  #count SOS and EOS already


        

    def addSentence(self, sentence):
        #sentence = clean_text(sentence)
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            #therefore word is not in dictionary
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            #the word is in vocabulary, so increase its count
            self.word2count[word] += 1


def clean_text(text):
    #clean the text by removing unneccessary characters and abbreviations
    text = text.lower()
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "that is", text)
    text = re.sub(r"where's", "where is", text)
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
    text = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", text)
    return text



def filterPair(p):
    return len(p[0].split(' ')) < max_sentence_length and \
        len(p[1].split(' ')) < max_sentence_length


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]


def preprocess():
    lines = open('movie_lines.txt',encoding='utf-8', errors = 'ignore').read().split('\n')
    conv_lines = open('movie_conversations.txt', encoding='utf-8', errors='ignore').read().split('\n')
    # Create a dictionary to map each line's id with its text
    id2line = {}
    for line in lines:
        _line = line.split(' +++$+++ ')
        if len(_line) == 5:
            id2line[_line[0]] = _line[4]
    # Create a list of all of the conversations' lines' ids.
    convs = [ ]
    for line in conv_lines[:-1]:
        _line = line.split(' +++$+++ ')[-1][1:-1].replace("'","").replace(" ","")
        convs.append(_line.split(','))
    # Sort the sentences into questions (inputs) and answers (targets)
    pairs = []
    for conv in convs:
        for i in range(len(conv)-1):
            pairs.append([clean_text(id2line[conv[i]]), clean_text(id2line[conv[i+1]])])

    input_lang = Lang('questions')
    output_lang = Lang('answers')
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        #print(pair)
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs



def indexesFromSentence(lang, sentence, evaluate):
    if not evaluate:
        return [lang.word2index[word] for word in sentence.split(' ')]
    else:
        result_vector = []
        for word in sentence.split(' '):
            try:
                result_vector.append(lang.word2index[word])
            except:
                result_vector.append(SOS_token)
        return result_vector

def variableFromSentence(lang, sentence, evaluate = False):
    indexes = indexesFromSentence(lang, sentence, evaluate)
    indexes.append(EOS_token)
    result = Variable(torch.LongTensor(indexes).view(-1, 1))
    if use_cuda:
        return result.cuda()
    else:
        return result


if __name__=='__main__':
    input_lang, output_lang, pairs = preprocess()
    lang_dict = {'input': input_lang, 'output': output_lang, 'pairs': pairs}
    with open('lang_dict','wb') as g:
        pickle.dump(lang_dict, g)
        g.close()
    print('Data preprocessed')






























