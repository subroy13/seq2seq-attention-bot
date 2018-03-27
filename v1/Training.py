import random
import torch
from torch.autograd import Variable
from torch import optim
import pickle

import time
import math

import Chatbot_class
from preprocess import Lang
from preprocess import variableFromSentence

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def variablesFromPair(pair):
    input_variable = variableFromSentence(input_lang, pair[0])
    target_variable = variableFromSentence(output_lang, pair[1])
    return (input_variable, target_variable)

        

def trainIters(n_iters, print_every = 1000, save_every = 1000, learning_rate = 0.01, first_time = True):

    start = time.time()
    print_loss_total = 0 #reset every print_every

    if first_time:
        encoder = Chatbot_class.EncoderRNN(input_lang.n_words, hidden_size)
        decoder = Chatbot_class.AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p = 0.1)
        if use_cuda:
            encoder = encoder.cuda()
            decoder = decoder.cuda()
    
    else:
        print('loading Encoder...')
        with open('encoder','rb') as f:
            encoder = pickle.load(f)
            f.close()
        print('Encoder loaded')
        print('loading Decoder...')
        with open('decoder','rb') as f:
            decoder = pickle.load(f)
            f.close()
        print('Decoder loaded')
    

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_pairs = [variablesFromPair(random.choice(pairs)) for i in range(n_iters)]
    criterion = torch.nn.NLLLoss()

    for iteration in range(1, n_iters +1):
        training_pair = training_pairs[iteration -1]
        input_variable = training_pair[0]
        target_variable = training_pair[1]

        loss = Chatbot_class.train(input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion) 
        
        print_loss_total += loss

        if iteration % print_every == 0:   
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iteration / n_iters), iteration, iteration / n_iters * 100, print_loss_avg))
            answer = Chatbot_class.evaluate(encoder, decoder, "Who are you?", input_lang, output_lang)
            print(answer)

        if iteration % save_every == 0:
            with open('encoder','wb') as f:
                pickle.dump(encoder, f)
                f.close()

            with open('decoder','wb') as f:
                pickle.dump(decoder, f)
                f.close()



if __name__ == '__main__':
    with open('lang_dict','rb') as f:
        lang_dict = pickle.load(f)
        input_lang = lang_dict['input']
        output_lang = lang_dict['output']
        pairs = lang_dict['pairs']
        f.close()
    print('data loaded')

    hidden_size = 256
    dropout_p = 0.1
    max_seq_length = 10
    use_cuda = torch.cuda.is_available()
    print('Running trainiters')

    trainIters(1000, print_every = 500, save_every = 5000, first_time = False)
















