import _pickle as pickle
from Chatbot_class import evaluate
from preprocess import Lang

print('Loading encoder...')
print('Please wait...')
with open('encoder','rb') as f:
    encoder = pickle.load(f)
    f.close()
print('Encoder loaded')
print('Loading decoder...')
print('Please wait...')
with open('decoder','rb') as f:
    decoder = pickle.load(f)
    f.close()
print('Decoder loaded')

print('Loading data...')
with open('lang_dict','rb') as f:
    lang_dict = pickle.load(f)
    input_lang = lang_dict['input']
    output_lang = lang_dict['output']
    f.close()
print('Data loaded')
print('Chatbot deployed')
print()

input_query = str(input(':: '))
while (input_query.lower() != 'quit' and input_query.lower() != 'exit'):
    answer = evaluate(encoder, decoder, input_query, input_lang, output_lang)
    reply = ':>'
    for word in answer:
        if word!='<EOS>':
            reply = (reply + ' ' + word)
    print(reply)        
    input_query = str(input(':: '))

