# Seq2seq Attention Bot

  It is a chatbot with seq2seq neural network with basic attention mechanism, completely implemented in Python using `Tensorflow 2.0` and `keras` package. Here we use [Cornell Movie Corpus Dataset!](http://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html)
  
  The follwoing steps are needed to be performed to run the chatbot.
1. Choose any version to work with.
2. The data is first needed to be preprocessed using **preprocess.py**. It then creates two languages ( one for questions and one for answers) and saves them. 
3. The file **Chatbot_class.py** contains the python implementation of our seq2seq model using basic RNN and GRU Cell.
4. The file **training.py** needs to be executed to run the training over the dataset. This training step differs between v_1 and v_2. It creates two .pickle file called *encoder* and *decoder* which contians the trained parameters. **Note: This step may take a while. It is recommended to use a GPU to perform the training.**
5. Finally **bot_app.py** can be executed to run the chatbot from console.


In this model, we use Cornell-Corpus Movie dialogue dataset for training the chatbot.
    
1. Extract the .zip file contaiing the Movie Dialogue dataset.
  
2. Run `preprocess.py` create the language specific files and the tokenizers, where we store all the language information. 
  
3. Run `Training.py` to train the dataset
  
4. If you run it for the first time, set `first_time = True`, it will save the encoder and decoder after training
  
5. If you want to train it more, set `first_time = False`, it will load the poreviously saved encoder and decoder and start training from it.
  
6. You need to run `bot_app.py` after training to run the chatbot in an interactive mode from console.

7. `bot_app.py` implements 3 different sampling schemes in order to generate specific replies to the query.

  
