# seq2seq_attention_bot

  It is a chatbot with seq2seq neural network with basic attention mechanism, completely implemented in Python using Pytorch module. Here we use [Cornell Movie Corpus Dataset!](http://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html)
  
  The follwoing steps are needed to be performed to run the chatbot.
1. Choose any version to work with.
2. The data is first needed to be preprocessed using **preprocess.py**. It then creates two languages ( one for questions and one for answers) and saves them into the working directory as .pickle file. 
3. The file **Chatbot_class.py** contains the python implementation of our seq2seq model using basic RNN and GRU Cell.
4. The file **training.py** needs to be executed to run the training over the dataset. This training step differs between v_1 and v_2. It creates two .pickle file called *encoder* and *decoder* which contians the trained parameters. **Note: This step may take a while. It is recommended to use a GPU to perform the training.**
5. Finally **bot_app.py** can be executed to run the chatbot from console.


  
