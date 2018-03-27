  In this model, we use Cornell-Corpus Movie dialogue dataset for training the chatbot.
    
  -- Extract the .zip file
  
  -- Run preprocess.py create the lang_dict, where we store all the language information. 
  
  -- Run Training.py to train the dataset
  
  ** If you run it for the first time, set first_time = True, it will save the encoder and decoder after training
  
  ** If you want to train it more, set first_time = False, it will load the poreviously saved encoder and decoder and start training from it.
  
 -- You need to run bot_app.py after training to run the chatbot in an interactive mode.
