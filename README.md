# Crypto_Bot

Uses Open AI Gym environment to create autonomous cryptocurrency bot to trade cryptocurrencies.

Steps to get started using the bot:

1) Sign up for a binance account, you can use this link to create your account:
            https://accounts.binance.us/en/register?ref=56432230

2) After Creating an account, go to "API Management" to get your API keys:
          
          You will receive two API Keys:
          
                  API Key
                  
                  Secret Key-Make sure to save secret key as you will only see this key once.
                  
                  
3) Download Crypto_Bot_Class.py File from GitHub
            In Crypto_Bot_Class.py file replace "API_KEY" Variable with your API Key from binance
            In Crypto_Bot_Class.py file replace "SECRET_KEY" Variable with your Secret Key from binance

4) Download Model Files from Model Folder on Github
            In Crypto_Bot_Class.py file replace each:
            
                  XGB_MODEL= **Your XGB_Model File Path**
                  
                  CROSSNN_MODEL= **Your CrossNN_Model File Path**
                  
                  MLP_MODEL= **Your MLP_Model File Path**
                  
                  FOREST_MODEL=**Your Forest_Model File Path**
                  
                  BIG_NN_MODEL=**Your BIG_NN_Model File Path**
                  

5) Download Data Files from Data Folder on Github
            In Crypto_Bot_Class.py file replace each:
                VAL_SET=**Your VALIDATION DATA File Path**
                TRAIN_SET**Your TRAIN DATA File Path**
                

How to select which bots to use:
    By Default- VALIDATION=True:
          This means that your file will run on the Validation data and will print a graph comparing each model to the market over the last three months. 
    
    If you change: VALIDATION=False then the each of the 5 Models will be used to make live trades through the binance API. You can select which model you want to make trades by     changing the models from True to False:
    
        Ex: Change: XGB_MODEL_ON=True    to     XGB_MODEL_ON=False
        
 

