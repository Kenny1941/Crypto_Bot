# Crypto_Bot

Uses Open AI Gym environment to create autonomous cryptocurrency bot to trade cryptocurrencies.

Steps to get started using the bot:

1) Sign up for a binance account, using the links below will allow you to use the bots through the binance API:

In the United States:
https://accounts.binance.us/en/register?ref=56432230
            
Outside the United States:
https://accounts.binance.com/en/register?ref=56432230

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
                

6) Make sure you have all necessary packages installed on your computer, if you don't have a package installed you use "pip install **package name**" in the terminal to install the package:

            import os
            import time
            from time import sleep
            import requests
            import random
            from binance.client import Client
            from binance.streams import ThreadedWebsocketManager
            import gym
            import pickle
            import numpy as np
            import math
            import matplotlib.pyplot as plt
            import pandas as pd
            import torch
            import torch.nn as nn
            import torch.nn.functional as F            
            import torch.optim as optim
            

Selecting Bots:

By Default- VALIDATION=True: This means that your file will run on the Validation data and will print a graph comparing each model to the market over the last 1000 Hours.

You can see how each of the five models compares to the market.

By Default:

            CROSSNN_MODEL_ON=True

            MLP_MODEL_ON=True

            FOREST_MODEL_ON=True

            BIG_NN_MODEL_ON=True

            BIG_NN_MODEL_2_ON=True

Which means all 5 models will be used on the validating set.

To turn a model off, switch True to False.

        Ex: Change: BIG_NN_MODEL_ON=True    to     BIG_NN_MODEL_ON=False

**NO TRADES ARE EXECUTED WHEN VALDIATION=True**

To switch to live trading:

            Change VALIDATION=True to VALIDATION=False

            and

            Change LIVE_TRADING=False to LIVE_TRADING=True

We recommend when using live trading only have one bot turned on. i.e. change

            CROSSNN_MODEL_ON=False

            MLP_MODEL_ON=False

            FOREST_MODEL_ON=False

            BIG_NN_MODEL_ON=True

            BIG_NN_MODEL_2_ON=False


Only BIG_NN_MODEL will be used to execute live trades with this configuration. You are free to choose any model you want for live trading and free to use more than one model at a time for live trading. However, as of now BIG_NN_MODEL seems to be performing the best on backtesting and training data.



