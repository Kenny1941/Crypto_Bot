# -*- coding: utf-8 -*-
"""
Created on Sat Dec 11 01:44:47 2021
@author: wkjon
"""

import os

import time
import datetime
from time import sleep
import requests
import random

from binance.client import Client
from binance.streams import ThreadedWebsocketManager


#Each Crypto Bot is an Open AI Agent uses Wrapper from Open AI to change methods
import gym

#Models used for determining Crypto AI Bot Actions are saved as Pickle files
import pickle

#Basic Imports
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd


#Imports for Neural Network Models using Pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

'''
This File is the basic implementation of the crypto-bot. Each Crypto-Bot is an Agent acting in the
Open AI Gym Enviroment (Mountain-Car). We use Open AI Gym Wrapper to change action, step, observation, and reward methods.
By changing these methods, each agent, AKA each crypto-bot, acts by sending Buy or Sell requests
through the binance API
Each Agents Basic Structure can be found in the BasicWrapper Class:
    For each crypto-currency the below steps are taken
        1) Observation
           ---> Pulls data from Binance API by calling get_kline()
           
        2) get_act
          ---> calls process_data which organizes data into dataframe structure with same
               format as training data
          ---> Uses Pre-trained Model to make prediction about new hour's price
          ---> Uses Threshold to determine whether agent should buy, sell, or hold 
               
        3) get_action
          ---> Executes a trade through binance API based on get_act prediction
          
        4) get_reward
          ---> returns net profit from all trades from start to current time
               
Important Notes:
If VALIDATION==True
    --->This will run the model using backtesting data, last 1000 hours of data. 
 If LIVE_TRDAING==True
    --->This will make live trades through binance API
'''
#Download Following Models from the "Model" Folder on Github:
        #Cross_NN_Model.sav
        #Big_NN_Model.pth
        #Big_NN_Model_2.pth
        #Forest_Model.sav
        #MLP_Model.sav
#Copy file path to variables below

CROSSNN_MODEL= **Your Cross_NN_Model.sav File Path** #Ex: r'C:\Users\wkjon\.spyder-py3\Cross_NN_Model.sav'
MLP_MODEL=**Your MLP_Model.sav File Path** #Ex: r'C:\Users\wkjon\.spyder-py3\MLP_Model.sav'
FOREST_MODEL=**Your Forest_Model.sav File Path** #Ex: r'C:\Users\wkjon\.spyder-py3\Forest_Model.sav'
BIG_NN_MODEL=**Your Big_NN_Model.pth File Path** #Ex: r'C:\Users\wkjon\.spyder-py3\Big_NN\Big_NN_Model.pth'
BIG_NN_MODEL_2=**Your Big_NN_Model_2.pth File Path** #Ex: r'C:\Users\wkjon\.spyder-py3\Big_NN\Big_NN_Model_2.pth'


#Download the data sets "Validation_Data" and "Training_Data" from Github Folder "Data". For each data set save the file path 
VAL_SET=pd.read_csv(**Your Validation_Data File Path**) #Ex: pd.read_csv(r'C:\Users\wkjon\.spyder-py3\Validation_Data.csv')
TRAIN_SET=pd.read_csv(**Your Training_Data File Path**) #Ex: pd.read_csv(r'C:\Users\wkjon\.spyder-py3\Training_Data.csv')

#Quantity per trade in dollar amount
#MOST AMOUNT OF MONEY IN MARKET AT ANY TIME WILL BE 3 x QUANTITY X NUMBER OF MODELS BEING USED
#i.e. if QUANTITY=100 and you are using 2 Models then most amount of money in market is $100 x 3 x 2 = $300
#If Quantity is set too low (below 20), API will not be able to execute trades
QUANTITY=100

#These variables control which dataset is being used. 
#If TRAIN=True: Model will record results over the training set.
#If VALIDATION=True: Model will record results over the validating set. 
#If LIVE_TRADING=True: Model will execute real trades through binance API.

#**Only one of below variables should be set to true at a time
TRAIN=False 
VALIDATION=True
LIVE_TRADING=False


#The below variables determine which Models are being used. 
#Set to False if you do not want to use the model.
#We Recommend only using one Model when LIVE_TRADING=True, currently BIG_NN_MODEL performs best

CROSSNN_MODEL_ON=True
MLP_MODEL_ON=True
FOREST_MODEL_ON=True
BIG_NN_MODEL_ON=True
BIG_NN_MODEL_2_ON=True



##############################################################################
#THIS SECTION CREATES CONNECTION WITH BINANCE API

#REPLACE THESE KEYS WITH YOUR OWN KEYS FROM BINANCE.US OR BINANCE.COM
API_Key=**Your API Key** 
Secret_Key=**Your Secret Key**

#IF USING BINANCE.US USE:
client = Client(API_Key, Secret_Key, tld='us')

#IF USING BINANCE.COM USE:
#client = Client(Test_Key, Test_Secret_Key)

#Should print summary of account
print('Account Summary')
print(client.get_account())

##############################################################################
##############################################################################

#DO NOT NEED TO CHANGE ANYTHING BELOW THIS TO HAVE BOT WORK

##############################################################################
##############################################################################

#This Setion Gets Raw Data from Binance, Validating Sets, and Training Sets

#List of coins being traded
ticker_list=['BTC', 'ETH', 'LTC']

#Structure of client.get_historical_klines as a dictionary object

    #1499040000000,      // Open time
    #"0.01634790",       // Open
    #"0.80000000",       // High
    #"0.01575800",       // Low
    #"0.01577100",       // Close
    #"148976.11427815",  // Volume
    #1499644799999,      // Close time
    #"2434.19055334",    // Quote asset volume
    #308,                // Number of trades
    #"1756.87402397",    // Taker buy base asset volume
    #"28.46694368",      // Taker buy quote asset volume
    #"17928899.62484339" // Ignore


#Pulls most current hourly data when LIVE_TRADING=True
def get_kline():
    tick_interval = '1h'
    ticker_list=['BTCUSDT', 'ETHUSDT', 'LTCUSDT']
    price_dict={}
    for ticker in ticker_list:
        market = ticker
        url = 'https://api.binance.com/api/v3/klines?symbol='+market+'&interval='+tick_interval+'&limit=25'
        klines = requests.get(url).json()
        klines = pd.DataFrame(klines, columns=['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 'Quote asset volume',
                     'Number of trades', 'Taker buy base asset volume', 'Taker buy quote asset volume', 'Junk'])
        price_dict.update({ticker: klines})
    return price_dict

#Pulls last 1000 hours of candlestick data
def get_kline_2():
    tick_interval = '1h'
    ticker_list=['BTCUSDT', 'ETHUSDT', 'LTCUSDT',]
    price_dict={}
    df=pd.DataFrame()
    for ticker in ticker_list:
        market = ticker
        url = 'https://api.binance.com/api/v3/klines?symbol='+market+'&interval='+tick_interval+'&limit=1000'
        klines = requests.get(url).json()
        klines = pd.DataFrame(klines, columns=['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 'Quote asset volume',
                     'Number of trades', 'Taker buy base asset volume', 'Taker buy quote asset volume', 'Junk'])
        
        klines['Coin']=ticker
        df=pd.concat([df, klines])
    return df

#Pulls Last 1000 hours as a backtesting set when VALIDATION=True
def get_kline_val(val_set, count):
    ticker_list=['BTCUSDT', 'ETHUSDT', 'LTCUSDT']
    price_dict={}
    for ticker in ticker_list:
        temp=val_set[val_set['Coin']==ticker]
        temp.index=range(len(temp.index))
        temp=temp.loc[count-24:count,:]
        klines=pd.DataFrame(temp,columns=['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time',
       'Quote asset volume', 'Number of trades', 'Taker buy base asset volume',
       'Taker buy quote asset volume', 'Junk'])
        price_dict.update({ticker: klines}) 
    return price_dict


#IGNORE, an extra validating set with 3 months of data 
def get_kline_val_2(val_set, count):
    ticker_list=['BTC', 'ETH', 'LTC']
    price_dict={}
    for ticker in ticker_list:
        temp=val_set[val_set['Coin']==ticker]
        temp.index=range(len(temp.index))
        temp=temp.loc[count-24:count,:]
        klines=pd.DataFrame(columns=['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time',
       'Quote asset volume', 'Number of trades', 'Taker buy base asset volume',
       'Taker buy quote asset volume', 'Junk'])
        klines['Open time']=temp['time']
        klines['Open']=temp['open']
        klines['High']=temp['high']
        klines['Low']=temp['low']
        klines['Close']=temp['close']
        klines['Volume']=temp['volumefrom']
        price_dict.update({ticker+'USDT': klines}) 
    return price_dict



#pulls training data when TRAIN=True 
def get_kline_train(val_set, count):
    ticker_list=['BTC', 'ETH', 'LTC']
    price_dict={}
    target={}
    for ticker in ticker_list:
        temp=val_set[val_set['Coin']==ticker]
        temp=temp.iloc[::-1]
        temp.index=range(len(temp.index))
        temp_1=temp.loc[count-24:count,:]
        temp_2=temp.loc[count+1,:]
        target_temp=(temp_2.loc['close']-temp_2.loc['open'])/temp_2.loc['open']
        klines=pd.DataFrame(columns=['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time',
       'Quote asset volume', 'Number of trades', 'Taker buy base asset volume',
       'Taker buy quote asset volume', 'Junk'])
        klines['Open time']=temp_1['time']
        klines['Open']=temp_1['open']
        klines['High']=temp_1['high']
        klines['Low']=temp_1['low']
        klines['Close']=temp_1['close']
        klines['Volume']=temp_1['volumefrom']
        price_dict.update({ticker+'USDT': klines}) 
        target.update({ticker+'USDT': target_temp})
    return price_dict, target

##############################################################################
#This Section is the Bot, BasicWrapper is generic version

class BasicWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.keys={'BTCUSDT':0, 'ETHUSDT':0, 'LTCUSDT':0}
        keys={'BTCUSDT':0, 'ETHUSDT':0, 'LTCUSDT':0}
        self.Owned=keys.copy()
        self.bought_price={'BTCUSDT':1, 'ETHUSDT':1, 'LTCUSDT':1}
        self.sell_price=keys.copy()
        self.current_price=keys.copy()
        self.Quantity={'BTCUSDT': (10/float(self.bought_price['BTCUSDT'])), 'ETHUSDT': QUANTITY/(self.bought_price['ETHUSDT']), 'LTCUSDT': QUANTITY/(self.bought_price['LTCUSDT'])}
        self.profit=0
        self.rewards=keys.copy()
        self.act=keys.copy()
        self.actions=self.keys.copy()
        self.klines=None
        self.val_set_hour=get_kline_2()
        self.train_set_hour=TRAIN_SET
        self.validation=VALIDATION
        self.train=TRAIN
        self.Live_Trading=LIVE_TRADING
        self.count=24
        self.target=keys.copy()

#Processes data from raw candlestick data to dataset including time-series data, averages, and percent change
    def process_data(self, key):
        df=self.klines[key]
        df.index=range(len(df))
        self.current_price[key]=df.loc[len(df)-1,'Close']
        row_df=pd.DataFrame(index=[0], columns=['0 Back Percent Volume', '0 Back Percent Changes',
       '1 Back Percent Volume', '1 Back Percent Changes',
       '2 Back Percent Volume', '2 Back Percent Changes',
       '3 Back Percent Volume', '3 Back Percent Changes',
       '4 Back Percent Volume', '4 Back Percent Changes',
       '5 Back Percent Volume', '5 Back Percent Changes',
       '6 Back Percent Volume', '6 Back Percent Changes',
       '7 Back Percent Volume', '7 Back Percent Changes',
       '8 Back Percent Volume', '8 Back Percent Changes',
       '9 Back Percent Volume', '9 Back Percent Changes',
       '10 Back Percent Volume', '10 Back Percent Changes',
       '11 Back Percent Volume', '11 Back Percent Changes',
       '12 Back Percent Volume', '12 Back Percent Changes',
       '13 Back Percent Volume', '13 Back Percent Changes',
       '14 Back Percent Volume', '14 Back Percent Changes',
       '15 Back Percent Volume', '15 Back Percent Changes',
       '16 Back Percent Volume', '16 Back Percent Changes',
       '17 Back Percent Volume', '17 Back Percent Changes',
       '18 Back Percent Volume', '18 Back Percent Changes',
       '19 Back Percent Volume', '19 Back Percent Changes',
       '20 Back Percent Volume', '20 Back Percent Changes',
       '21 Back Percent Volume', '21 Back Percent Changes',
       '22 Back Percent Volume', '22 Back Percent Changes',
       '23 Back Percent Volume', '23 Back Percent Changes',
       '24 Average Volume', '12 Average Volume', '6 Average Volume',
       '3 Average Volume', '24 Average Percent Changes',
       '12 Average Percent Changes', '6 Average Percent Changes',
       '3 Average Percent Changes'], dtype=float)
        
        
        df['Volume'] = df['Volume'].astype(float)
        row_df['24 Average Volume']=df.loc[:,'Percent Volume'].mean()
        row_df['12 Average Volume']=df.loc[13:, 'Percent Volume'].mean()
        row_df['6 Average Volume'] = df.loc[19:, 'Percent Volume'].mean()
        row_df['3 Average Volume'] = df.loc[22:, 'Percent Volume'].mean()
        row_df['24 Average Percent Changes']=df['Percent Changes'].mean()
        row_df['12 Average Percent Changes']=df.loc[13:, 'Percent Changes'].mean()
        row_df['6 Average Percent Changes'] = df.loc[19:, 'Percent Changes'].mean()
        row_df['3 Average Percent Changes'] = df.loc[22:, 'Percent Changes'].mean()
        for i in range(0,len(df)-1):
            k=(len(df))-i-1
            row_df[str(i) + ' Back Percent Volume'] = df.loc[k, 'Percent Volume']
            row_df[str(i) + ' Back Percent Changes'] = df.loc[k, 'Percent Changes']
        return row_df


    def step(self):
        info=self.observation()
        next_state=self.action()
        reward=self.reward()
        return next_state, reward, info

#Pulls Raw Data
    def observation(self):
        if self.validation==True:
            self.klines=get_kline_val(self.val_set_hour,self.count)
            self.count+=1
        elif self.train==True:
            self.klines, self.target=get_kline_train(self.train_set_hour,self.count)
            self.count+=1
            if self.count>36000:
                self.count=24
        else:
            self.klines=get_kline()
        for key in self.klines:
            prices_new=self.klines[key].loc[:, 'Close'].astype(float)
            Volume = self.klines[key].loc[:, 'Volume'].astype(float)
            price_old=prices_new.shift(1)
            Volume_old = Volume.shift(1)
            percent_change=(prices_new-price_old)/price_old
            percent_change_volume = (Volume - Volume_old) / Volume_old
            self.current_price[key]=prices_new.iloc[-1]
            self.klines[key]['Percent Changes']=percent_change
            self.klines[key]['Percent Volume'] = percent_change_volume
        return

#Calculates Profit and Reward Function for Training 
    def reward(self):
        reward=0
        for key in self.keys:
            if self.Owned[key]==1:
                reward+=(float(self.current_price[key])-self.bought_price[key])*self.Quantity[key]
        reward+=self.profit
        self.rewards=self.target
        return reward

#Executes Trades 
    def action(self):
        act=self.get_act()
        for key in self.keys:
            #Buying Section
            if self.act[key]==1 and self.Owned[key]==0:
                
                #Buy Order for Validation and Training 
                if self.validation==True or self.train==True:                    
                    self.bought_price[key]=float(self.current_price[key])
                    self.Quantity[key]=round(QUANTITY/self.bought_price[key],4)
                
                #Buy Order for Live Trading 
                elif self.Live_Trading==True:                    
                    self.bought_price[key]=float(self.current_price[key])
                    self.Quantity[key]=round(QUANTITY/self.bought_price[key],4)
                    buy_order = client.create_order(symbol=key, side='BUY', type='MARKET', quantity=self.Quantity[key])
                    if len(buy_order['fills'])==0:
                        print('NOT FILLED BUY '+str(key))
                        client.cancel_orders(symbol=key)
                        continue
                    self.bought_price[key]=float(buy_order['fills'][0]['price'])
                    self.Quantity[key]=float(buy_order['executedQty'])
                    print('Bought '+str(self.Quantity[key])+' '+ key +' at $'+ str(self.bought_price[key]))                
                self.Owned[key]=1
            
            #Selling Section                  
            elif self.act[key]==0 and self.Owned[key]==1:
                
                #Sell Order for Validation and Training
                if self.validation==True or self.train==True:
                    self.sell_price[key]=float(self.current_price[key])
                
                #Sell Order for Live Trading
                elif self.Live_Trading==True:
                    sell_order = client.create_order(symbol=key, side='SELL', type='MARKET', quantity=self.Quantity[key])
                    if len(sell_order['fills'])==0:
                        print('NOT FILLED SELL '+str(key))
                        continue
                    self.sell_price[key]=float(sell_order['fills'][0]['price'])
                    print('Sold '+str(self.Quantity[key])+' '+ key +' at $'+ str(self.bought_price[key]))
                    profit=(self.sell_price[key]-self.bought_price[key])*self.Quantity[key]
                    print('For Profit of: $'+ str(profit))
                
                profit=(self.sell_price[key]-self.bought_price[key])*self.Quantity[key]
                self.profit+=profit  
                self.Owned[key]=0
        return act

#Determines whether buy, sell, or hold should take place
#Determined by the Machine Learning Model, default is set to buy once and hold
    def get_act(self):
        for key in self.keys:
            self.act[key]=1   
        return self.act
           
##############################################################################
#This section is specific instances of the bot inheriting from BasicWrapper

#Keeps Track of Market, does not execute trades
class Market(BasicWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.counts=self.keys
        
    def get_act(self):
        for key in self.keys:
            if self.counts[key]==0:
                self.bought_price[key]=float(self.current_price[key])
                self.counts[key]=1
            self.act[key]=0
            self.Quantity[key]=QUANTITY/self.bought_price[key]
        return self.act
    
    def reward(self):
        reward=0
        for key in self.keys:
            if self.Owned[key]==0:
                reward+=(float(self.current_price[key])-self.bought_price[key])*self.Quantity[key]
        reward+=self.profit
        self.rewards=self.target
        return reward

class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, H_1,H_2, D_out):
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H_1)
        self.linear2 = torch.nn.Linear(H_1, H_2)
        self.linear3 = torch.nn.Linear(H_2, D_out)

    def forward(self, x):
        h_relu = self.linear1(x).clamp(min=0)
        h_relu = self.linear2(h_relu).clamp(min=0)
        y_pred = self.linear3(h_relu)
        y_pred=torch.sigmoid(y_pred)

        return y_pred

#Neural Net with Cross-Entropy Cost Function: Trained            
class CrossNN(BasicWrapper):
    def __init__(self, env):
        super().__init__(env)
        filename =CROSSNN_MODEL
        self.model= pickle.load(open(filename, 'rb'))
                
    def get_act(self):
        for key in self.klines:
            row=self.process_data(key)
            row=pd.DataFrame(row)
            row=torch.tensor(np.array(row), dtype=torch.float)
            predict=self.model.forward(row)
            predict=float(predict[0][1].detach().numpy())

            if predict>0.45:
                self.act[key]=1
            elif predict<.45:
                self.act[key]=0
        return self.act

#Sklearn MLP: Trained    
class MLP_Bot(BasicWrapper):
    def __init__(self, env):
        super().__init__(env)
        filename =MLP_MODEL
        self.model= pickle.load(open(filename, 'rb'))
        
        
    def get_act(self):
        for key in self.klines:
            row=self.process_data(key)
            if TRAIN==True:
                row=row.replace(np.nan,0)
                row=row.replace(np.inf,0)
            row=np.array(row)
            predict=self.model.predict(row)
            if predict>0.8:
                self.act[key]=1
            elif predict<.5:
                self.act[key]=0
        return self.act

#Sklearn Random Forest: Trained    
class Forest(BasicWrapper):
    def __init__(self, env):
        super().__init__(env)
        filename = FOREST_MODEL
        self.model= pickle.load(open(filename, 'rb'))
        
        
    def get_act(self):
        for key in self.klines:
            row=self.process_data(key)
            if TRAIN==True:
                row=row.replace(np.nan,0)
                row=row.replace(np.inf,0)
            row=np.array(row)
            predict=self.model.predict_proba(row)[0,1]
            if predict>0.5:
                self.act[key]=1
            elif predict<.45:
                self.act[key]=0
        return self.act


#General Pytorch Nueral Net Class    
class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1=nn.Linear(input_size, hidden_size)
        self.linear2=nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x=F.relu(self.linear1(x))
        x=self.linear2(x)
        return x
    
    def save(self, file_name="Big_NN.pth"):
        model_folder_path='./Big_NN'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name=os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

#Pytorch Neural Net: Trained    
class Big_NN(BasicWrapper):
    def __init__(self, env):
        super().__init__(env)
        filename = BIG_NN_MODEL
        model =Linear_QNet(56, 256, 2)
        model.load_state_dict(torch.load(filename))
        self.model=model
        
        
    def get_act(self):
        for key in self.klines:
            final_move = [0,0]
            state=self.process_data(key)
            state=state.replace(np.inf,0)
            state=state.replace(np.nan,0)
            state0 = torch.tensor(np.array(state), dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
            self.actions[key]=final_move
            if final_move==[1,0]:
                self.act[key]=1
            elif final_move==[0,1]:
                self.act[key]=0
        return

#Second Pytorch Neural Net: Trained
class Big_NN_2(BasicWrapper):
    def __init__(self, env):
        super().__init__(env)
        filename = BIG_NN_MODEL_2
        model =Linear_QNet(56, 256, 2)
        model.load_state_dict(torch.load(filename))
        self.model=model
        
        
    def get_act(self):
        for key in self.klines:
            final_move = [0,0]
            state=self.process_data(key)
            state=state.replace(np.inf,0)
            state=state.replace(np.nan,0)
            state0 = torch.tensor(np.array(state), dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
            self.actions[key]=final_move
            if final_move==[1,0]:
                self.act[key]=1
            elif final_move==[0,1]:
                self.act[key]=0
        return 

#############################################################################
#This section executes step functions for bots when turned on

if __name__=="__main__":
    
#Will execute step() function for all Models with ON=True
#i.e. FOREST_ON=True will loop using Random Forest Model
    
    hour=time.asctime().split(' ')[3]
    hour=hour.split(':')[0]
    hour=int(hour)
    temp_hour=hour
    plt.ion()
    fig, ax = plt.subplots()
    ax1 = fig.add_subplot(211)
    
    env_1 = Market(gym.make('MountainCar-v0'))
    env_2 = Forest(gym.make('MountainCar-v0'))
    env_3 = Big_NN(gym.make('MountainCar-v0'))
    env_4 = MLP_Bot(gym.make('MountainCar-v0'))
    env_5 = CrossNN(gym.make('MountainCar-v0'))
    env_6 = Big_NN_2(gym.make('MountainCar-v0'))
    num_steps = 1000

    obs = env_1.reset()
    
    rewards=[]
    rewards_2=[]
    rewards_3=[]
    rewards_4=[]
    rewards_5=[]
    rewards_6=[]

    for step in range(num_steps):
        #If LIVE_TRADING=True, will wait til top of hour to execute trades
        if LIVE_TRADING==True:
            while hour==temp_hour:
                print('Waiting For Next Hour...')
                temp_hour=time.asctime().split(' ')[3]
                temp_hour=hour.split(':')[0]
                temp_hour=int(hour)
                time.sleep(60)
                
        hour=temp_hour

        obs, reward, info = env_1.step()
        rewards.append(reward)
        y=range(len(rewards))
        plt.plot(y,rewards, label='Market',color= 'green')
        
        if FOREST_MODEL_ON==True:
            print('Forest Model Step')
            obs_2, reward_2, info_2=env_2.step()
            rewards_2.append(reward_2)
            plt.plot(y, rewards_2, label='Forest', color='black')
            
        if BIG_NN_MODEL_ON==True:
            print('Big NN Model Step')
            obs_3, reward_3, info_3=env_3.step()
            rewards_3.append(reward_3)
            plt.plot(y, rewards_3, label='Big Neural Net', color='orange')
            
        if MLP_MODEL_ON==True:
            print('MLP Model Step')
            obs_4, reward_4, info_4=env_4.step()
            rewards_4.append(reward_4)
            plt.plot(y, rewards_4, label='MLP_Bot', color='red')
            
        if CROSSNN_MODEL_ON==True:
            print('Cross NN Model Step')
            obs_5, reward_5, info_5=env_5.step()
            rewards_5.append(reward_5)
            plt.plot(y, rewards_5, label='CrossNN', color='purple')
            
        if BIG_NN_MODEL_2_ON==True:
            print('Big NN 2 Model Step')
            obs_6, reward_6, info_6=env_6.step()
            rewards_6.append(reward_6)
            plt.plot(y, rewards_6, label='Big Neural Net 2', color='blue')

        

        leg = plt.legend(loc='upper center')
        plt.title('Profit for Each Bot', loc='center')
        plt.draw()
        plt.pause(.1)
        plt.clf()

        


