# -*- coding: utf-8 -*-
"""
Created on Sat Dec 11 01:44:47 2021

@author: wkjon
"""

import os

import time
from time import sleep

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
If self.validation==True
    --->This will run the model using backtesting data, last three months of hourly data.        


'''


##############################################################################
#THIS SECTION CREATES CONNECTION WITH BINANCE API

#Test Key using Paper Trading API
#REPLACE THESE KEYS WITH YOUR OWN KEYS FROM BINANCE.US
Test_Key='VJvm6j63VzZjlKxmBGgiGMh12mZZk9pJzg0b8bPcHtdO9d7FlL0raLIpFtI4oZJs'
Test_Secret_Key='D4rnRxgRfc9oajWAmHHov9kNjHuKdI7YlvLNQaWNUbg3XFrp8MPAYlM6P6bnru8O'

client = Client(Test_Key, Test_Secret_Key)

client.API_URL = 'https://testnet.binance.vision/api'

print(client.get_account())

print(client.get_open_orders)

#List of Coins from Training Data
ticker_list=['BTCUSDT', 'BNBUSDT', 'ETHUSDT', 'LTCUSDT', 'XRPUSDT']


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


##############################################################################
#THIS SECTION GETS LATEST CANDLESTICK DATA FROM BINANCE API
def get_price():

    # get latest price from Binance API
    ticker_list=['BTCUSDT', 'BNBUSDT', 'ETHUSDT', 'LTCUSDT', 'XRPUSDT']
    price_dict={}
    for ticker in ticker_list:
        ticker_price = client.get_symbol_ticker(symbol=ticker)
        price_dict.update({ticker: ticker_price})
    # print full output (dictionary)
    price=float(ticker_price['price'])
    sleep(59)
    return price_dict


def get_kline():
    # get 1 hour klines last 24 hours from Binance API
    ticker_list=['BTCUSDT', 'BNBUSDT', 'ETHUSDT', 'LTCUSDT', 'XRPUSDT']
    price_dict={}
    for ticker in ticker_list:
        #Set to get one hour data for last 24 hours
        #klines = client.get_historical_klines(ticker, Client.KLINE_INTERVAL_1HOUR, "30 HOUR ago", "Now UTC+5")
        klines=client.get_klines(symbol=ticker,interval=Client.KLINE_INTERVAL_1HOUR)
        klines = pd.DataFrame(klines, columns=['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 'Quote asset volume',
                     'Number of trades', 'Taker buy base asset volume', 'Taker buy quote asset volume', 'Junk'])
        print(ticker)
        print(klines)
        

        price_dict.update({ticker: klines})
    return price_dict

get_kline()

#Pulls Validation Data for backtesting 
def get_kline_val(val_set, count):
    ticker_list=['BTC', 'BNB', 'ETH', 'LTC', 'XRP']
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


##############################################################################

class BasicWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        #keys={'BTCUSDT':0, 'BNBUSDT':0, 'ETHUSDT':0, 'LTCUSDT':0, 'XRPUSDT':0}
        self.keys={'BTCUSDT':0, 'ETHUSDT':0, 'LTCUSDT':0}
        keys={'BTCUSDT':0, 'ETHUSDT':0, 'LTCUSDT':0}
        self.Owned=keys.copy()
        self.bought_price={'BTCUSDT':1, 'BNBUSDT':1, 'ETHUSDT':1, 'LTCUSDT':1, 'XRPUSDT':1}
        self.sell_price=keys.copy()
        self.current_price=keys.copy()
        self.Quantity={'BTCUSDT': 100/(self.bought_price['BTCUSDT']), 'BNBUSDT': 100/(self.bought_price['BNBUSDT']), 'ETHUSDT': 100/(self.bought_price['ETHUSDT']), 'LTCUSDT': 100/(self.bought_price['LTCUSDT']), 'XRPUSDT': 100/(self.bought_price['XRPUSDT'])}
        self.profit=0
        self.act=keys.copy()
        self.klines=None
        self.val_set_hour=pd.read_csv(r'C:\Users\wkjon\.spyder-py3\crypto_AI_hour_validation.csv')
        self.validation=False
        self.count=24

    def process_data(self, key):
        df=self.klines[key]
        print(df)
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
        row_df['24 Average Volume']=df['Percent Volume'].mean()
        row_df['12 Average Volume']=df.loc[12:, 'Percent Volume'].mean()
        row_df['6 Average Volume'] = df.loc[18:, 'Percent Volume'].mean()
        row_df['3 Average Volume'] = df.loc[21:, 'Percent Volume'].mean()
        row_df['24 Average Percent Changes']=df['Percent Changes'].mean()
        row_df['12 Average Percent Changes']=df.loc[12:, 'Percent Changes'].mean()
        row_df['6 Average Percent Changes'] = df.loc[18:, 'Percent Changes'].mean()
        row_df['3 Average Percent Changes'] = df.loc[21:, 'Percent Changes'].mean()
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

    def observation(self):
        if self.validation==True:
            self.klines=get_kline_val(self.val_set_hour,self.count)
            self.count+=1
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

    def reward(self):
        reward=0
        for key in self.keys:
            if self.Owned[key]==1:
                reward+=(self.current_price[key]-self.bought_price[key])*self.Quantity[key]
        reward+=self.profit
        return reward

    def action(self):
        act=self.get_act()
        for key in self.keys:
            if self.act[key]==1 and self.Owned[key]==0:
                if self.validation==True:
                    self.bought_price[key]=self.current_price[key].copy()
                    self.Quantity[key]=round(100/self.bought_price[key],3)
                    
                else:
                    self.bought_price[key]=self.current_price[key]
                    self.Quantity[key]=round(100/self.bought_price[key],3)
                    print(key) 
                    print(self.Quantity[key])
                    buy_order = client.create_order(symbol=key, side='SELL', type='MARKET', quantity=self.Quantity[key])
                    if len(buy_order['fills'])==0:
                        print('NOT FILLED BUY '+str(key))
                        continue
                    self.bought_price[key]=float(buy_order['fills'][0]['price'])
                self.Owned[key]=1
            elif self.act[key]==0 and self.Owned[key]==1:
                if self.validation==True:
                    self.sell_price[key]=self.current_price[key]
                else:
                    sell_order = client.create_order(symbol=key, side='SELL', type='MARKET', quantity=self.Quantity[key])
                    if len(sell_order['fills'])==0:
                        print('NOT FILLED SELL '+str(key))
                        continue
                    self.sell_price[key]=float(sell_order['fills'][0]['price'])
                profit=(self.sell_price[key]-self.bought_price[key])*self.Quantity[key]
                self.profit+=profit  
                self.Owned[key]=0
        return act

    def get_act(self):
        for key in self.keys:
            self.act[key]=1   
        return self.act
    

    
class XGB_Bot(BasicWrapper):
    def __init__(self, env):
        super().__init__(env)
        filename = r'C:\Users\wkjon\.spyder-py3\finalized_model.sav'
        self.model= pickle.load(open(filename, 'rb'))
        
        
    def get_act(self):
        #self.observation(0)
        for key in self.klines:
            row=self.process_data(key)
            print(row)
            predict=self.model.predict_proba(row)[0,1]
            if predict>0.475:
                self.act[key]=1
            elif predict<.4:
                self.act[key]=0
        return self.act
    
   

class LRWrapper(BasicWrapper):
    def __init__(self, env):
        super().__init__(env)
        filename = r'C:\Users\wkjon\.spyder-py3\finalized_model_minute.sav'
        self.model= pickle.load(open(filename, 'rb'))

    def get_act(self):
        if len(self.percent_changes)<10:
            return 2
        else:
            data=self.percent_changes[-9:]
            data=np.array(data)
            data=data.reshape(1,9)
            predict=self.model.predict(data)
            if predict>0.000:
                return 1
            else:
                return 0
            
##############################################################################

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

            
class CrossNN(BasicWrapper):
    def __init__(self, env):
        super().__init__(env)
        filename = r'C:\Users\wkjon\.spyder-py3\CrossNN.sav'
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
            elif predict<.4:
                self.act[key]=0
        return self.act
    
class MLP_Bot(BasicWrapper):
    def __init__(self, env):
        super().__init__(env)
        filename = r'C:\Users\wkjon\.spyder-py3\finalized_model_mlp.sav'
        self.model= pickle.load(open(filename, 'rb'))
        
        
    def get_act(self):
        #self.observation(0)
        for key in self.klines:
            row=self.process_data(key)
            predict=self.model.predict(row)
            if predict>0.8:
                self.act[key]=1
            elif predict<.5:
                self.act[key]=0
        return self.act
    
class Combo(BasicWrapper):
    def __init__(self, env):
        super().__init__(env)
        filename = r'C:\Users\wkjon\.spyder-py3\finalized_model_mlp.sav'
        self.model_1= pickle.load(open(filename, 'rb'))
        filename = r'C:\Users\wkjon\.spyder-py3\finalized_model.sav'
        self.model_2= pickle.load(open(filename, 'rb'))
        filename = r'C:\Users\wkjon\.spyder-py3\CrossNN.sav'
        self.model= pickle.load(open(filename, 'rb'))
        
        
    def get_act(self):
        #self.observation(0)
        for key in self.klines:
            count=0
            row=self.process_data(key)
            predict=self.model_1.predict(row)
            if predict>0.8:
                count+=1
            else:
                self.act[key]=0
                
            predict=self.model_2.predict(row)
            if predict>0.8:
                count+=0
            else:
                self.act[key]=0
                
            row=pd.DataFrame(row)
            row=torch.tensor(np.array(row), dtype=torch.float)
            predict=self.model.forward(row)
            predict=float(predict[0][1].detach().numpy())

            if predict>0.465:
                count+=1
            else:
                self.act[key]=0
            
            if count==2:
                self.act[key]=1
            
            
        return self.act

    
class Forest(BasicWrapper):
    def __init__(self, env):
        super().__init__(env)
        filename = r'C:\Users\wkjon\.spyder-py3\finalized_model_rfc.sav'
        self.model= pickle.load(open(filename, 'rb'))
        
        
    def get_act(self):
        #self.observation(0)
        for key in self.klines:
            row=self.process_data(key)
            for i in row.columns:
                print(i)
                print(row.loc[:,i])
            predict=self.model.predict_proba(row)[0,1]
            if predict>0.5:
                self.act[key]=1
            elif predict<.45:
                self.act[key]=0
        return self.act
    
    

if __name__=="__main__":
    plt.ion()
    fig, ax = plt.subplots()
    ax1 = fig.add_subplot(211)
    
    env_1 = BasicWrapper(gym.make('MountainCar-v0'))
    env_2 = Forest(gym.make('MountainCar-v0'))
    env_3 = XGB_Bot(gym.make('MountainCar-v0'))
    env_4 = MLP_Bot(gym.make('MountainCar-v0'))
    env_5 = CrossNN(gym.make('MountainCar-v0'))
    env_6 = Combo(gym.make('MountainCar-v0'))
    num_steps = 1800

    obs = env_1.reset()
    
    rewards=[]
    rewards_2=[]
    rewards_3=[]
    rewards_4=[]
    rewards_5=[]
    rewards_6=[]

    for step in range(num_steps):


    # apply the action
        obs, reward, info = env_1.step()
        obs_2, reward_2, info_2=env_2.step()
        obs_3, reward_3, info_3=env_3.step()
        obs_4, reward_4, info_4=env_4.step()
        obs_5, reward_5, info_5=env_5.step()
        obs_6, reward_6, info_6=env_6.step()
        
        rewards.append(reward)
        rewards_2.append(reward_2)
        rewards_3.append(reward_3)
        rewards_4.append(reward_4)
        rewards_5.append(reward_5)
        rewards_6.append(reward_6)

        
        y=range(len(rewards))
        plt.plot(y,rewards, label='Market',color= 'green')
        plt.plot(y, rewards_2, label='Forest', color='black')
        plt.plot(y, rewards_3, label='XGB_Bot', color='orange')
        plt.plot(y, rewards_4, label='MLP_Bot', color='red')
        plt.plot(y, rewards_5, label='CrossNN', color='purple')
        plt.plot(y, rewards_6, label='Combo', color='blue')
        leg = plt.legend(loc='upper center')
        plt.title('Profit for Each Bot', loc='center')
        plt.draw()
        plt.pause(.1)
        plt.clf()
        
        if env_1.validation==False:
            sleep(10)



