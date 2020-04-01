#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 11:54:44 2020

@author: Abderrahim
"""
import requests



def getChannels(token):
    ''' 
    function returns an object containing a object containing all the
    channels in a given workspace
    ''' 
    channelsURL = "https://slack.com/api/conversations.list?token=%s" % token
    channelList = requests.get(channelsURL).json()["channels"] # an array of channels
    channels = {}
    # putting the channels and their ids into a dictonary
    for channel in channelList:
        channels[channel["name"]] = channel["id"]
    return {"channels": channels}

def getUsers(token):
    # this function get a list of users in workplace including bots 
    channelsURL = "https://slack.com/api/users.list?token=%s&pretty=1" % token
    members = requests.get(channelsURL).json()["members"]
    return members

def send_message(message,slack_token="",channelId=""):
    data = {
        'token': slack_token,
        'channel': channelId,    # User ID. 
        'as_user': True,
        'text': message
    }
    
    requests.post(url='https://slack.com/api/chat.postMessage',
                  data=data)
    
def getMessages(token="", channelId=""):
    # print("Getting Messages")
    # this function get all the messages from the slack team-search channel
    # it will only get all the messages from the team-search channel
    slack_url = "https://slack.com/api/conversations.history?token=" + token + "&channel=" + channelId
    messages = requests.get(slack_url).json()
    messages = messages["messages"][0]["text"] if messages["messages"][0]["user"]!="UUV47PX5K" else ""
    return messages