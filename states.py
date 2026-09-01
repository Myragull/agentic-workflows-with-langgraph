# so now we are creating a graph in langgraph that has states nodes and edges
# and the very first thing you creates is state
# and before doing anything crete a pipeline of the flow from to where the data 
# go for easier so that you can easily create the graph and the flow of the data

import os

# here are few methods witht he help of which we can create states here han 
# 1 : for creating the state we import typed dictionary (Most common approach)
from typing import TypedDict
# it us used a dictionary 

class State(TypedDict):
    topic : str
    summary : str
    score : str

# Create a class called State, using TypedDict as its base
# I want State to behave according to the rules of a TypedDict    


# 2 Pydantic Approach alos used in fast APi
# it is good at data validation and type checking at run time
# alos uses for not making any kind of mistakes 

from pydantic import BaseModel, field_validator

class State(BaseModel):
    topic : str
    summary : str
    score : str

    @field_validator
    def score_positive(cls,v):
        if v < 0:
            raise ValueError("Score must be a positive number")


# python dataclasses

# Standard python dataclass but it is used rarelty

from dataclasses import dataclass , field

@dataclass
class State:
    topic : str
    summary : str
    score : str = field(default=0)


# from langgrapg.graph import MessagesState    

# class State(MessagesState):
#     #message field is already included with add_messages
#     # just add your extra fields
#     user_name :str
#     language : str