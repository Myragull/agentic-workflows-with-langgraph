import os
from typing import TypedDict,Annotated, Literal
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph,START,END
from langgraph.prebuilt import ToolNode
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from dotenv import load_dotenv

load_dotenv()

#tools
# Firstly creates tools for the agent to use them

search_tools = TavilySearch(max_results = 3)

tools = [search_tools]

#llms
 
writer_llm = ChatGroq(model="openai/gpt-oss-120b",temperature=0.4) 
# bcoz this llm will also in needs of the tools for web search for that we can 
# also bind th tools with it
writer_llm_with_tools = writer_llm.bind_tools(tools)

# reviewer

reviewer_llm = ChatGroq(model="openai/gpt-oss-120b",temperature=0.1)


#state building

class State(TypedDict):
    topic : str
    # the single topic received by the writer llm
    messages : Annotated[list,add_messages]
    # for maintianing the conversation history acorss llms with api keys
    draft : str
    # not an original post a draft to be revied again and again
    review_feedback : str
    # the feedback received by the witer llm again and again
    is_approved :bool
    # whether the draft is approved or not by the reviewer llm
    attempts : int
    # how many attempts