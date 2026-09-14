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


WRITER_SYSTEM_PROMPT = (
    "You are an expert LinkedIn content writer. Your job is to write "
    "engaging, professional LinkedIn posts about the given topic. "
    "If the topic requires up-to-date information, statistics, or "
    "current trends, use the web search tool to gather fresh context "
    "before writing. If you have already received feedback on a "
    "previous draft, carefully address every point in the new draft. "
    "Rules for good LinkedIn posts: strong hook in the first line, "
    "1 clear takeaway, easy to skim (short paragraphs), around "
    "150–200 words, ends with a question or call-to-action to invite "
    "engagement. Do not use hashtags."
)

# Writer Node
def writer_node(state : State) -> dict:
    """Writes (or rewrites) the LinkedIn post. Can call Tavily to search first."""
    attempt = state.get("attempt", 0) + 1
    topic = state["topic"]
    previous_feedback = state['review_feedback']
if attempt == 1:
    user_message = (
        f"Write a LinkedIn post on this topic {topic}"
        f"if you need current info search the web first"
    )
else:
    user_message = (
        f"your previous draft on '{topic}' was rejected"
        f"Here is the reviewer's feedback \n\n {previous_feedback}\n\n"
        f"Write a new, improved draft that fixes every issue mentiond"
        f"do not repeat the same mistake"
    )
# conversation going to the llm with the previous feedback and the new topic to write a new draft
messages = [("system", WRITER_SYSTEM_PROMPT),("human", user_message)]
# calls the llm
response = writer_llm_with_tools.invoke(messages)

return {
    "messages" : [("human",user_message),response],
    "attempt": attempt,
}


# making the tool node to be used for calling all the other tools
tool_node = ToolNode(tools)

# extraction node 
# to extract the conten or the draft or the mesage from the writer node

def extract_draft_node(state:State) -> dict:
    """After the writer finishes tool calls, pulls the final text out as the draft."""
    last_message = state['messages'][-1]
# making a drat of the message or the AI response here
    draft = last_message.content
    print(f"\n\n Generated post \n{draft}|n")
    return {"draft":draft}


