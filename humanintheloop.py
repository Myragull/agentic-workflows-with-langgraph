import os
from typing import TypedDict, Annotated, Literal

from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, interrupt
from langgraph.checkpoint.memory import MemorySaver

from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from dotenv import load_dotenv

load_dotenv()

# tools
# Firstly creates tools for the agent to use them

search_tools = TavilySearch(max_results=3)

tools = [search_tools]

# llms

writer_llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.4
)

# bcoz this llm will also in needs of the tools for web search for that we can
# also bind the tools with it
writer_llm_with_tools = writer_llm.bind_tools(tools)

# reviewer

reviewer_llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.1
)


# state building

class State(TypedDict):
    topic: str

    # the single topic received by the writer llm
    messages: Annotated[list, add_messages]

    # for maintaining the conversation history across llms with api keys
    draft: str

    # not an original post a draft to be reviewed again and again
    review_feedback: str

    # the feedback received by the writer llm again and again
    is_approved: bool

    # whether the draft is approved or not by the human
    attempt: int

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
def writer_node(state: State) -> dict:
    """Writes (or rewrites) the LinkedIn post. Can call Tavily to search first."""

    attempt = state.get("attempt", 0) + 1
    topic = state["topic"]
    previous_feedback = state["review_feedback"]

    if attempt == 1:
        user_message = (
            f"Write a LinkedIn post on this topic {topic}"
            f"if you need current info search the web first"
        )
    else:
        user_message = (
            f"your previous draft on '{topic}' was rejected"
            f"Here is the reviewer's feedback \n\n {previous_feedback}\n\n"
            f"Write a new, improved draft that fixes every issue mentioned"
            f"do not repeat the same mistake"
        )

    # conversation going to the llm with the previous feedback
    # and the new topic to write a new draft
    messages = [
        ("system", WRITER_SYSTEM_PROMPT),
        ("human", user_message)
    ]

    # calls the llm
    response = writer_llm_with_tools.invoke(messages)

    return {
        "messages": [("human", user_message), response],
        "attempt": attempt,
    }


# making the tool node to be used for calling all the other tools
tool_node = ToolNode(tools)


# extraction node
# to extract the content or the draft or the message from the writer node

def extract_draft_node(state: State) -> dict:
    """After the writer finishes tool calls, pulls the final text out as the draft."""

    last_message = state["messages"][-1]

    # making a draft of the message or the AI response here
    draft = last_message.content

    print(f"\n\nGenerated post\n{draft}\n")

    return {
        "draft": draft
    }


# Human review node

def human_review_node(state: State) -> dict:
    """Pauses the graph and waits for the human to approve or give feedback."""

    human_response = interrupt({
        "draft": state["draft"],
        "attempt": state["attempt"],
        "instruction": "Type 'approved' to accept, or type your feedback to request a revision."
    })

    response = human_response.strip()

    if response.lower() in ["approved", "approve", "yes", "ok", "good"]:
        return {
            "is_approved": True,
            "review_feedback": "Approved by human."
        }
    else:
        return {
            "is_approved": False,
            "review_feedback": response
        }


# Router function
# For creating the logic around nodes

def should_use_tool(state: State):
    last_message = state["messages"][-1]

    if getattr(last_message, "tool_calls", None):
        return "tools"

    return "extract_draft"


def should_stop_looping(state: State):
    if state["is_approved"]:
        print("post has been approved \n")
        return END

    if state["attempt"] >= 3:
        print("reached max attempts")
        return END

    return "writer"


# build the graph

graph = StateGraph(State)

graph.add_node("writer", writer_node)
graph.add_node("tools", tool_node)
graph.add_node("extract_draft", extract_draft_node)
graph.add_node("human_reviewer", human_review_node)

graph.add_edge(START, "writer")

graph.add_conditional_edges(
    "writer",
    should_use_tool
)

# If Tavily was used, go back to the writer
# so the writer can use the search results
graph.add_edge("tools", "writer")

graph.add_edge("extract_draft", "human_reviewer")

graph.add_conditional_edges(
    "human_reviewer",
    should_stop_looping
)


# Checkpointer for maintaining state using thread_id

checkpointer = MemorySaver()

app = graph.compile(
    checkpointer=checkpointer
)


print("=" * 55)
print("Welcome to the LinkedIn Post Generator")
print("=" * 55)
print("\nThis tool will draft a LinkedIn post for you, review it")
print("itself, and iterate until it's publish-ready.")

print("=" * 55)

topic = input(
    "\nWhat topic do you want a LinkedIn post about?\n> "
).strip()


if not topic:
    print("\nNo topic given. Exiting.")

else:
    print("\nStarting generation...\n")

    config = {
        "configurable": {
            "thread_id": "linkedin_session_1"
        }
    }

    initial_state = {
        "topic": topic,
        "messages": [],
        "draft": "",
        "review_feedback": "",
        "is_approved": False,
        "attempt": 0,
    }

    result = app.invoke(
        initial_state,
        config=config
    )

    # Human-in-the-loop
    while "__interrupt__" in result:

        interrupt_data = result["__interrupt__"][0].value

        print("\n" + "=" * 55)
        print("HUMAN REVIEW")
        print("=" * 55)

        print(f"\nAttempt: {interrupt_data['attempt']}")
        print("\nDraft:")
        print(interrupt_data["draft"])

        print("\n" + interrupt_data["instruction"])

        human_input = input("\nYour response: ").strip()

        result = app.invoke(
            Command(resume=human_input),
            config=config
        )

    print("\n" + "=" * 55)
    print("FINAL LINKEDIN POST")
    print("=" * 55)

    print(result["draft"])

    print("=" * 55)
    print(f"Total attempts: {result['attempt']}")
    print(f"Approved: {result['is_approved']}")