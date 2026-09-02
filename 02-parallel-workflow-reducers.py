import os
from typing import TypedDict, Annotated

from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph

load_dotenv()


# LLM
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.1
)


# Reducer
def merge_score_dicts(existing: dict, newupdate: dict) -> dict:
    if existing is None:
        return newupdate

    return {**existing, **newupdate}


# State
class AnalyzerState(TypedDict):
    raw_text: str
    safety_score: Annotated[dict[str, int], merge_score_dicts]


# NODES

def toxicity_node(state: AnalyzerState) -> dict:
    print("\n[Branch 1] Analyzing Toxicity and Hate Speech...")

    prompt = (
        "Analyze the following text for profanity, aggression, hate speech, "
        "or toxicity. "
        "Provide a score from 0 to 100, where 0 means perfectly clean "
        "and 100 means highly toxic. "
        "Return only the plain integer number. Nothing else.\n\n"
        f"Text:\n{state['raw_text']}"
    )

    response = llm.invoke(prompt)

    try:
        score = int(response.content.strip())
    except ValueError:
        score = 0

    # Return a sub-dictionary under our shared state key
    return {
        "safety_score": {
            "toxicity_level": score
        }
    }


def copyright_node(state: AnalyzerState) -> dict:
    print("\n[Branch 2] Analyzing Copyright Infringement...")

    prompt = (
        "Analyze the following text for potential copyright infringement. "
        "Provide a score from 0 to 100, where 0 means no infringement "
        "and 100 means clear infringement. "
        "Return only the plain integer number. Nothing else.\n\n"
        f"Text:\n{state['raw_text']}"
    )

    response = llm.invoke(prompt)

    try:
        score = int(response.content.strip())
    except ValueError:
        score = 0

    return {
        "safety_score": {
            "copyright_risk": score
        }
    }


def culture_node(state: AnalyzerState) -> dict:
    print("\n[Branch 3] Analyzing Cultural Sensitivity...")

    prompt = (
        "Analyze the following text for potential cultural insensitivity. "
        "Provide a score from 0 to 100, where 0 means no issues "
        "and 100 means highly insensitive. "
        "Return only the plain integer number. Nothing else.\n\n"
        f"Text:\n{state['raw_text']}"
    )

    response = llm.invoke(prompt)

    try:
        score = int(response.content.strip())
    except ValueError:
        score = 0

    return {
        "safety_score": {
            "cultural_insensitivity": score
        }
    }


# BUILD GRAPH

builder = StateGraph(AnalyzerState)

# Add nodes
builder.add_node("toxicity", toxicity_node)
builder.add_node("copyright", copyright_node)
builder.add_node("culture", culture_node)


# FAN-OUT
builder.add_edge(START, "toxicity")
builder.add_edge(START, "copyright")
builder.add_edge(START, "culture")


# FAN-IN / END
builder.add_edge("toxicity", END)
builder.add_edge("copyright", END)
builder.add_edge("culture", END)


# Compile
app = builder.compile()
 
# RUN GRAPH

sample_state = {
    "raw_text": (
        "This is a sample text that may contain some "
        "offensive language or copyrighted material."
    ),
    "safety_score": {}
}

final_state = app.invoke(sample_state)

print("\nFinal Scores:")
print(final_state["safety_score"])