import os
from typing import TypedDict

class pipelinestate(TypedDict):
    raw_input : str
    edited_text : str
    script_text : str
    final_output : str

from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.7
)
    
def editor_node(state: pipelinestate) ->dict:
    """" Staeg1: Cleans up grammer ,removes typos,and refines the
    tone ."""

    prompt=(
        "You are an expert copyeditor.Cleans up the following raw text."
        "Fix any grammatical errors,spellling mistakes,and smooth out the transition flow ."
        "While keeping the core message intact. Return only the edited text \n\n"
        f"Text:\n{state['raw_input']}"
    )
    response = llm.invoke(prompt)

    return {"edited_text": response.content.strip()}


def scriptwriter_node(state: pipelinestate) -> dict:
    """Stage 2: Formats the clean text into an engaging video script style."""
    print("\n--- [stage 2] Executing Scriptwriter Node ---")

    prompt = (
        "you are a charismatic Youtube content  creator . Take this edited text and transform "
        "it into a highly engaging ,punchy,conversational video script hook. Make it sounds "
        "Like a real person speaking passionaely. Return only the script content . \n\n"
        f"Edited Text:\n{state['edited_text']}"
    )

    response = llm.invoke(prompt)
    return {"script_text": response.content.strip()}


def translator_node(state: pipelinestate) -> dict:
     """Stage 3 : Translates the script into the natural flowing Hinglish"""
     print("\n--- [stage 3] Executing Translator Node ---")

     prompt = (
         "You are an expert content localizer for the Indian market. Take the following script ."
         "and convert it into natural,flowing 'Hinglish'.Do not simply translate it sentence"
         "or repeat information. Alternating comfortably between Hindi and english phrases just"
         "Return only the final Hinglish text.\n\n"
         f"Script :\n{state['script_text']}"
     )

     response = llm.invoke(prompt)
     return {"final_output": response.content.strip()} 



# now your stataes and noides are ready and now it is time to create the graph
# and for creaing the grpah you ahev to connect these nodes 
# and for that you ahev to use the edges 
# edges are very importnat to create the workflow 
# there are different kinds of wrokflow sequential, conditional and loops ones

# firstly we will craete the graph
from langgraph.graph import StateGraph,START,END

# create the graph
graph= StateGraph(pipelinestate)

# ad the nodes in our graoh 
graph.add_node("editor", editor_node)
graph.add_node("scriptwriter", scriptwriter_node)
graph.add_node("translator", translator_node)

# now we will finally adds the edges(sequential-one after another)
graph.add_edge(START,"editor")
graph.add_edge("editor","scriptwriter")
graph.add_edge("scriptwriter","translator")
graph.add_edge("translator",END)

#compile the graph

app=graph.compile()

result = app.invoke({
    "raw_input": "Ai Agents are the future of tech . they can think ,plan and act on their own "
    "Langraph helps to build these agents with proper control and memory."

})

#output
print("your result are :")
print(result["final_output"])