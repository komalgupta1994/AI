from typing import List, Union
from dotenv import load_dotenv
from langchain_core.tools import tool, render_text_description, Tool
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.schema import AgentAction, AgentFinish
from callbacks import AgentCallbackHandler

load_dotenv()


# After using the tool decorator, the simple function "get_text_length" has been converted into langchain structured tool.
# Where it has now some other properties like name, description etc. Which helps llm to understand the tool better.
@tool
def get_text_length(text: str) -> int:
    """Returns the length of the text by characters."""
    text = text.strip("'\n").strip('"')
    return len(text)

def find_tool_by_name(tools: List[Tool], name: str) -> Tool:
    for tool in tools:
        if tool.name == name:
            return tool
    raise ValueError(f"Tool with name {name} not found")


def main():
    print("Hello from react-langchain!")


if __name__ == "__main__":
    main()
    # print(get_text_length(text="Hello, world!!")) # After using the tool decorator, we can't call the function normally.
    # print(get_text_length.invoke(input={"text": "Hello World!"})) # We need to use the invoke method to call the function.
    tools = [get_text_length]

    template = """
    Answer the following questions as best you can. You have access to the following tools:

    {tools}
    
    Use the following format:
    
    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question
    
    Begin!
    
    Question: {input}
    Thought: {agent_scratchpad}
    """

    prompt = PromptTemplate.from_template(template=template).partial(
        tools=render_text_description(tools), tool_names=", ".join([tool.name for tool in tools])
    )

    llm = ChatOpenAI(temperature=0, stop=["\nObservation"], callbacks=[AgentCallbackHandler()])
    intermediate_steps = []

    agent = {"input": lambda x:x["input"], "agent_scratchpad": lambda x: (x["agent_scratchpad"])} | prompt | llm | ReActSingleInputOutputParser()
    
    agent_step = ""
    while not isinstance(agent_step, AgentFinish):
        agent_step: Union[AgentAction, AgentFinish] = agent.invoke({"input": "What is the length in characters of the text 'Hello, how are you?'?", "agent_scratchpad": (intermediate_steps)})

        if isinstance(agent_step, AgentAction):
            tool_name = agent_step.tool
            tool_to_use = find_tool_by_name(tools, tool_name)
            tool_input = agent_step.tool_input

            observation = tool_to_use.func(str(tool_input))
            intermediate_steps.append((agent_step, str(observation)))

    if (isinstance(agent_step, AgentFinish)):
        print(f"Final Answer: {agent_step.return_values}")
