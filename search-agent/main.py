from dotenv import load_dotenv
load_dotenv()

from langchain import hub
from langchain.agents import AgentExecutor
from langchain.agents.react.agent import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch


tools = [TavilySearch()]

# react_prompt
# Answer the following questions as best you can. You have access to the following tools:

# {tools}

# Use the following format:

# Question: the input question you must answer
# Thought: you should always think about what to do
# Action: the action to take, should be one of [{tool_names}]
# Action Input: the input to the action
# Observation: the result of the action
# ... (this Thought/Action/Action Input/Observation can repeat N times)
# Thought: I now know the final answer
# Final Answer: the final answer to the original input question

# Begin!

# Question: {input}
# Thought:{agent_scratchpad}

# llm = ChatOpenAI(model="gpt-4")
llm = ChatOllama(temperature=0, model="gemma3:270m")
react_prompt = hub.pull("hwchase17/react")
agent = create_react_agent(llm = llm, tools=tools, prompt=react_prompt)
agent_executor = AgentExecutor(agent=agent, tools= tools, verbose=True)
chain = agent_executor


def main():
    result = chain.invoke(
        input={
            "input": "Search for a list of the best restaurants in Gurgaon City."
            }
        )
    print(result)



if __name__ == "__main__":
    main()
