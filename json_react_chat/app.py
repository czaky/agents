"LangChain react agent with custome executor using tools."

from typing import Sequence, Optional
from datetime import datetime, timedelta

from langchain_community.chat_models import ChatOllama
from langchain_community.tools import WikipediaQueryRun
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from langchain_community.tools.pubmed.tool import PubmedQueryRun
from langchain_community.tools.searx_search.tool import SearxSearchRun
from langchain_community.tools.semanticscholar.tool import SemanticScholarQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.utilities.searx_search import SearxSearchWrapper
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.tools import BaseTool, tool
from langchain_experimental.tools import PythonREPLTool
from langchain.agents import AgentExecutor, Tool
from langchain.agents.format_scratchpad import format_log_to_messages
from langchain.agents.json_chat.prompt import TEMPLATE_TOOL_RESPONSE
from langchain.agents.output_parsers import JSONAgentOutputParser
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools.render import render_text_description

import chainlit as cl
from chainlit.playground.config import add_llm_provider
from chainlit.playground.providers.langchain import LangchainGenericProvider

# from langchain.globals import set_verbose, set_debug
# set_verbose(True)
# set_debug(True)


MODEL = "dolphin-mixtral:8x7b-v2.7-q3_K_L"

SYSTEM_PROMPT = """
You are a question answering assistant using actions to respond to comments or question.
The actions provide observations which can prompt more thoughts and follow up actions.

ACTIONS
------
You have access to the following actions:

{tools}

RESPONSE FORMAT INSTRUCTIONS
----------------------------

When responding use one of the three formats:

**Format 1:**
Use this response format if you know the answer and want to respond directly. 
Markdown code snippet formatted in the following schema:

```json
{{
    "action": "Final Answer",
    "action_input": string // [Not null] You should put what you want to return here. 
```

**Format 2:**
Use this response format if you cannot find the answer. 
Markdown code snippet formatted in the following schema:

```json
{{
    "action": "Final Answer",
    "action_input": string // [Not null] Admit that you don't know the answer. 
}}
```

**Format 3:**
Use this response format if you need to use an action to find the answer.
Markdown code snippet formatted in the following schema:

```json
{{
    "thought": string, // [optional] Motivation for the action.
    "action": string, // [Not null] A new action to take. Must be one of: {tool_names}
    "action_input": string // [Not null] A valid and unique input to the action.
}}
```

-------------------------------------
(Use each action only once and only if necessary.)
(Use the full text of observations and thoughts for the final answer.)
(If you know the answer to the user comment, respond with final answer.)
(If no action is needed or available, respond with the the final answer.)
(Respond with a markdown code snippet of a json blob with a thought and a single action, and NOTHING else.)
"""

HUMAN_PROMPT = """
{input}

--------------------------------------
(Respond with a markdown code snippet of a json blob with a thought and a single action, and NOTHING else.)
"""

TEMPLATE_TOOL_RESPONSE = """
Assistant Observation: {observation}

--------------------------------------
(Use each action only once and only if necessary.)
(Use the full text of observations and thoughts for the final answer.)
(If you know the answer to the user comment, respond with final answer.)
(If no action is needed or available, respond with the the final answer.)
(Respond with a markdown code snippet of a json blob with a thought and a single action, and NOTHING else.)
"""


def create_json_chat_agent(
    llm: BaseLanguageModel,
    tools: Sequence[BaseTool],
    prompt: ChatPromptTemplate,
):
    prompt = prompt.partial(
        tools=render_text_description(list(tools)),
        tool_names=", ".join([t.name for t in tools]),
    )

    agent = (
        RunnablePassthrough.assign(
            agent_scratchpad=lambda x: format_log_to_messages(
                x["intermediate_steps"], template_tool_response=TEMPLATE_TOOL_RESPONSE
            )
        )
        | prompt
        | llm
        | JSONAgentOutputParser()
    )
    return agent


def int_or_none(s: str) -> Optional[int]:
    try:
        return int(s)
    except ValueError:
        return None


@tool("current_date")
def current_date(day: str = "now") -> str:
    "Use this to get the current date or current year. [use 'now' for input]."
    date = datetime.now()
    if day.lower() == "yesterday":
        date -= timedelta(days=1)
        return "Yesterday was: " + date.strftime("%A %Y-%m-%d")
    if day.lower() == "tomorrow":
        date += timedelta(days=1)
        return "Tomorrow is: " + date.strftime("%A %Y-%m-%d")
    if delta := int_or_none(day):
        date += timedelta(days=delta)
        return date.strftime("%Y-%m-%d")
    return "Current date is: " + date.strftime("%A %Y-%m-%d")


@cl.on_chat_start
async def on_chat_start():
    print("----------------------------------------------------------")
    llm = ChatOllama(
        model=MODEL,
        temperature=1,
        verbose=True,
        format="json",
        base_url="http://ollama:11434",
    )

    # Add the LLM provider to debug prompts.
    add_llm_provider(
        LangchainGenericProvider(
            id=llm._llm_type,
            name="dolphin-mixtral",
            llm=llm,
            is_chat=True,
        )
    )

    tools = [
        current_date,
        WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper()),
        # ArxivQueryRun(),
        # SemanticScholarQueryRun(),
        # PubmedQueryRun(),
        SearxSearchRun(
            wrapper=SearxSearchWrapper(searx_host="http://searxng:8080", unsecure=True)
        ),
        SearxSearchRun(
            name="searx_news",
            wrapper=SearxSearchWrapper(
                searx_host="http://searxng:8080", categories=["news"], unsecure=True
            ),
            description="Use this to search for latest news.",
        ),
        # PythonREPLTool(
        #    description="Use this if you need to execute valid python code. Code needs to end with `print($RESULT)`.",
        #    # return_direct=True,
        # ),
    ]

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", HUMAN_PROMPT),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )

    memory = ConversationBufferWindowMemory(
        k=5, memory_key="chat_history", output_key="output", return_messages=True
    )

    executor = AgentExecutor(
        agent=create_json_chat_agent(llm, tools, prompt),
        tools=tools,
        verbose=True,
        max_iterations=7,
        memory=memory,
        handle_parsing_errors=True,
        return_intermediate_steps=True,
    )
    # print(executor)
    cl.user_session.set("executor", executor)


@cl.on_message
async def on_message(message: cl.Message):
    inputs = {"input": message.content}
    executor = cl.user_session.get("executor")
    assert executor
    config = {"callbacks": [cl.LangchainCallbackHandler()]}
    response = executor.invoke(inputs, config=config)
    await cl.Message(content=response["output"]).send()
