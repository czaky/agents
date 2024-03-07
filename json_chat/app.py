"LangChain app using tools."

from langchain_community.chat_models import ChatOllama
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools import WikipediaQueryRun
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from langchain_community.tools.pubmed.tool import PubmedQueryRun
from langchain_community.tools.searx_search.tool import SearxSearchRun
from langchain_community.tools.semanticscholar.tool import SemanticScholarQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.utilities.searx_search import SearxSearchWrapper
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_core.tools import BaseTool
from langchain_experimental.tools import PythonREPLTool
from langchain.agents import AgentExecutor
from langchain.agents.agent import BaseSingleActionAgent, BaseMultiActionAgent
from langchain.agents.format_scratchpad import format_log_to_messages
from langchain.agents.json_chat.prompt import TEMPLATE_TOOL_RESPONSE
from langchain.agents.output_parsers import JSONAgentOutputParser
from langchain.globals import set_verbose, set_debug
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools.render import render_text_description
from typing import Sequence

import chainlit as cl
from chainlit.playground.config import add_llm_provider
from chainlit.playground.providers.langchain import LangchainGenericProvider

from dotenv import load_dotenv

load_dotenv()

# set_verbose(True)
# set_debug(True)


MODEL = "dolphin-mixtral:8x7b-v2.7-q3_K_L"

SYSTEM_PROMPT = """
You are a question answering assistant using tools to help answer my questions.

TOOLS
------
You have access to the following tools:

{tools}

RESPONSE FORMAT INSTRUCTIONS
----------------------------

When responding use one of the two formats:

**Format 1:**
Use this response format if you know the answer and want to respond directly. 
Markdown code snippet formatted in the following schema:

```json
{{
    "action": "Final Answer",
    "action_input": string // You should put what you want to return here.
}}
```

**Format 2:**
Use this response format if you need to use a tool to find the answer.
Markdown code snippet formatted in the following schema:

```json
{{
    "action": string, // The action to take. Must be one of: {tool_names}
    "action_input": string // The valid input to the action.
}}
```
"""

HUMAN_PROMPT = """
USER'S INPUT
--------------------
{input}

(Respond with a markdown code snippet of a json blob with a single action, and NOTHING else.)
"""

TEMPLATE_TOOL_RESPONSE = """
TOOL RESPONSE: 
---------------------
{observation}

USER'S INPUT
--------------------
What is the response to my last comment?
Use the response from the tool to generate the answer without mentioning the tool name! 
(Respond with a markdown code snippet of a json blob with a single action, and NOTHING else.)
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


@cl.on_chat_start
async def on_chat_start():
    llm = ChatOllama(
        model=MODEL,
        temperature=1,
        verbose=True,
        format="json",
        base_url="http://ollama:11434",
    )

    # Add the LLM provider
    add_llm_provider(
        LangchainGenericProvider(
            # It is important that the id of the provider matches the _llm_type
            id=llm._llm_type,
            # The name is not important. It will be displayed in the UI.
            name="dolphin-mixtral",
            # This should always be a Langchain llm instance (correctly configured)
            llm=llm,
            # If the LLM works with messages, set this to True
            is_chat=True,
        )
    )

    tools = [
        WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper()),
        ArxivQueryRun(),
        SemanticScholarQueryRun(),
        PubmedQueryRun(),
        # DuckDuckGoSearchRun(),
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
        PythonREPLTool(
            description="Use this to execute valid python code. Code needs to end with `print($RESULT)`.",
            # return_direct=True,
        ),
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
        k=3, memory_key="chat_history", output_key="output", return_messages=True
    )

    executor = AgentExecutor(
        agent=create_json_chat_agent(llm, tools, prompt),
        tools=tools,
        verbose=True,
        max_iterations=3,
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
