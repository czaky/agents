# JSON chat agent using dolphin-mixtral:8x7b-v2.7-q3_K_L

The agent uses several tools and communicates
with LangChain agent executor using JSON.

The following are the available tools:
wikipedia, arxiv, semanticscholar,
pub_med, searx_search, Python_REPL.

The prompts have been optimized for:
`dolphin-mixtral-8x7b-Q3`
quantized model that runs in 20GB VRAM.

This is only an example and will break with
other models and in other applications.

The UI is provided by chainlit.

### Installing

TODO: Add docker-compose and Dockerfile here.

The app will access ollama models at:
`http://ollama:11434`

Ollama needs to pull and start the:
`dolphin-mixtral:8x7b-v2.7-q3_K_L`
(or any comparable) model.

It will access a SearxNG instance at:
`http://searxng:8080`

Those two servers should be setup by the docker-compose (TODO).

After installing the requirements:

```
pip install -r requirements.txt
```

### Running

it should be possible to run the agent:

```
chainlit run -w app.py
```
