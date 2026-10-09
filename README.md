# Course FAQ RAG experiments

This repository contains experimental Retrieval-Augmented Generation (RAG) implementations for answering questions about DataTalks.Club course FAQs. It demonstrates local keyword search, retrieval with Weaviate, OpenAI-powered answers, tool-calling agents, and a LangGraph workflow.

## Implementations

- **Local RAG (`rag/`):** fetch FAQ documents, index them with `minsearch`, retrieve relevant entries, and construct a prompt for OpenAI.
- **Weaviate RAG (`rag_weaviate/`):** retrieve context from a Weaviate Cloud collection and send it to OpenAI.
- **Basic agentic RAG (`basic_agentic_rag/`):** an OpenAI tool-calling loop that can search the FAQ collection multiple times.
- **LangGraph (`langgraph_rag/`):** a state graph that alternates between model reasoning and Weaviate search tool calls.
- **Tutorial notebooks (`test/`):** interactive exercises covering model calls, FAQ data, retrieval, prompting, and agents.

## Requirements

- Python 3.13 (selected by `.python-version`)
- [uv](https://docs.astral.sh/uv/) for dependency and environment management
- An OpenAI API key for model-backed examples
- A Weaviate Cloud instance and API key for Weaviate-backed examples

## Setup

From the repository root, install the dependencies:

```bash
uv sync
```

Create a `.env` file in the repository root and set the credentials needed by the example you plan to run:

```dotenv
OPENAI_API_KEY=your-openai-api-key
WEAVIATE_URL=https://your-cluster.weaviate.network
WEAVIATE_API_KEY=your-weaviate-api-key
```

Local `minsearch` retrieval does not require Weaviate credentials. Calls to OpenAI require `OPENAI_API_KEY`; the Weaviate examples also require `WEAVIATE_URL` and `WEAVIATE_API_KEY`. Keep real credentials out of source control.

## Run examples

Run commands from the repository root.

### Weaviate retrieval and generation

Requires a populated `FAQDocuments` collection:

```bash
uv run python rag_weaviate/rag_helper.py --question "Can I join the course after it starts?"
```

### Basic tool-calling agent

This agent searches `FAQDocuments` and can make multiple searches before answering:

```bash
uv run ./basic_agentic_rag/rag_helper.py --question "I just discovered the course. Can I join it?"
```

### LangGraph agent

```bash
uv run python langgraph_rag/run.py
```

The LangGraph runner connects to Weaviate and uses a LangChain chat model with the Weaviate search tool. The script defines a `--question` option but currently sends a hard-coded sample question to the graph, so the option does not affect the request yet.

## Populate Weaviate

The ingestion script downloads FAQ data from DataTalks.Club and loads it into `FAQDocuments`:

```bash
uv run python rag_weaviate/injest.py
```

> **Warning:** this script calls `client.collections.delete_all()` before creating `FAQDocuments`. That removes **all collections** from the connected Weaviate instance. Use a disposable instance or modify the script before running it where existing data must be retained.

## Notebooks

The notebooks under `test/` introduce the project step by step:

1. `01-setup.ipynb` — call an OpenAI model.
2. `02-data.ipynb` — download FAQ data and build/query a `minsearch` index.
3. `03-prompt.ipynb` — build retrieved context and prompts.
4. `04-llm.ipynb` and `04-agent.ipynb` — explore model calls, tools, and agent patterns.

Open them in VS Code or start Jupyter with `uv run jupyter lab`. Run cells in order. Cells that call OpenAI require a valid API key.

## Project structure

```text
basic_agentic_rag/   OpenAI tool-calling FAQ agent
langgraph/           LangGraph graph, state, routing, tools, and runner
rag/                 FAQ download, minsearch index, and local RAG helper
rag_weaviate/        Weaviate ingestion and RAG helper
test/                Learning notebooks
```

## Notes

- FAQ entries are downloaded from public DataTalks.Club endpoints by the ingestion examples; endpoint availability and contents may change.
- All Weaviate-backed examples expect a collection named `FAQDocuments`.
- The LangGraph search tool returns up to three nearby entries and does not currently filter results by course.
- These examples are not yet unified behind one application CLI. Some scripts also retain hard-coded sample values; inspect the script before relying on command-line options.
