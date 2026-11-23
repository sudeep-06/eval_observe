# Evaluation & Observability for AI Applications

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![LlamaIndex](https://img.shields.io/badge/LlamaIndex-RAG-orange.svg)](https://www.llamaindex.ai/)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--3.5-green.svg)](https://platform.openai.com/)
[![Opik](https://img.shields.io/badge/Comet-Opik-purple.svg)](https://www.comet.com/site/products/opik/)

An end-to-end example for tracing, evaluating, and observing a Retrieval-Augmented Generation (RAG) application using **LlamaIndex**, **OpenAI**, and **Comet Opik**.

The project shows how to:

- Log functions and OpenAI calls
- Trace LlamaIndex indexing and query operations
- Build a RAG pipeline from local documents
- Prepare a question-and-answer evaluation dataset
- Measure hallucination, answer relevance, context precision, and context recall
- Upload experiments and results to Opik for analysis

## Repository Structure

```text
.
├── data/
│   ├── paul_graham/          # Source documents used by the RAG pipeline
│   └── test.csv              # Evaluation questions, answers, and contexts
├── data_gen.ipynb            # Dataset-generation notebook
├── demo.ipynb                # Tracing and RAG evaluation walkthrough
└── README.md
```

## How It Works

1. `SimpleDirectoryReader` loads documents from `data/paul_graham`.
2. LlamaIndex creates a vector index from the documents.
3. OpenAI generates answers to user queries using the retrieved context.
4. Opik's OpenAI and LlamaIndex integrations capture traces and LLM activity.
5. A CSV dataset is converted into an Opik evaluation dataset.
6. Opik runs the RAG application against the dataset and reports evaluation metrics.

## Features

- Function tracing with `@track`
- Automatic OpenAI call instrumentation
- LlamaIndex callback-based tracing
- Local vector-indexed RAG retrieval
- CSV-based evaluation datasets
- Context-aware RAG metrics
- Hosted or local Opik support

## Prerequisites

- Python 3.10 or later
- An OpenAI account and API key
- A Comet account and Opik API key
- Jupyter Notebook or JupyterLab

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/sudeep-06/eval_observe.git
cd eval_observe
```

### 2. Create a virtual environment

**macOS/Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows**

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install opik llama-index llama-index-agent-openai llama-index-llms-openai openai python-dotenv pandas jupyter
```

## API Keys

Create a `.env` file in the repository root:

```dotenv
OPENAI_API_KEY=your-openai-api-key
COMET_API_KEY=your-comet-api-key
```

You can obtain credentials from:

- [OpenAI API keys](https://platform.openai.com/api-keys)
- [Comet / Opik](https://www.comet.com/signup)

Keep API keys out of source control. Add `.env` to `.gitignore` if it is not already ignored.

## Configure Opik

Before running the demo, configure Opik in a notebook or terminal:

```python
import opik

opik.configure(use_local=False)
```

Set `use_local=True` instead if you are running an Opik instance locally. See the [Opik documentation](https://www.comet.com/docs/opik/) for setup options.

## Download the sample documents

The notebook creates `data/paul_graham` and downloads a Paul Graham essay:

```python
import os
import requests

os.makedirs("./data/paul_graham", exist_ok=True)

url = "https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt"
response = requests.get(url)

with open("./data/paul_graham/paul_graham_essay.txt", "wb") as file:
    file.write(response.content)
```

The repository may already contain the required data directory.

## Run the Demo

Start Jupyter:

```bash
jupyter notebook demo.ipynb
```

Then run the notebook cells in order.

### What the demo covers

- Loading environment variables
- Configuring Opik
- Tracing a Python function with `@track`
- Tracking OpenAI API calls
- Setting up the LlamaIndex Opik callback
- Building a vector index
- Querying the RAG pipeline
- Loading `data/test.csv`
- Creating an Opik dataset
- Running evaluations
- Viewing traces and experiments in Opik

## Evaluation Dataset

The notebook loads `data/test.csv`, which contains:

| Column | Description |
| --- | --- |
| `Question` | The question sent to the RAG application |
| `Answer` | The expected answer |
| `Context` | Supporting source text |

Each row is transformed into an Opik dataset item:

```python
import pandas as pd

from opik import Opik

frame = pd.read_csv("data/test.csv")
qa_pairs = [
    {
        "input": row["Question"],
        "expected_output": row["Answer"],
        "context": row["Context"],
    }
    for _, row in frame.iterrows()
]

client = Opik()
dataset = client.get_or_create_dataset(name="Test dataset")
dataset.insert(qa_pairs)
```

## Evaluation Metrics

The demo evaluates RAG outputs using the following Opik metrics:

| Metric | Measures |
| --- | --- |
| Hallucination | Whether the response introduces unsupported information |
| Answer relevance | How relevant the answer is to the input question |
| Context precision | How much retrieved context is relevant to the answer |
| Context recall | How much necessary context is recovered |

```python
from opik.evaluation.metrics import (
    Hallucination,
    AnswerRelevance,
    ContextPrecision,
    ContextRecall,
)

metrics = [
    Hallucination(),
    AnswerRelevance(),
    ContextPrecision(),
    ContextRecall(),
]
```

## Run an Evaluation

The evaluation task sends each dataset question through the RAG query engine:

```python
from opik import track


@track
def my_llm_application(question: str) -> str:
    response = query_engine.query(question)
    return str(response)


def evaluation_task(item):
    return {
        "output": my_llm_application(item["input"])
    }
```

Then run the Opik evaluation:

```python
from opik.evaluation import evaluate

evaluation = evaluate(
    dataset=dataset,
    task=evaluation_task,
    scoring_metrics=metrics,
    experiment_config={"model": "gpt-3.5-turbo"},
)
```

Evaluation results and traces can be explored in the Opik application.

## Notebooks

| Notebook | Purpose |
| --- | --- |
| `demo.ipynb` | End-to-end tracing and RAG evaluation example |
| `data_gen.ipynb` | Dataset-generation workflow |

## Troubleshooting

### Missing OpenAI API key

Confirm that `.env` contains:

```dotenv
OPENAI_API_KEY=your-openai-api-key
```

Also confirm that `load_dotenv()` runs before creating the OpenAI or LlamaIndex clients.

### Missing Comet API key

Sign in to Comet, create an API key, and add it to `.env`:

```dotenv
COMET_API_KEY=your-comet-api-key
```

### Missing data

Run the document-download cell in `demo.ipynb`, or place supported documents inside:

```text
data/paul_graham/
```

### Package import errors

Reinstall the project dependencies inside the active virtual environment:

```bash
python -m pip install opik llama-index llama-index-agent-openai llama-index-llms-openai openai --upgrade
```

## License

No license is currently specified for this repository.
