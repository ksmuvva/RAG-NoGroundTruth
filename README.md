# RAG Evaluation Framework

A local Python framework for evaluating RAG (Retrieval-Augmented Generation) pipeline quality using RAGAS and DeepEval metrics.

## Features

- **Document Upload**: Upload PDF, TXT, DOCX, and other documents to your RAG API
- **Batch Evaluation**: Evaluate multiple questions from CSV files
- **Dual Framework**: Compare results from both RAGAS and DeepEval
- **Three Key Metrics**:
  - Faithfulness: Is the answer grounded in context?
  - Answer Relevancy: Does the answer address the question?
  - Context Relevancy: Is the retrieved context relevant?
- **Multi-format Reports**: CSV, JSON, Markdown, and HTML outputs
- **Interactive Mode**: Test individual questions interactively

## Installation

```bash
# Clone the repository
cd rag-eval-framework

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Configuration

1. Copy the environment template:
```bash
cp .env.example .env
```

2. Edit `.env` with your API keys:
```env
# RAG API
RAG_API_BASE_URL=http://localhost:8000
RAG_API_KEY=your-rag-api-key

# Azure OpenAI
AZURE_OPENAI_API_KEY=your-azure-key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4
AZURE_OPENAI_API_VERSION=2024-02-15-preview
```

3. Review `config.yaml` for additional settings.

## Quick Start

### 1. Upload Documents

```bash
# Upload single document
python -m src.main docs upload ./data/documents/manual.pdf

# Upload batch from directory
python -m src.main docs upload-batch ./data/documents/ --recursive
```

### 2. Run Evaluation

**From CSV (questions only)**:
```bash
python -m src.main evaluate --input ./data/input/questions.csv
```

**From CSV (pre-fetched responses)**:
```bash
python -m src.main evaluate --responses ./data/output/rag_responses.csv
```

**Interactive mode**:
```bash
python -m src.main evaluate --interactive
```

### 3. Generate Reports

```bash
python -m src.main report generate --input ./data/output/eval_report.json --format all
```

## CLI Commands

```bash
# Document management
rag-eval docs upload <file>           # Upload single document
rag-eval docs upload-batch <dir>      # Upload directory

# Evaluation
rag-eval evaluate -i <csv>            # Evaluate from questions CSV
rag-eval evaluate -r <csv>            # Evaluate from responses CSV
rag-eval evaluate -I                  # Interactive mode

# Reports
rag-eval report generate -i <json>    # Generate reports

# Configuration
rag-eval config show                  # Show current config
rag-eval config validate              # Validate API keys
rag-eval health                       # Check API connectivity
```

## Input Formats

**Questions CSV** (`data/input/questions.csv`):
```csv
question
"What is machine learning?"
"How does search work?"
```

**Responses CSV** (`data/output/rag_responses.csv`):
```csv
question,response,context
"What is ML?","Machine learning is...","ML is a branch of AI..."
```

## Output Reports

Reports are generated in `data/output/` with timestamp:
- `eval_report_YYYYMMDD_HHMMSS.csv` - Detailed results
- `eval_report_YYYYMMDD_HHMMSS.json` - Machine-readable
- `eval_report_YYYYMMDD_HHMMSS.md` - Human-readable
- `eval_report_YYYYMMDD_HHMMSS.html` - Styled report

## Project Structure

```
rag-eval-framework/
├── src/
│   ├── main.py              # CLI entry point
│   ├── config/              # Configuration
│   ├── input/               # Input handlers
│   ├── api/                 # RAG API client
│   ├── evaluation/          # RAGAS & DeepEval
│   ├── reporting/           # Report generation
│   └── utils/               # Utilities
├── data/
│   ├── input/               # Input CSVs
│   ├── documents/           # Documents to upload
│   └── output/              # Generated reports
├── tests/                   # Unit tests
├── config.yaml              # Configuration
└── requirements.txt         # Dependencies
```

## Running Tests

```bash
pytest tests/ -v
```

## Metrics Explained

| Metric | Description | Good Score |
|--------|-------------|------------|
| Faithfulness | Measures if the answer is factually consistent with the provided context | > 0.8 |
| Answer Relevancy | Measures if the answer directly addresses the question | > 0.8 |
| Context Relevancy | Measures if retrieved context is relevant to the question | > 0.7 |

## Troubleshooting

**RAG API connection failed**:
- Check `RAG_API_BASE_URL` in `.env`
- Verify the API is running: `curl http://localhost:8000/health`

**Azure OpenAI errors**:
- Verify `AZURE_OPENAI_API_KEY` and `AZURE_OPENAI_ENDPOINT`
- Check deployment name matches your Azure resource

**Empty evaluation results**:
- Ensure your RAG API returns `answer` and `contexts` fields
- Check field mappings in `config.yaml`

## License

MIT
