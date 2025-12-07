# RAG Evaluation Pipeline (No Ground Truth Required)

A streamlined pipeline for evaluating Retrieval-Augmented Generation (RAG) systems without requiring ground truth answers. This tool automatically evaluates RAG responses using industry-standard metrics from **RAGAS** and **DeepEval**.

## Features

- **No Ground Truth Required**: Evaluate RAG quality using faithfulness, answer relevancy, and context relevancy metrics
- **Dual Evaluation Framework**: Leverages both RAGAS and DeepEval for comprehensive assessment
- **Simple Workflow**: Upload documents → Query with questions → Get evaluation results
- **Multiple Input Formats**: Supports CSV questions, PDF, DOCX, and TXT documents
- **Configurable Thresholds**: Customize pass/fail criteria for each metric
- **Detailed Reports**: CSV output with per-question scores and summary statistics

## Prerequisites

- Python 3.9+
- Access to a RAG API endpoint (for document upload and querying)
- Azure OpenAI API credentials (for LLM-based evaluation)

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/ksmuvva/RAG-NoGroundTruth.git
   cd RAG-NoGroundTruth
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

### Step 1: Create Environment File

Copy the example environment file and configure your credentials:

```bash
cp .env.example .env
```

### Step 2: Configure Environment Variables

Edit the `.env` file with your actual credentials:

```env
# RAG API Configuration
RAG_API_BASE_URL=http://localhost:8000    # Your RAG API base URL
RAG_API_KEY=your_rag_api_key              # Your RAG API key

# Azure OpenAI Configuration (required for RAGAS/DeepEval evaluation)
AZURE_OPENAI_API_KEY=your_azure_openai_api_key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4
AZURE_OPENAI_API_VERSION=2024-02-15-preview
```

#### Environment Variables Explained

| Variable | Description | Required |
|----------|-------------|----------|
| `RAG_API_BASE_URL` | Base URL of your RAG API server | Yes |
| `RAG_API_KEY` | API key for authenticating with your RAG API | Yes |
| `AZURE_OPENAI_API_KEY` | Azure OpenAI API key for LLM-based evaluation | Yes |
| `AZURE_OPENAI_ENDPOINT` | Your Azure OpenAI resource endpoint | Yes |
| `AZURE_OPENAI_DEPLOYMENT_NAME` | Name of your GPT-4 deployment | Yes |
| `AZURE_OPENAI_API_VERSION` | Azure OpenAI API version | Yes |

### Step 3: Configure Thresholds (Optional)

Edit `config.yaml` to customize evaluation thresholds:

```yaml
thresholds:
  faithfulness: 0.8        # Minimum score for faithfulness (0-1)
  answer_relevancy: 0.8    # Minimum score for answer relevancy (0-1)
  context_relevancy: 0.8   # Minimum score for context relevancy (0-1)
```

## Usage

### Basic Usage

Run the pipeline with a CSV file containing questions:

```bash
python rag_pipeline.py <input_file>
```

### Examples

**Evaluate with a questions CSV:**
```bash
python rag_pipeline.py data/sample_questions.csv
```

**Upload a document to the RAG system:**
```bash
python rag_pipeline.py document.pdf
```

### Input Format

#### CSV File Format

Your CSV file should contain a column with questions. The pipeline automatically detects columns containing "question" in the name:

```csv
question
What is the main purpose of this document?
How does the system handle authentication?
What are the key features described?
```

#### Supported Document Formats

For document upload:
- PDF (`.pdf`)
- Microsoft Word (`.docx`)
- Plain Text (`.txt`)

## Output

The pipeline generates evaluation results in the `data/evaluation_output_folder/` directory.

### Output CSV Columns

| Column | Description |
|--------|-------------|
| `Questions` | The original question |
| `Answer` | RAG system's response |
| `Context` | Retrieved context(s) used for the answer |
| `RAGAS_Faithfulness` | RAGAS faithfulness score (0-1) |
| `RAGAS_Answer_Relevancy` | RAGAS answer relevancy score (0-1) |
| `RAGAS_Context_Relevancy` | RAGAS context precision score (0-1) |
| `DeepEval_Faithfulness` | DeepEval faithfulness score (0-1) |
| `DeepEval_Answer_Relevancy` | DeepEval answer relevancy score (0-1) |
| `DeepEval_Context_Relevancy` | DeepEval contextual relevancy score (0-1) |
| `Faithfulness_Pass` | Whether faithfulness meets threshold |
| `Answer_Relevancy_Pass` | Whether answer relevancy meets threshold |
| `Context_Relevancy_Pass` | Whether context relevancy meets threshold |

### Sample Output

After running the pipeline, you'll see a summary like:

```
==================================================
EVALUATION SUMMARY
==================================================

Faithfulness:
  RAGAS:    0.850
  DeepEval: 0.820
  Pass Rate: 85.0% [PASS]

Answer Relevancy:
  RAGAS:    0.920
  DeepEval: 0.880
  Pass Rate: 90.0% [PASS]

Context Relevancy:
  RAGAS:    0.780
  DeepEval: 0.750
  Pass Rate: 70.0% [FAIL]
==================================================
```

## Evaluation Metrics

### Faithfulness
Measures whether the answer is factually consistent with the retrieved context. High faithfulness means the answer doesn't contain hallucinations.

### Answer Relevancy
Measures how relevant the generated answer is to the original question. High relevancy means the answer directly addresses the question.

### Context Relevancy
Measures how relevant the retrieved context is to answering the question. High relevancy means the RAG system retrieved appropriate documents.

## Project Structure

```
RAG-NoGroundTruth/
├── rag_pipeline.py      # Main pipeline script
├── config.yaml          # Configuration file
├── requirements.txt     # Python dependencies
├── .env.example         # Example environment variables
├── .gitignore           # Git ignore patterns
└── data/
    ├── sample_questions.csv      # Example questions file
    ├── RAG_output_folder/        # RAG query outputs
    ├── RAG_evaluator_input/      # Evaluator input copies
    └── evaluation_output_folder/ # Final evaluation results
```

## RAG API Requirements

Your RAG API must implement the following endpoints:

### Upload Endpoint
- **URL**: `POST /api/v1/documents/upload`
- **Headers**: `X-API-Key: <your_api_key>`
- **Body**: `multipart/form-data` with `file` field

### Query Endpoint
- **URL**: `POST /api/v1/query`
- **Headers**: `X-API-Key: <your_api_key>`, `Content-Type: application/json`
- **Body**: `{"question": "your question here"}`
- **Response**: `{"answer": "...", "contexts": ["context1", "context2"]}`

## Troubleshooting

### Common Issues

**1. "RAGAS not installed" warning**
```bash
pip install ragas
```

**2. "DeepEval not installed" warning**
```bash
pip install deepeval
```

**3. Azure OpenAI authentication errors**
- Verify your `AZURE_OPENAI_API_KEY` is correct
- Check that your endpoint URL ends with `.openai.azure.com`
- Ensure the deployment name matches your Azure resource

**4. RAG API connection errors**
- Verify `RAG_API_BASE_URL` is accessible
- Check that `RAG_API_KEY` is valid
- Ensure the API endpoints match your RAG system

**5. Timeout errors**
Increase the timeout in `config.yaml`:
```yaml
rag_api:
  timeout: 120  # Increase from default 60 seconds
```

## Alternative LLM Providers

While this pipeline is configured for Azure OpenAI, you can modify it to use other providers:

### Using OpenAI Directly
Set these environment variables instead:
```env
OPENAI_API_KEY=your_openai_api_key
```

### Using Other Providers
RAGAS and DeepEval support multiple LLM providers. Refer to their documentation:
- [RAGAS Documentation](https://docs.ragas.io/)
- [DeepEval Documentation](https://docs.confident-ai.com/)

## License

This project is provided as-is for evaluation purposes.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
