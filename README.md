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

## Data Format Examples

This section provides detailed examples of the data at each stage of the pipeline.

### Input: Questions CSV

The input file contains questions to be sent to the RAG system:

```csv
question
What is the main purpose of this document?
How does the system handle user authentication?
What are the key performance metrics described?
What security measures are implemented?
How is data stored and retrieved?
```

### RAG API Response

The RAG API returns an answer and the retrieved contexts for each question:

```json
{
    "answer": "The document describes a cloud-based data management system designed to handle large-scale data processing with high availability and fault tolerance.",
    "contexts": [
        "The primary purpose of the CloudData system is to provide enterprise-grade data management capabilities...",
        "Key design goals include: high availability (99.99% uptime), fault tolerance through automatic failover...",
        "The system is architected for handling petabyte-scale datasets while maintaining sub-second query response times..."
    ]
}
```

### Intermediate Data: RAG Output CSV

After querying the RAG API, the pipeline creates a CSV with three columns:

| Questions | Answer | Context |
|-----------|--------|---------|
| What is the main purpose of this document? | The document describes a cloud-based data management system designed to handle large-scale data processing with high availability and fault tolerance. | The primary purpose of the CloudData system is to provide enterprise-grade data management capabilities... \| Key design goals include: high availability (99.99% uptime), fault tolerance through automatic failover... \| The system is architected for handling petabyte-scale datasets while maintaining sub-second query response times... |
| How does the system handle user authentication? | The system implements OAuth 2.0 with support for multi-factor authentication (MFA) and integrates with enterprise identity providers via SAML 2.0. | Authentication is handled through the AuthService module which implements OAuth 2.0... \| Multi-factor authentication can be enabled on a per-user or organization-wide basis... |

**Note**: Multiple contexts are joined with ` | ` (pipe with spaces) as the delimiter.

### Final Output: Evaluation Results CSV

The evaluation adds metric scores and pass/fail indicators:

| Questions | Answer | Context | RAGAS_Faithfulness | RAGAS_Answer_Relevancy | RAGAS_Context_Precision | DeepEval_Faithfulness | DeepEval_Answer_Relevancy | DeepEval_Contextual_Relevancy | Faithfulness_Pass | Answer_Relevancy_Pass | Context_Relevancy_Pass |
|-----------|--------|---------|-------------------|----------------------|------------------------|----------------------|--------------------------|------------------------------|-------------------|----------------------|----------------------|
| What is the main purpose of this document? | The document describes a cloud-based... | The primary purpose... | 0.923 | 0.891 | 0.850 | 0.889 | 0.912 | 0.834 | True | True | True |
| How does the system handle user authentication? | The system implements OAuth 2.0... | Authentication is handled... | 0.867 | 0.945 | 0.780 | 0.834 | 0.901 | 0.756 | True | True | False |

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

This pipeline uses the **original RAGAS and DeepEval libraries** directly to compute evaluation metrics. No custom implementations are used—all metric calculations leverage the official library implementations.

### Overview of Metrics

| Metric | RAGAS | DeepEval | What It Measures |
|--------|-------|----------|------------------|
| Faithfulness | `faithfulness` | `FaithfulnessMetric` | Is the answer factually consistent with the context? |
| Answer Relevancy | `answer_relevancy` | `AnswerRelevancyMetric` | Does the answer address the question? |
| Context Quality | `context_precision` | `ContextualRelevancyMetric` | Is the retrieved context appropriate? |

---

### RAGAS Metrics (Original Library)

RAGAS (Retrieval Augmented Generation Assessment) is used directly via the `ragas` Python package. The pipeline imports and uses the original RAGAS metrics:

```python
from ragas import evaluate as ragas_eval
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from datasets import Dataset

# Convert data to RAGAS Dataset format
dataset = Dataset.from_dict({
    "question": [question],
    "answer": [answer],
    "contexts": [contexts]  # List of context strings
})

# Run evaluation using original RAGAS metrics
ragas_result = ragas_eval(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_precision]
)

# Extract scores from RAGAS EvaluationResult
ragas_df = ragas_result.to_pandas()
faithfulness_score = float(ragas_df["faithfulness"].iloc[0])
answer_relevancy_score = float(ragas_df["answer_relevancy"].iloc[0])
context_precision_score = float(ragas_df["context_precision"].iloc[0])
```

#### RAGAS Faithfulness
- **Library**: `ragas.metrics.faithfulness`
- **Range**: 0.0 to 1.0
- **Calculation**: Decomposes the answer into individual claims, then verifies each claim against the provided context using an LLM. The score is the ratio of supported claims to total claims.
- **Formula**: `faithfulness = (number of claims supported by context) / (total number of claims)`

#### RAGAS Answer Relevancy
- **Library**: `ragas.metrics.answer_relevancy`
- **Range**: 0.0 to 1.0
- **Calculation**: Generates hypothetical questions that the answer could address, then computes semantic similarity between these generated questions and the original question.
- **Method**: Uses embedding-based similarity to measure how well the answer addresses the question.

#### RAGAS Context Precision
- **Library**: `ragas.metrics.context_precision`
- **Range**: 0.0 to 1.0
- **Calculation**: Evaluates whether relevant contexts appear earlier in the retrieved context list. Higher scores mean more relevant contexts are ranked higher.
- **Note**: This measures **ranking quality**, not just relevance.

---

### DeepEval Metrics (Original Library)

DeepEval is used directly via the `deepeval` Python package. The pipeline imports and uses the original DeepEval metrics:

```python
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualRelevancyMetric
)
from deepeval.test_case import LLMTestCase
from deepeval.models import AzureOpenAI as DeepEvalAzureOpenAI

# Initialize DeepEval model (Azure OpenAI)
deepeval_model = DeepEvalAzureOpenAI(
    model=deployment_name,
    deployment_name=deployment_name,
    azure_openai_api_key=api_key,
    azure_endpoint=endpoint,
    openai_api_version=api_version
)

# Create test case with input, output, and retrieval context
test_case = LLMTestCase(
    input=question,
    actual_output=answer,
    retrieval_context=contexts  # List of context strings
)

# Initialize and measure each metric
faithfulness_metric = FaithfulnessMetric(threshold=0.8, model=deepeval_model)
faithfulness_metric.measure(test_case)
faithfulness_score = faithfulness_metric.score

answer_relevancy_metric = AnswerRelevancyMetric(threshold=0.8, model=deepeval_model)
answer_relevancy_metric.measure(test_case)
answer_relevancy_score = answer_relevancy_metric.score

contextual_relevancy_metric = ContextualRelevancyMetric(threshold=0.8, model=deepeval_model)
contextual_relevancy_metric.measure(test_case)
contextual_relevancy_score = contextual_relevancy_metric.score
```

#### DeepEval Faithfulness
- **Library**: `deepeval.metrics.FaithfulnessMetric`
- **Range**: 0.0 to 1.0
- **Calculation**: Uses an LLM to extract claims from the answer and verify each claim against the retrieval context. Measures factual consistency.
- **Similar to**: RAGAS faithfulness, but may use different prompting strategies.

#### DeepEval Answer Relevancy
- **Library**: `deepeval.metrics.AnswerRelevancyMetric`
- **Range**: 0.0 to 1.0
- **Calculation**: Uses an LLM to determine if the actual output (answer) addresses the input (question) appropriately.
- **Method**: Direct LLM-based assessment of relevance.

#### DeepEval Contextual Relevancy
- **Library**: `deepeval.metrics.ContextualRelevancyMetric`
- **Range**: 0.0 to 1.0
- **Calculation**: Evaluates whether the retrieved contexts are relevant to answering the input question.
- **Note**: This measures **retrieval relevance** (different from RAGAS context_precision which measures ranking).

---

### Pass/Fail Determination

A metric passes if **either** the RAGAS score **or** the DeepEval score meets the threshold:

```python
# Pass if either framework meets threshold (OR logic)
faithfulness_pass = (ragas_faithfulness >= threshold) or (deepeval_faithfulness >= threshold)
answer_relevancy_pass = (ragas_answer_relevancy >= threshold) or (deepeval_answer_relevancy >= threshold)
context_relevancy_pass = (ragas_context_precision >= threshold) or (deepeval_contextual_relevancy >= threshold)
```

**Default threshold**: `0.8` (80%) for all metrics, configurable in `config.yaml`.

---

### Key Differences Between RAGAS and DeepEval Context Metrics

| Aspect | RAGAS `context_precision` | DeepEval `ContextualRelevancyMetric` |
|--------|---------------------------|--------------------------------------|
| **Focus** | Ranking quality | Retrieval relevance |
| **Question** | Are relevant contexts ranked higher? | Are retrieved contexts relevant to the query? |
| **Use Case** | Evaluating retrieval ranking | Evaluating retrieval selection |

Both metrics are valuable: RAGAS helps ensure your best contexts appear first, while DeepEval ensures all retrieved contexts are actually relevant.

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
