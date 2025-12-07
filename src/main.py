"""CLI entry point for RAG Evaluation Framework."""
import asyncio
import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.api.document_uploader import DocumentUploader
from src.api.rag_client import RAGClient
from src.config.settings import RAGEvalError, load_config, get_azure_openai_config, get_rag_api_key
from src.evaluation.orchestrator import EvaluationOrchestrator
from src.input.console_handler import (
    confirm_action,
    display_error,
    display_results,
    display_success,
    display_welcome,
    get_question_interactive,
)
from src.input.csv_handler import load_questions_from_csv, load_responses_from_csv, save_responses_to_csv
from src.reporting.report_generator import ReportGenerator
from src.utils.logger import setup_logging

console = Console()


@click.group()
@click.option('--config', '-c', default='config.yaml', help='Config file path')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
@click.pass_context
def cli(ctx, config, verbose):
    """RAG Evaluation Framework - Evaluate your RAG pipeline quality."""
    ctx.ensure_object(dict)

    try:
        ctx.obj['settings'] = load_config(config)
        ctx.obj['verbose'] = verbose

        # Setup logging
        log_level = "DEBUG" if verbose else ctx.obj['settings'].logging.level
        log_file = ctx.obj['settings'].logging.outputs.file.path if ctx.obj['settings'].logging.outputs.file.enabled else None
        setup_logging(log_level, log_file)

    except RAGEvalError as e:
        console.print(f"[bold red]Configuration Error:[/bold red] {e.message}")
        sys.exit(1)


# ============================================================================
# Document Commands
# ============================================================================

@cli.group()
def docs():
    """Document management commands."""
    pass


@docs.command('upload')
@click.argument('file_path', type=click.Path(exists=True))
@click.option('--collection', '-c', default='default', help='Target collection')
@click.option('--metadata', '-m', default=None, help='JSON metadata')
@click.pass_context
def upload_doc(ctx, file_path, collection, metadata):
    """Upload a document to RAG API."""
    settings = ctx.obj['settings']

    try:
        uploader = DocumentUploader(settings)

        # Parse metadata if provided
        meta_dict = None
        if metadata:
            import json
            meta_dict = json.loads(metadata)

        console.print(f"[dim]Uploading {file_path}...[/dim]")

        result = uploader.upload_document_sync(file_path, meta_dict, collection)

        console.print(Panel.fit(
            f"[green]Upload Successful[/green]\n\n"
            f"Document ID: {result.document_id}\n"
            f"Filename: {result.filename}\n"
            f"Status: {result.status}\n"
            f"Chunks Created: {result.chunks_created or 'N/A'}",
            title="Upload Result"
        ))

    except RAGEvalError as e:
        display_error(e.message)
        sys.exit(1)


@docs.command('upload-batch')
@click.argument('directory', type=click.Path(exists=True))
@click.option('--recursive', '-r', is_flag=True, help='Include subdirectories')
@click.option('--filter', '-f', 'file_filter', default=None, help='Extensions filter (comma-separated)')
@click.option('--collection', '-c', default='default', help='Target collection')
@click.pass_context
def upload_batch(ctx, directory, recursive, file_filter, collection):
    """Upload multiple documents from directory."""
    settings = ctx.obj['settings']

    try:
        uploader = DocumentUploader(settings)

        # Parse filter
        extensions = None
        if file_filter:
            extensions = [f".{ext.strip().lstrip('.')}" for ext in file_filter.split(',')]

        result = uploader.upload_batch_sync(directory, recursive, extensions, collection)

        console.print(Panel.fit(
            f"[green]Batch Upload Complete[/green]\n\n"
            f"Total Files: {result.total}\n"
            f"Successful: {result.successful}\n"
            f"Failed: {result.failed}",
            title="Batch Upload Result"
        ))

        if result.errors:
            console.print("\n[yellow]Errors:[/yellow]")
            for err in result.errors[:5]:
                console.print(f"  - {err['filename']}: {err['error']}")

    except RAGEvalError as e:
        display_error(e.message)
        sys.exit(1)


# ============================================================================
# Evaluation Commands
# ============================================================================

@cli.command()
@click.option('--input', '-i', 'input_file', type=click.Path(exists=True), help='Input CSV file with questions')
@click.option('--responses', '-r', 'responses_file', type=click.Path(exists=True), help='CSV with pre-fetched responses')
@click.option('--output', '-o', default='./data/output', help='Output directory')
@click.option('--framework', '-f', type=click.Choice(['both', 'ragas', 'deepeval']), default='both')
@click.option('--interactive', '-I', is_flag=True, help='Interactive mode')
@click.option('--collection', '-c', default=None, help='RAG collection name')
@click.pass_context
def evaluate(ctx, input_file, responses_file, output, framework, interactive, collection):
    """Run RAG evaluation."""
    settings = ctx.obj['settings']

    # Update framework settings based on option
    if framework == 'ragas':
        settings.evaluation.frameworks.deepeval.enabled = False
    elif framework == 'deepeval':
        settings.evaluation.frameworks.ragas.enabled = False

    # Update output directory
    settings.io.output.directory = output

    try:
        orchestrator = EvaluationOrchestrator(settings)
        report_gen = ReportGenerator(settings)

        if interactive:
            _run_interactive_mode(orchestrator, settings)
        elif responses_file:
            _run_responses_evaluation(responses_file, orchestrator, report_gen, settings)
        elif input_file:
            _run_batch_evaluation(input_file, orchestrator, report_gen, settings, collection)
        else:
            console.print("[yellow]Please provide --input, --responses, or use --interactive mode[/yellow]")
            sys.exit(1)

    except RAGEvalError as e:
        display_error(e.message)
        sys.exit(1)


def _run_interactive_mode(orchestrator, settings):
    """Run interactive evaluation mode."""
    display_welcome()

    while True:
        question = get_question_interactive()

        if question is None:
            console.print("[dim]Goodbye![/dim]")
            break

        if question == "BATCH_MODE":
            console.print("[yellow]Batch mode not implemented in interactive session. Use --input flag.[/yellow]")
            continue

        console.print("[dim]Processing...[/dim]")

        try:
            result = orchestrator.evaluate_single_sync(question)

            ragas_scores = {}
            deepeval_scores = {}
            explanations = {}

            if result.ragas_result:
                ragas_scores = {
                    "faithfulness": result.ragas_result.faithfulness.score,
                    "answer_relevancy": result.ragas_result.answer_relevancy.score,
                    "context_relevancy": result.ragas_result.context_relevancy.score,
                }
                explanations = {
                    "faithfulness": result.ragas_result.faithfulness.explanation,
                    "answer_relevancy": result.ragas_result.answer_relevancy.explanation,
                    "context_relevancy": result.ragas_result.context_relevancy.explanation,
                }

            if result.deepeval_result:
                deepeval_scores = {
                    "faithfulness": result.deepeval_result.faithfulness.score,
                    "answer_relevancy": result.deepeval_result.answer_relevancy.score,
                    "context_relevancy": result.deepeval_result.context_relevancy.score,
                }

            display_results(
                question,
                result.input.answer,
                ragas_scores,
                deepeval_scores,
                explanations,
            )

        except RAGEvalError as e:
            display_error(e.message)


def _run_batch_evaluation(input_file, orchestrator, report_gen, settings, collection):
    """Run batch evaluation from CSV questions."""
    console.print(f"[dim]Loading questions from {input_file}...[/dim]")

    questions = load_questions_from_csv(input_file)
    console.print(f"[green]Loaded {len(questions)} questions[/green]")

    # First, query RAG API and save responses
    console.print("\n[bold]Step 1: Querying RAG API[/bold]")
    rag_client = RAGClient(settings)

    responses_data = []
    for i, question in enumerate(questions):
        console.print(f"[dim]({i+1}/{len(questions)}) Querying: {question[:50]}...[/dim]")
        try:
            response = asyncio.run(rag_client.query(question, collection))
            responses_data.append({
                "question": question,
                "response": response.answer,
                "context": "; ".join(response.contexts),
            })
        except RAGEvalError as e:
            console.print(f"[red]Query failed: {e.message}[/red]")
            responses_data.append({
                "question": question,
                "response": f"Error: {e.message}",
                "context": "",
            })

    # Save responses to CSV
    responses_path = f"{settings.io.output.directory}/rag_responses.csv"
    save_responses_to_csv(responses_data, responses_path)
    console.print(f"[green]Saved responses to {responses_path}[/green]")

    # Run evaluation
    console.print("\n[bold]Step 2: Running Evaluation[/bold]")
    batch_result = asyncio.run(orchestrator.evaluate_from_responses(responses_data))

    # Generate reports
    console.print("\n[bold]Step 3: Generating Reports[/bold]")
    output_paths = report_gen.generate_report(batch_result)

    # Display summary
    _display_summary(batch_result, output_paths)


def _run_responses_evaluation(responses_file, orchestrator, report_gen, settings):
    """Run evaluation from pre-fetched responses CSV."""
    console.print(f"[dim]Loading responses from {responses_file}...[/dim]")

    responses = load_responses_from_csv(responses_file)
    console.print(f"[green]Loaded {len(responses)} responses[/green]")

    # Run evaluation
    console.print("\n[bold]Running Evaluation[/bold]")
    batch_result = asyncio.run(orchestrator.evaluate_from_responses(responses))

    # Generate reports
    console.print("\n[bold]Generating Reports[/bold]")
    output_paths = report_gen.generate_report(batch_result)

    # Display summary
    _display_summary(batch_result, output_paths)


def _display_summary(batch_result, output_paths):
    """Display evaluation summary."""
    stats = batch_result.get_summary_statistics()

    console.print("\n")
    console.print(Panel.fit(
        f"[bold green]Evaluation Complete[/bold green]\n\n"
        f"Total: {batch_result.total_evaluations}\n"
        f"Successful: {batch_result.successful_evaluations}\n"
        f"Failed: {batch_result.failed_evaluations}",
        title="Summary"
    ))

    # Show average scores
    if stats.get("ragas"):
        table = Table(title="Average Scores")
        table.add_column("Metric")
        table.add_column("RAGAS", justify="center")
        table.add_column("DeepEval", justify="center")

        for metric in ["faithfulness", "answer_relevancy", "context_relevancy"]:
            ragas_avg = stats.get("ragas", {}).get(metric, {}).get("avg", 0)
            deepeval_avg = stats.get("deepeval", {}).get(metric, {}).get("avg", 0)
            table.add_row(
                metric.replace("_", " ").title(),
                f"{ragas_avg:.3f}",
                f"{deepeval_avg:.3f}",
            )

        console.print(table)

    # Show output paths
    console.print("\n[bold]Generated Reports:[/bold]")
    for fmt, path in output_paths.items():
        console.print(f"  - {fmt}: {path}")


# ============================================================================
# Report Commands
# ============================================================================

@cli.group()
def report():
    """Report generation commands."""
    pass


@report.command('generate')
@click.option('--input', '-i', 'input_file', required=True, type=click.Path(exists=True), help='Input JSON results')
@click.option('--format', '-f', 'formats', multiple=True, default=['all'], help='Output formats')
@click.option('--output', '-o', default='./data/output', help='Output directory')
@click.pass_context
def generate_report(ctx, input_file, formats, output):
    """Generate reports from evaluation results."""
    import json
    settings = ctx.obj['settings']
    settings.io.output.directory = output

    # Determine formats
    if 'all' in formats:
        formats = ['csv', 'json', 'markdown', 'html']

    settings.io.output.report_formats = list(formats)

    try:
        # Load results
        with open(input_file) as f:
            data = json.load(f)

        # Convert to BatchEvaluationOutput
        from src.evaluation.models import BatchEvaluationOutput, EvaluationOutput, EvaluationInput, FrameworkResult, MetricScore

        results = []
        for eval_data in data.get("evaluations", []):
            # Reconstruct evaluation output
            eval_input = EvaluationInput(
                question=eval_data.get("question", ""),
                answer=eval_data.get("rag_response", {}).get("answer", ""),
                contexts=eval_data.get("rag_response", {}).get("contexts", []),
            )

            ragas_result = None
            if eval_data.get("ragas_evaluation"):
                ragas = eval_data["ragas_evaluation"]
                ragas_result = FrameworkResult(
                    framework="ragas",
                    faithfulness=MetricScore(score=ragas.get("faithfulness", {}).get("score", 0)),
                    answer_relevancy=MetricScore(score=ragas.get("answer_relevancy", {}).get("score", 0)),
                    context_relevancy=MetricScore(score=ragas.get("context_relevancy", {}).get("score", 0)),
                )

            deepeval_result = None
            if eval_data.get("deepeval_evaluation"):
                de = eval_data["deepeval_evaluation"]
                deepeval_result = FrameworkResult(
                    framework="deepeval",
                    faithfulness=MetricScore(score=de.get("faithfulness", {}).get("score", 0)),
                    answer_relevancy=MetricScore(score=de.get("answer_relevancy", {}).get("score", 0)),
                    context_relevancy=MetricScore(score=de.get("context_relevancy", {}).get("score", 0)),
                )

            results.append(EvaluationOutput(
                input=eval_input,
                ragas_result=ragas_result,
                deepeval_result=deepeval_result,
            ))

        batch_output = BatchEvaluationOutput(
            total_evaluations=len(results),
            successful_evaluations=len(results),
            results=results,
        )

        # Generate reports
        report_gen = ReportGenerator(settings)
        output_paths = report_gen.generate_report(batch_output)

        console.print("[green]Reports generated:[/green]")
        for fmt, path in output_paths.items():
            console.print(f"  - {fmt}: {path}")

    except Exception as e:
        display_error(str(e))
        sys.exit(1)


# ============================================================================
# Config Commands
# ============================================================================

@cli.group()
def config():
    """Configuration commands."""
    pass


@config.command('show')
@click.pass_context
def show_config(ctx):
    """Show current configuration."""
    settings = ctx.obj['settings']

    console.print(Panel.fit(
        f"RAG API: {settings.rag_api.base_url}\n"
        f"LLM Provider: {settings.llm.provider}\n"
        f"LLM Model: {settings.llm.deployment_name}\n"
        f"Frameworks: RAGAS={settings.evaluation.frameworks.ragas.enabled}, DeepEval={settings.evaluation.frameworks.deepeval.enabled}\n"
        f"Output Dir: {settings.io.output.directory}",
        title="Configuration"
    ))


@config.command('validate')
@click.pass_context
def validate_config(ctx):
    """Validate configuration and API keys."""
    settings = ctx.obj['settings']
    issues = []

    # Check RAG API key
    rag_key = get_rag_api_key()
    if not rag_key:
        issues.append("RAG_API_KEY not set")

    # Check Azure OpenAI
    try:
        azure_config = get_azure_openai_config()
        console.print("[green]Azure OpenAI configuration: OK[/green]")
    except RAGEvalError as e:
        issues.append(f"Azure OpenAI: {e.message}")

    if issues:
        console.print("\n[yellow]Issues found:[/yellow]")
        for issue in issues:
            console.print(f"  - {issue}")
    else:
        console.print("\n[green]All configurations valid![/green]")


@config.command('init')
@click.option('--force', '-f', is_flag=True, help='Overwrite existing files')
def init_config(force):
    """Initialize configuration files."""
    config_path = Path('config.yaml')
    env_path = Path('.env')

    if config_path.exists() and not force:
        console.print("[yellow]config.yaml already exists. Use --force to overwrite.[/yellow]")
    else:
        # Copy from .env.example template or create minimal
        console.print("[green]config.yaml created. Edit with your settings.[/green]")

    if not env_path.exists():
        env_content = """# RAG API Configuration
RAG_API_BASE_URL=http://localhost:8000
RAG_API_KEY=your-rag-api-key-here

# Azure OpenAI Configuration
AZURE_OPENAI_API_KEY=your-azure-openai-key-here
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4
AZURE_OPENAI_API_VERSION=2024-02-15-preview
"""
        with open(env_path, 'w') as f:
            f.write(env_content)
        console.print("[green].env created. Edit with your API keys.[/green]")


# ============================================================================
# Health Check Command
# ============================================================================

@cli.command()
@click.pass_context
def health(ctx):
    """Check RAG API connectivity and LLM API keys."""
    settings = ctx.obj['settings']

    console.print("[bold]Health Check[/bold]\n")

    # Check RAG API
    console.print("Checking RAG API...")
    try:
        import httpx
        response = httpx.get(f"{settings.rag_api.base_url}/health", timeout=5)
        if response.status_code == 200:
            console.print(f"  [green]RAG API: OK ({settings.rag_api.base_url})[/green]")
        else:
            console.print(f"  [yellow]RAG API: Status {response.status_code}[/yellow]")
    except Exception as e:
        console.print(f"  [red]RAG API: Failed - {str(e)}[/red]")

    # Check Azure OpenAI
    console.print("\nChecking Azure OpenAI...")
    try:
        azure_config = get_azure_openai_config()
        console.print(f"  [green]Azure OpenAI: Configured[/green]")
        console.print(f"  [dim]Endpoint: {azure_config['endpoint']}[/dim]")
        console.print(f"  [dim]Deployment: {azure_config['deployment_name']}[/dim]")
    except RAGEvalError as e:
        console.print(f"  [red]Azure OpenAI: {e.message}[/red]")


def main():
    """Main entry point."""
    cli(obj={})


if __name__ == '__main__':
    main()
