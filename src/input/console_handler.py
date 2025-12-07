"""Interactive console input handler."""
from typing import Callable, Optional

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table


console = Console()


def display_welcome():
    """Display welcome banner."""
    console.print(Panel.fit(
        "[bold cyan]RAG Evaluation Framework - Interactive Mode[/bold cyan]",
        border_style="cyan"
    ))
    console.print()
    console.print("[dim]Commands:[/dim]")
    console.print("  [green]quit[/green]  - Exit interactive mode")
    console.print("  [green]batch[/green] - Switch to CSV batch mode")
    console.print()


def get_question_interactive() -> Optional[str]:
    """
    Get a question from interactive console input.

    Returns:
        Question string or None if user wants to quit
    """
    question = Prompt.ask(
        "[bold green]>[/bold green] Enter your question",
        default=""
    )

    if question.lower() in ['quit', 'exit', 'q']:
        return None

    if question.lower() == 'batch':
        return "BATCH_MODE"

    return question.strip() if question.strip() else None


def display_results(
    question: str,
    answer: str,
    ragas_scores: dict,
    deepeval_scores: dict,
    explanations: Optional[dict] = None
):
    """
    Display evaluation results in formatted table.

    Args:
        question: Original question
        answer: RAG answer
        ragas_scores: RAGAS metric scores
        deepeval_scores: DeepEval metric scores
        explanations: Optional explanations per metric
    """
    console.print()
    console.rule("[bold]Results[/bold]")
    console.print()

    # Question and answer
    console.print(f"[bold]Question:[/bold] {question}")
    console.print()
    console.print(f"[bold]Answer:[/bold] {answer[:500]}{'...' if len(answer) > 500 else ''}")
    console.print()

    # Scores table
    table = Table(title="Evaluation Scores", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("RAGAS", justify="center")
    table.add_column("DeepEval", justify="center")
    table.add_column("Explanation", style="dim")

    metrics = ["faithfulness", "answer_relevancy", "context_relevancy"]

    for metric in metrics:
        ragas_score = ragas_scores.get(metric, 0)
        deepeval_score = deepeval_scores.get(metric, 0)

        # Color based on score
        ragas_color = _get_score_color(ragas_score)
        deepeval_color = _get_score_color(deepeval_score)

        explanation = ""
        if explanations:
            explanation = explanations.get(metric, "")[:50]
            if len(explanations.get(metric, "")) > 50:
                explanation += "..."

        table.add_row(
            metric.replace("_", " ").title(),
            f"[{ragas_color}]{ragas_score:.2f}[/{ragas_color}]",
            f"[{deepeval_color}]{deepeval_score:.2f}[/{deepeval_color}]",
            explanation
        )

    console.print(table)
    console.print()


def _get_score_color(score: float) -> str:
    """Get color based on score value."""
    if score >= 0.8:
        return "green"
    elif score >= 0.6:
        return "yellow"
    else:
        return "red"


def display_progress(message: str, completed: int = 0, total: int = 0):
    """Display progress message."""
    if total > 0:
        console.print(f"[dim]{message}[/dim] [{completed}/{total}]")
    else:
        console.print(f"[dim]{message}[/dim]")


def display_error(message: str):
    """Display error message."""
    console.print(f"[bold red]Error:[/bold red] {message}")


def display_success(message: str):
    """Display success message."""
    console.print(f"[bold green]Success:[/bold green] {message}")


def confirm_action(message: str, default: bool = False) -> bool:
    """
    Ask for user confirmation.

    Args:
        message: Confirmation message
        default: Default response

    Returns:
        True if confirmed, False otherwise
    """
    response = Prompt.ask(
        f"{message} [y/n]",
        default="y" if default else "n"
    )
    return response.lower() in ['y', 'yes']
