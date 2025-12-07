"""Multi-format report generator."""
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.api.schemas import EvaluationReport
from src.config.settings import Settings
from src.evaluation.models import BatchEvaluationOutput, EvaluationOutput
from src.reporting.statistics import calculate_comparison_summary, calculate_statistics
from src.utils.file_utils import generate_output_filename, write_json, write_text
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ReportGenerator:
    """Multi-format report generator."""

    def __init__(self, settings: Settings):
        """
        Initialize report generator.

        Args:
            settings: Application settings
        """
        self.settings = settings
        self.output_dir = settings.io.output.directory
        self.filename_template = settings.io.output.filename_template
        self.timestamp_format = settings.io.output.timestamp_format

    def generate_report(
        self,
        batch_output: BatchEvaluationOutput,
        formats: Optional[List[str]] = None
    ) -> Dict[str, str]:
        """
        Generate reports in multiple formats.

        Args:
            batch_output: Batch evaluation output
            formats: List of formats (csv, json, markdown, html)

        Returns:
            Dict mapping format to output file path
        """
        formats = formats or self.settings.io.output.report_formats

        # Build report object
        report = self._build_report(batch_output)

        # Generate outputs
        output_paths = {}
        base_name = generate_output_filename(
            self.filename_template,
            self.timestamp_format,
        )

        for fmt in formats:
            try:
                if fmt == "csv":
                    path = self.generate_csv(report, f"{self.output_dir}/{base_name}.csv")
                elif fmt == "json":
                    path = self.generate_json(report, f"{self.output_dir}/{base_name}.json")
                elif fmt == "markdown":
                    path = self.generate_markdown(report, f"{self.output_dir}/{base_name}.md")
                elif fmt == "html":
                    path = self.generate_html(report, f"{self.output_dir}/{base_name}.html")
                else:
                    logger.warning(f"Unknown format: {fmt}")
                    continue

                output_paths[fmt] = path
                logger.info(f"Generated {fmt} report", path=path)

            except Exception as e:
                logger.error(f"Failed to generate {fmt} report", error=str(e))

        return output_paths

    def _build_report(self, batch_output: BatchEvaluationOutput) -> EvaluationReport:
        """Build EvaluationReport from BatchEvaluationOutput."""
        # Calculate statistics
        stats = calculate_statistics(
            batch_output.results,
            {
                "faithfulness": self.settings.evaluation.thresholds.faithfulness,
                "answer_relevancy": self.settings.evaluation.thresholds.answer_relevancy,
                "context_relevancy": self.settings.evaluation.thresholds.context_relevancy,
            }
        )

        # Calculate comparison summary
        comparison = calculate_comparison_summary(
            batch_output.results,
            self.settings.evaluation.comparison.discrepancy_threshold,
        )

        # Convert to report format
        from src.api.schemas import SingleEvaluationResult, RAGResponse, RAGASResult, DeepEvalResult, MetricResult

        evaluations = []
        for result in batch_output.results:
            rag_response = RAGResponse(
                question=result.input.question,
                answer=result.input.answer,
                contexts=result.input.contexts,
            )

            ragas_eval = None
            if result.ragas_result:
                ragas_eval = RAGASResult(
                    faithfulness=MetricResult(
                        score=result.ragas_result.faithfulness.score,
                        explanation=result.ragas_result.faithfulness.explanation,
                    ),
                    answer_relevancy=MetricResult(
                        score=result.ragas_result.answer_relevancy.score,
                        explanation=result.ragas_result.answer_relevancy.explanation,
                    ),
                    context_relevancy=MetricResult(
                        score=result.ragas_result.context_relevancy.score,
                        explanation=result.ragas_result.context_relevancy.explanation,
                    ),
                    evaluation_time_ms=result.ragas_result.evaluation_time_ms,
                )

            deepeval_eval = None
            if result.deepeval_result:
                deepeval_eval = DeepEvalResult(
                    faithfulness=MetricResult(
                        score=result.deepeval_result.faithfulness.score,
                        explanation=result.deepeval_result.faithfulness.explanation,
                    ),
                    answer_relevancy=MetricResult(
                        score=result.deepeval_result.answer_relevancy.score,
                        explanation=result.deepeval_result.answer_relevancy.explanation,
                    ),
                    context_relevancy=MetricResult(
                        score=result.deepeval_result.context_relevancy.score,
                        explanation=result.deepeval_result.context_relevancy.explanation,
                    ),
                    evaluation_time_ms=result.deepeval_result.evaluation_time_ms,
                )

            evaluations.append(SingleEvaluationResult(
                evaluation_id=result.evaluation_id,
                timestamp=result.timestamp,
                question=result.input.question,
                rag_response=rag_response,
                ragas_evaluation=ragas_eval,
                deepeval_evaluation=deepeval_eval,
            ))

        return EvaluationReport(
            report_id=batch_output.batch_id,
            generated_at=datetime.now(),
            framework_versions={"ragas": "latest", "deepeval": "latest"},
            config_snapshot={},
            evaluations=evaluations,
            statistics=stats,
            comparison_summary=comparison,
        )

    def generate_csv(self, report: EvaluationReport, output_path: str) -> str:
        """Generate detailed CSV report."""
        rows = []

        for eval_result in report.evaluations:
            row = {
                "evaluation_id": eval_result.evaluation_id,
                "timestamp": eval_result.timestamp.isoformat(),
                "question": eval_result.question,
                "answer": eval_result.rag_response.answer[:500],
                "contexts": "; ".join(eval_result.rag_response.contexts)[:500],
            }

            # RAGAS scores
            if eval_result.ragas_evaluation:
                row["ragas_faithfulness"] = eval_result.ragas_evaluation.faithfulness.score
                row["ragas_answer_relevancy"] = eval_result.ragas_evaluation.answer_relevancy.score
                row["ragas_context_relevancy"] = eval_result.ragas_evaluation.context_relevancy.score

            # DeepEval scores
            if eval_result.deepeval_evaluation:
                row["deepeval_faithfulness"] = eval_result.deepeval_evaluation.faithfulness.score
                row["deepeval_answer_relevancy"] = eval_result.deepeval_evaluation.answer_relevancy.score
                row["deepeval_context_relevancy"] = eval_result.deepeval_evaluation.context_relevancy.score

            rows.append(row)

        # Create DataFrame and save
        df = pd.DataFrame(rows)
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False, encoding='utf-8')

        return str(path.absolute())

    def generate_json(self, report: EvaluationReport, output_path: str) -> str:
        """Generate JSON report (machine-readable)."""
        return write_json(report.model_dump(), output_path)

    def generate_markdown(self, report: EvaluationReport, output_path: str) -> str:
        """Generate Markdown report (human-readable)."""
        lines = []

        # Header
        lines.append("# RAG Evaluation Report")
        lines.append("")
        lines.append(f"**Report ID:** {report.report_id}")
        lines.append(f"**Generated:** {report.generated_at.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"**Total Evaluations:** {report.statistics.total_evaluations}")
        lines.append("")

        # Summary Statistics
        lines.append("## Summary Statistics")
        lines.append("")
        lines.append("### RAGAS Scores")
        lines.append("")
        lines.append("| Metric | Average | Min | Max | Std Dev |")
        lines.append("|--------|---------|-----|-----|---------|")
        for metric in ["faithfulness", "answer_relevancy", "context_relevancy"]:
            avg = report.statistics.ragas_averages.get(metric, 0)
            min_val = report.statistics.ragas_min.get(metric, 0)
            max_val = report.statistics.ragas_max.get(metric, 0)
            std = report.statistics.ragas_std.get(metric, 0)
            lines.append(f"| {metric.replace('_', ' ').title()} | {avg:.3f} | {min_val:.3f} | {max_val:.3f} | {std:.3f} |")
        lines.append("")

        lines.append("### DeepEval Scores")
        lines.append("")
        lines.append("| Metric | Average | Min | Max | Std Dev |")
        lines.append("|--------|---------|-----|-----|---------|")
        for metric in ["faithfulness", "answer_relevancy", "context_relevancy"]:
            avg = report.statistics.deepeval_averages.get(metric, 0)
            min_val = report.statistics.deepeval_min.get(metric, 0)
            max_val = report.statistics.deepeval_max.get(metric, 0)
            std = report.statistics.deepeval_std.get(metric, 0)
            lines.append(f"| {metric.replace('_', ' ').title()} | {avg:.3f} | {min_val:.3f} | {max_val:.3f} | {std:.3f} |")
        lines.append("")

        # Pass Rates
        lines.append("### Pass Rates")
        lines.append("")
        lines.append("| Metric | Pass Rate |")
        lines.append("|--------|-----------|")
        for metric, rate in report.statistics.pass_rates.items():
            lines.append(f"| {metric.replace('_', ' ').title()} | {rate*100:.1f}% |")
        lines.append("")

        # Framework Comparison
        if report.comparison_summary:
            lines.append("## Framework Comparison")
            lines.append("")
            lines.append("### Average Score Differences")
            lines.append("")
            lines.append("| Metric | Avg Difference | Correlation |")
            lines.append("|--------|----------------|-------------|")
            for metric in ["faithfulness", "answer_relevancy", "context_relevancy"]:
                diff = report.comparison_summary.avg_score_difference.get(metric, 0)
                corr = report.comparison_summary.correlation.get(metric, 0)
                lines.append(f"| {metric.replace('_', ' ').title()} | {diff:.3f} | {corr:.3f} |")
            lines.append("")

            # Discrepancies
            if report.comparison_summary.significant_discrepancies:
                lines.append("### Significant Discrepancies")
                lines.append("")
                lines.append("| Question (truncated) | Metric | RAGAS | DeepEval | Diff |")
                lines.append("|---------------------|--------|-------|----------|------|")
                for disc in report.comparison_summary.significant_discrepancies[:10]:
                    q = disc.question[:40] + "..." if len(disc.question) > 40 else disc.question
                    lines.append(f"| {q} | {disc.metric} | {disc.ragas_score:.2f} | {disc.deepeval_score:.2f} | {disc.difference:.2f} |")
                lines.append("")

        # Individual Results (limited)
        lines.append("## Detailed Results")
        lines.append("")
        for i, eval_result in enumerate(report.evaluations[:10]):
            lines.append(f"### Evaluation {i+1}")
            lines.append("")
            lines.append(f"**Question:** {eval_result.question}")
            lines.append("")
            lines.append(f"**Answer:** {eval_result.rag_response.answer[:300]}...")
            lines.append("")
            lines.append("| Framework | Faithfulness | Answer Relevancy | Context Relevancy |")
            lines.append("|-----------|--------------|------------------|-------------------|")

            if eval_result.ragas_evaluation:
                lines.append(f"| RAGAS | {eval_result.ragas_evaluation.faithfulness.score:.2f} | {eval_result.ragas_evaluation.answer_relevancy.score:.2f} | {eval_result.ragas_evaluation.context_relevancy.score:.2f} |")
            if eval_result.deepeval_evaluation:
                lines.append(f"| DeepEval | {eval_result.deepeval_evaluation.faithfulness.score:.2f} | {eval_result.deepeval_evaluation.answer_relevancy.score:.2f} | {eval_result.deepeval_evaluation.context_relevancy.score:.2f} |")
            lines.append("")

        if len(report.evaluations) > 10:
            lines.append(f"*... and {len(report.evaluations) - 10} more evaluations (see CSV for full details)*")

        content = "\n".join(lines)
        return write_text(content, output_path)

    def generate_html(self, report: EvaluationReport, output_path: str) -> str:
        """Generate HTML report with styling."""
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>RAG Evaluation Report - {report.report_id}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }}
        h2 {{ color: #555; margin-top: 30px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
        .score-high {{ color: #4CAF50; font-weight: bold; }}
        .score-medium {{ color: #FF9800; font-weight: bold; }}
        .score-low {{ color: #f44336; font-weight: bold; }}
        .meta {{ color: #666; margin-bottom: 20px; }}
        .summary-box {{ background: #e8f5e9; padding: 15px; border-radius: 5px; margin: 20px 0; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>RAG Evaluation Report</h1>
        <div class="meta">
            <p><strong>Report ID:</strong> {report.report_id}</p>
            <p><strong>Generated:</strong> {report.generated_at.strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p><strong>Total Evaluations:</strong> {report.statistics.total_evaluations}</p>
        </div>

        <div class="summary-box">
            <h3>Quick Summary</h3>
            <p>Average Faithfulness: <span class="{self._get_score_class(report.statistics.ragas_averages.get('faithfulness', 0))}">{report.statistics.ragas_averages.get('faithfulness', 0):.2f}</span> (RAGAS) / <span class="{self._get_score_class(report.statistics.deepeval_averages.get('faithfulness', 0))}">{report.statistics.deepeval_averages.get('faithfulness', 0):.2f}</span> (DeepEval)</p>
            <p>Average Answer Relevancy: <span class="{self._get_score_class(report.statistics.ragas_averages.get('answer_relevancy', 0))}">{report.statistics.ragas_averages.get('answer_relevancy', 0):.2f}</span> (RAGAS) / <span class="{self._get_score_class(report.statistics.deepeval_averages.get('answer_relevancy', 0))}">{report.statistics.deepeval_averages.get('answer_relevancy', 0):.2f}</span> (DeepEval)</p>
            <p>Average Context Relevancy: <span class="{self._get_score_class(report.statistics.ragas_averages.get('context_relevancy', 0))}">{report.statistics.ragas_averages.get('context_relevancy', 0):.2f}</span> (RAGAS) / <span class="{self._get_score_class(report.statistics.deepeval_averages.get('context_relevancy', 0))}">{report.statistics.deepeval_averages.get('context_relevancy', 0):.2f}</span> (DeepEval)</p>
        </div>

        <h2>Pass Rates</h2>
        <table>
            <tr><th>Metric</th><th>Pass Rate</th></tr>
            {"".join(f"<tr><td>{m.replace('_', ' ').title()}</td><td>{r*100:.1f}%</td></tr>" for m, r in report.statistics.pass_rates.items())}
        </table>

        <h2>Detailed Results</h2>
        <table>
            <tr>
                <th>Question</th>
                <th>RAGAS Faith.</th>
                <th>RAGAS Ans.Rel.</th>
                <th>RAGAS Ctx.Rel.</th>
                <th>DeepEval Faith.</th>
                <th>DeepEval Ans.Rel.</th>
                <th>DeepEval Ctx.Rel.</th>
            </tr>
            {"".join(self._format_result_row(r) for r in report.evaluations[:20])}
        </table>
        {f'<p><em>Showing first 20 of {len(report.evaluations)} evaluations</em></p>' if len(report.evaluations) > 20 else ''}
    </div>
</body>
</html>"""

        return write_text(html, output_path)

    def _get_score_class(self, score: float) -> str:
        """Get CSS class based on score."""
        if score >= 0.8:
            return "score-high"
        elif score >= 0.6:
            return "score-medium"
        return "score-low"

    def _format_result_row(self, result) -> str:
        """Format a single result row for HTML table."""
        q = result.question[:50] + "..." if len(result.question) > 50 else result.question

        ragas = result.ragas_evaluation
        deepeval = result.deepeval_evaluation

        r_f = f"{ragas.faithfulness.score:.2f}" if ragas else "-"
        r_a = f"{ragas.answer_relevancy.score:.2f}" if ragas else "-"
        r_c = f"{ragas.context_relevancy.score:.2f}" if ragas else "-"
        d_f = f"{deepeval.faithfulness.score:.2f}" if deepeval else "-"
        d_a = f"{deepeval.answer_relevancy.score:.2f}" if deepeval else "-"
        d_c = f"{deepeval.context_relevancy.score:.2f}" if deepeval else "-"

        return f"<tr><td>{q}</td><td>{r_f}</td><td>{r_a}</td><td>{r_c}</td><td>{d_f}</td><td>{d_a}</td><td>{d_c}</td></tr>"
