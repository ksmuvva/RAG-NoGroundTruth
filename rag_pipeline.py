"""
Simplified RAG Evaluation Pipeline
User provides docs/CSV → RAG API → Auto evaluation → Results CSV
"""
import os
import re
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
import yaml
from dotenv import load_dotenv

load_dotenv()


class RAGPipeline:
    """Single class handling: upload → query → evaluate → output"""

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self._resolve_env_vars()
        self._setup_folders()
        self.api_key = os.getenv(self.config["rag_api"]["api_key_env"], "")

    def _resolve_env_vars(self):
        """Replace ${VAR:default} patterns in config"""
        def resolve(value):
            if isinstance(value, str):
                pattern = r'\$\{(\w+):?([^}]*)\}'
                match = re.search(pattern, value)
                if match:
                    env_var, default = match.groups()
                    return os.getenv(env_var, default)
            return value

        for section in self.config:
            if isinstance(self.config[section], dict):
                for key, val in self.config[section].items():
                    self.config[section][key] = resolve(val)

    def _setup_folders(self):
        """Ensure output folders exist"""
        for folder in self.config["folders"].values():
            Path(folder).mkdir(parents=True, exist_ok=True)

    def upload_documents(self, file_paths: list[str]) -> dict:
        """Upload documents (PDF, DOCX, TXT) to RAG API"""
        url = f"{self.config['rag_api']['base_url']}{self.config['rag_api']['upload_endpoint']}"
        headers = {"X-API-Key": self.api_key}
        results = {"success": [], "failed": []}

        for path in file_paths:
            try:
                with open(path, "rb") as f:
                    files = {"file": (Path(path).name, f)}
                    resp = requests.post(url, headers=headers, files=files,
                                        timeout=self.config["rag_api"]["timeout"])
                    resp.raise_for_status()
                    results["success"].append(path)
                    print(f"[OK] Uploaded: {path}")
            except Exception as e:
                results["failed"].append({"path": path, "error": str(e)})
                print(f"[FAIL] {path}: {e}")

        return results

    def query_rag(self, questions: list[str]) -> pd.DataFrame:
        """Query RAG API, return DataFrame with Questions, Answer, Context"""
        url = f"{self.config['rag_api']['base_url']}{self.config['rag_api']['query_endpoint']}"
        headers = {"X-API-Key": self.api_key, "Content-Type": "application/json"}
        rows = []

        for q in questions:
            try:
                resp = requests.post(url, headers=headers, json={
                                        "question": q,
                                        "include_context": True
                                    },
                                    timeout=self.config["rag_api"]["timeout"])
                resp.raise_for_status()
                data = resp.json()
                rows.append({
                    "Questions": q,
                    "Answer": data.get("answer", ""),
                    "Context": " | ".join(data.get("contexts", []))
                })
            except Exception as e:
                rows.append({"Questions": q, "Answer": f"ERROR: {e}", "Context": ""})
                print(f"[FAIL] Query '{q[:50]}...': {e}")

        return pd.DataFrame(rows)

    def process_csv(self, csv_path: str) -> pd.DataFrame:
        """Load CSV with questions, query RAG, return results"""
        df = pd.read_csv(csv_path)
        question_col = next((c for c in df.columns if "question" in c.lower()), df.columns[0])
        questions = df[question_col].dropna().tolist()
        return self.query_rag(questions)

    def run(self, input_path: str) -> str:
        """
        Main entry point:
        1. If CSV → extract questions and query RAG
        2. If doc (PDF/DOCX) → upload to RAG first
        3. Save results to RAG_output_folder with timestamp
        4. Copy to RAG_evaluator_input
        5. Run evaluation (RAGAS + DeepEval)
        6. Save metrics to evaluation_output_folder
        Returns: path to evaluation results CSV
        """
        input_path = Path(input_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Step 1: Handle input
        if input_path.suffix.lower() == ".csv":
            print(f"\n[1/5] Processing CSV: {input_path}")
            rag_results = self.process_csv(str(input_path))
        else:
            # Upload doc first, then need questions CSV
            print(f"\n[1/5] Uploading document: {input_path}")
            self.upload_documents([str(input_path)])
            print("Document uploaded. Please provide a CSV with questions to query.")
            return ""

        # Step 2: Save to RAG_output_folder
        output_name = f"rag_output_{timestamp}.csv"
        rag_output_path = Path(self.config["folders"]["rag_output"]) / output_name
        rag_results.to_csv(rag_output_path, index=False)
        print(f"[2/5] Saved RAG output: {rag_output_path}")

        # Step 3: Copy to RAG_evaluator_input
        eval_input_path = Path(self.config["folders"]["evaluator_input"]) / output_name
        shutil.copy(rag_output_path, eval_input_path)
        print(f"[3/5] Copied to evaluator input: {eval_input_path}")

        # Step 4: Run evaluation
        print("[4/5] Running RAGAS + DeepEval evaluation...")
        eval_results = self._evaluate(rag_results)

        # Step 5: Save evaluation results
        eval_output_name = f"evaluation_{timestamp}.csv"
        eval_output_path = Path(self.config["folders"]["evaluation_output"]) / eval_output_name
        eval_results.to_csv(eval_output_path, index=False)
        print(f"[5/5] Saved evaluation results: {eval_output_path}")

        self._print_summary(eval_results)
        return str(eval_output_path)

    def _evaluate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run RAGAS and DeepEval metrics (no ground truth needed)"""
        # Import flags
        ragas_available = False
        deepeval_available = False

        try:
            from ragas import evaluate as ragas_eval
            from ragas.metrics import faithfulness, answer_relevancy, context_precision
            from datasets import Dataset
            ragas_available = True
        except ImportError:
            print("[WARN] RAGAS not installed. Skipping RAGAS evaluation.")

        try:
            from deepeval.metrics import (
                FaithfulnessMetric,
                AnswerRelevancyMetric,
                ContextualRelevancyMetric
            )
            from deepeval.test_case import LLMTestCase
            from deepeval.models import AzureOpenAI as DeepEvalAzureOpenAI
            deepeval_available = True
        except ImportError:
            print("[WARN] DeepEval not installed. Skipping DeepEval evaluation.")

        if not ragas_available and not deepeval_available:
            print("[WARN] Neither RAGAS nor DeepEval available. Using fallback evaluation.")
            return self._fallback_evaluate(df)

        # Initialize DeepEval model if available
        deepeval_model = None
        if deepeval_available:
            try:
                deepeval_model = self._create_deepeval_model(DeepEvalAzureOpenAI)
            except Exception as e:
                print(f"[WARN] Failed to initialize DeepEval model: {e}")
                deepeval_available = False

        results = []
        thresholds = self.config["thresholds"]

        for _, row in df.iterrows():
            q, a, c = row["Questions"], row["Answer"], row["Context"]
            contexts = c.split(" | ") if c else []

            scores = {"Questions": q, "Answer": a, "Context": c}

            # RAGAS evaluation
            if ragas_available:
                try:
                    dataset = Dataset.from_dict({
                        "question": [q],
                        "answer": [a],
                        "contexts": [contexts]
                    })
                    ragas_result = ragas_eval(dataset, metrics=[faithfulness, answer_relevancy, context_precision])

                    # FIX: Access scores correctly from EvaluationResult
                    # The evaluate() function returns an EvaluationResult object.
                    # For single-sample evaluation, convert to pandas and get first row values.
                    ragas_df = ragas_result.to_pandas()
                    scores["RAGAS_Faithfulness"] = round(float(ragas_df["faithfulness"].iloc[0]), 3)
                    scores["RAGAS_Answer_Relevancy"] = round(float(ragas_df["answer_relevancy"].iloc[0]), 3)
                    # Note: RAGAS context_precision measures if relevant contexts are ranked higher.
                    # This differs from DeepEval's ContextualRelevancy which measures retrieval relevance.
                    scores["RAGAS_Context_Precision"] = round(float(ragas_df["context_precision"].iloc[0]), 3)
                except Exception as e:
                    print(f"[WARN] RAGAS failed for '{q[:30]}...': {e}")
                    scores["RAGAS_Faithfulness"] = 0.0
                    scores["RAGAS_Answer_Relevancy"] = 0.0
                    scores["RAGAS_Context_Precision"] = 0.0
            else:
                scores["RAGAS_Faithfulness"] = 0.0
                scores["RAGAS_Answer_Relevancy"] = 0.0
                scores["RAGAS_Context_Precision"] = 0.0

            # DeepEval evaluation
            if deepeval_available and deepeval_model:
                try:
                    test_case = LLMTestCase(
                        input=q,
                        actual_output=a,
                        retrieval_context=contexts
                    )

                    # FIX: Pass model to each metric - required by DeepEval
                    faith_metric = FaithfulnessMetric(
                        threshold=thresholds["faithfulness"],
                        model=deepeval_model
                    )
                    relevancy_metric = AnswerRelevancyMetric(
                        threshold=thresholds["answer_relevancy"],
                        model=deepeval_model
                    )
                    # Note: DeepEval ContextualRelevancyMetric measures if retrieved context
                    # is relevant to the input query (different from RAGAS context_precision)
                    context_metric = ContextualRelevancyMetric(
                        threshold=thresholds["context_relevancy"],
                        model=deepeval_model
                    )

                    faith_metric.measure(test_case)
                    relevancy_metric.measure(test_case)
                    context_metric.measure(test_case)

                    scores["DeepEval_Faithfulness"] = round(faith_metric.score, 3)
                    scores["DeepEval_Answer_Relevancy"] = round(relevancy_metric.score, 3)
                    scores["DeepEval_Contextual_Relevancy"] = round(context_metric.score, 3)
                except Exception as e:
                    print(f"[WARN] DeepEval failed for '{q[:30]}...': {e}")
                    scores["DeepEval_Faithfulness"] = 0.0
                    scores["DeepEval_Answer_Relevancy"] = 0.0
                    scores["DeepEval_Contextual_Relevancy"] = 0.0
            else:
                scores["DeepEval_Faithfulness"] = 0.0
                scores["DeepEval_Answer_Relevancy"] = 0.0
                scores["DeepEval_Contextual_Relevancy"] = 0.0

            # Pass/Fail based on thresholds
            scores["Faithfulness_Pass"] = (
                scores.get("RAGAS_Faithfulness", 0) >= thresholds["faithfulness"] or
                scores.get("DeepEval_Faithfulness", 0) >= thresholds["faithfulness"]
            )
            scores["Answer_Relevancy_Pass"] = (
                scores.get("RAGAS_Answer_Relevancy", 0) >= thresholds["answer_relevancy"] or
                scores.get("DeepEval_Answer_Relevancy", 0) >= thresholds["answer_relevancy"]
            )
            # Context relevancy pass uses both metrics (precision from RAGAS, relevancy from DeepEval)
            scores["Context_Relevancy_Pass"] = (
                scores.get("RAGAS_Context_Precision", 0) >= thresholds["context_relevancy"] or
                scores.get("DeepEval_Contextual_Relevancy", 0) >= thresholds["context_relevancy"]
            )

            results.append(scores)

        return pd.DataFrame(results)

    def _create_deepeval_model(self, AzureOpenAIClass):
        """Create DeepEval Azure OpenAI model from config and environment"""
        llm_config = self.config.get("llm", {})

        # Get configuration from environment/config
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")
        api_key = os.getenv("AZURE_OPENAI_API_KEY", "")
        deployment = llm_config.get("deployment", os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4"))
        api_version = llm_config.get("api_version", os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"))

        if not azure_endpoint or not api_key:
            raise ValueError(
                "DeepEval requires AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY environment variables. "
                "Please set these to use DeepEval metrics."
            )

        return AzureOpenAIClass(
            model=deployment,
            deployment_name=deployment,
            azure_openai_api_key=api_key,
            azure_endpoint=azure_endpoint,
            openai_api_version=api_version
        )

    def _fallback_evaluate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Simple fallback when evaluation libraries unavailable"""
        df["RAGAS_Faithfulness"] = 0.0
        df["RAGAS_Answer_Relevancy"] = 0.0
        df["RAGAS_Context_Precision"] = 0.0
        df["DeepEval_Faithfulness"] = 0.0
        df["DeepEval_Answer_Relevancy"] = 0.0
        df["DeepEval_Contextual_Relevancy"] = 0.0
        df["Faithfulness_Pass"] = False
        df["Answer_Relevancy_Pass"] = False
        df["Context_Relevancy_Pass"] = False
        return df

    def _print_summary(self, df: pd.DataFrame):
        """Print evaluation summary"""
        print("\n" + "=" * 50)
        print("EVALUATION SUMMARY")
        print("=" * 50)

        # Define metrics with their RAGAS and DeepEval column names
        # Note: Context metrics have different names due to different measurement approaches
        metrics_config = [
            {
                "name": "Faithfulness",
                "ragas_col": "RAGAS_Faithfulness",
                "deepeval_col": "DeepEval_Faithfulness",
                "pass_col": "Faithfulness_Pass",
                "threshold_key": "faithfulness"
            },
            {
                "name": "Answer Relevancy",
                "ragas_col": "RAGAS_Answer_Relevancy",
                "deepeval_col": "DeepEval_Answer_Relevancy",
                "pass_col": "Answer_Relevancy_Pass",
                "threshold_key": "answer_relevancy"
            },
            {
                "name": "Context Quality",
                "ragas_col": "RAGAS_Context_Precision",  # Measures ranking of relevant contexts
                "deepeval_col": "DeepEval_Contextual_Relevancy",  # Measures relevance of retrieved contexts
                "pass_col": "Context_Relevancy_Pass",
                "threshold_key": "context_relevancy"
            }
        ]

        for metric in metrics_config:
            ragas_col = metric["ragas_col"]
            deep_col = metric["deepeval_col"]
            pass_col = metric["pass_col"]

            ragas_avg = df[ragas_col].mean() if ragas_col in df.columns else 0
            deep_avg = df[deep_col].mean() if deep_col in df.columns else 0
            pass_rate = df[pass_col].mean() * 100 if pass_col in df.columns else 0

            threshold = self.config["thresholds"].get(metric["threshold_key"], 0.8)
            status = "PASS" if (ragas_avg >= threshold or deep_avg >= threshold) else "FAIL"

            print(f"\n{metric['name']}:")
            if "Context" in metric["name"]:
                # Show different labels for context metrics to clarify the difference
                print(f"  RAGAS (Precision):       {ragas_avg:.3f}")
                print(f"  DeepEval (Relevancy):    {deep_avg:.3f}")
            else:
                print(f"  RAGAS:    {ragas_avg:.3f}")
                print(f"  DeepEval: {deep_avg:.3f}")
            print(f"  Pass Rate: {pass_rate:.1f}% [{status}]")

        print("\n" + "=" * 50)


# CLI Interface
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python rag_pipeline.py <input_file>")
        print("  input_file: CSV with 'question' column, or PDF/DOCX to upload")
        print("\nExamples:")
        print("  python rag_pipeline.py questions.csv")
        print("  python rag_pipeline.py document.pdf")
        sys.exit(1)

    pipeline = RAGPipeline()
    result_path = pipeline.run(sys.argv[1])

    if result_path:
        print(f"\nDone! Results saved to: {result_path}")
