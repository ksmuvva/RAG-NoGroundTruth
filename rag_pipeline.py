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
        try:
            from ragas import evaluate as ragas_eval
            from ragas.metrics import faithfulness, answer_relevancy, context_precision
            from datasets import Dataset
        except ImportError:
            print("[WARN] RAGAS not installed. Using fallback evaluation.")
            return self._fallback_evaluate(df)

        try:
            from deepeval.metrics import (
                FaithfulnessMetric,
                AnswerRelevancyMetric,
                ContextualRelevancyMetric
            )
            from deepeval.test_case import LLMTestCase
        except ImportError:
            print("[WARN] DeepEval not installed. Using RAGAS only.")

        results = []
        thresholds = self.config["thresholds"]

        for _, row in df.iterrows():
            q, a, c = row["Questions"], row["Answer"], row["Context"]
            contexts = c.split(" | ") if c else []

            scores = {"Questions": q, "Answer": a, "Context": c}

            # RAGAS evaluation
            try:
                dataset = Dataset.from_dict({
                    "question": [q],
                    "answer": [a],
                    "contexts": [contexts]
                })
                ragas_result = ragas_eval(dataset, metrics=[faithfulness, answer_relevancy, context_precision])
                scores["RAGAS_Faithfulness"] = round(ragas_result["faithfulness"], 3)
                scores["RAGAS_Answer_Relevancy"] = round(ragas_result["answer_relevancy"], 3)
                scores["RAGAS_Context_Relevancy"] = round(ragas_result["context_precision"], 3)
            except Exception as e:
                print(f"[WARN] RAGAS failed for '{q[:30]}...': {e}")
                scores["RAGAS_Faithfulness"] = 0.0
                scores["RAGAS_Answer_Relevancy"] = 0.0
                scores["RAGAS_Context_Relevancy"] = 0.0

            # DeepEval evaluation
            try:
                test_case = LLMTestCase(
                    input=q,
                    actual_output=a,
                    retrieval_context=contexts
                )

                faith_metric = FaithfulnessMetric(threshold=thresholds["faithfulness"])
                relevancy_metric = AnswerRelevancyMetric(threshold=thresholds["answer_relevancy"])
                context_metric = ContextualRelevancyMetric(threshold=thresholds["context_relevancy"])

                faith_metric.measure(test_case)
                relevancy_metric.measure(test_case)
                context_metric.measure(test_case)

                scores["DeepEval_Faithfulness"] = round(faith_metric.score, 3)
                scores["DeepEval_Answer_Relevancy"] = round(relevancy_metric.score, 3)
                scores["DeepEval_Context_Relevancy"] = round(context_metric.score, 3)
            except Exception as e:
                print(f"[WARN] DeepEval failed for '{q[:30]}...': {e}")
                scores["DeepEval_Faithfulness"] = 0.0
                scores["DeepEval_Answer_Relevancy"] = 0.0
                scores["DeepEval_Context_Relevancy"] = 0.0

            # Pass/Fail based on thresholds
            scores["Faithfulness_Pass"] = (scores.get("RAGAS_Faithfulness", 0) >= thresholds["faithfulness"] or
                                           scores.get("DeepEval_Faithfulness", 0) >= thresholds["faithfulness"])
            scores["Answer_Relevancy_Pass"] = (scores.get("RAGAS_Answer_Relevancy", 0) >= thresholds["answer_relevancy"] or
                                               scores.get("DeepEval_Answer_Relevancy", 0) >= thresholds["answer_relevancy"])
            scores["Context_Relevancy_Pass"] = (scores.get("RAGAS_Context_Relevancy", 0) >= thresholds["context_relevancy"] or
                                                scores.get("DeepEval_Context_Relevancy", 0) >= thresholds["context_relevancy"])

            results.append(scores)

        return pd.DataFrame(results)

    def _fallback_evaluate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Simple fallback when evaluation libraries unavailable"""
        df["RAGAS_Faithfulness"] = 0.0
        df["RAGAS_Answer_Relevancy"] = 0.0
        df["RAGAS_Context_Relevancy"] = 0.0
        df["DeepEval_Faithfulness"] = 0.0
        df["DeepEval_Answer_Relevancy"] = 0.0
        df["DeepEval_Context_Relevancy"] = 0.0
        df["Faithfulness_Pass"] = False
        df["Answer_Relevancy_Pass"] = False
        df["Context_Relevancy_Pass"] = False
        return df

    def _print_summary(self, df: pd.DataFrame):
        """Print evaluation summary"""
        print("\n" + "=" * 50)
        print("EVALUATION SUMMARY")
        print("=" * 50)

        metrics = ["Faithfulness", "Answer_Relevancy", "Context_Relevancy"]
        for m in metrics:
            ragas_col = f"RAGAS_{m}"
            deep_col = f"DeepEval_{m}"
            pass_col = f"{m}_Pass"

            if ragas_col in df.columns:
                ragas_avg = df[ragas_col].mean()
                deep_avg = df[deep_col].mean() if deep_col in df.columns else 0
                pass_rate = df[pass_col].mean() * 100 if pass_col in df.columns else 0

                threshold = self.config["thresholds"].get(m.lower().replace("_", "_"), 0.8)
                status = "PASS" if (ragas_avg >= threshold or deep_avg >= threshold) else "FAIL"

                print(f"\n{m.replace('_', ' ')}:")
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
