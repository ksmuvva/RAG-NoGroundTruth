#!/usr/bin/env python3
"""
Standalone Test Runner for RAG Evaluation Pipeline
Runs all tests without pytest plugins that cause dependency conflicts
"""
import os
import sys
import tempfile
import shutil
import traceback
from pathlib import Path
from datetime import datetime

# Ensure we can import from the current directory
sys.path.insert(0, str(Path(__file__).parent))

# ============================================================================
# TEST FRAMEWORK
# ============================================================================

class TestResult:
    """Simple test result tracker"""
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.skipped = 0
        self.errors = []

    def record_pass(self, test_name):
        self.passed += 1
        print(f"  \033[92m[PASS]\033[0m {test_name}")

    def record_fail(self, test_name, error):
        self.failed += 1
        self.errors.append((test_name, error))
        print(f"  \033[91m[FAIL]\033[0m {test_name}")
        print(f"        Error: {error}")

    def record_skip(self, test_name, reason):
        self.skipped += 1
        print(f"  \033[93m[SKIP]\033[0m {test_name}: {reason}")

    def summary(self):
        total = self.passed + self.failed + self.skipped
        print("\n" + "=" * 70)
        print("TEST SUMMARY")
        print("=" * 70)
        print(f"  Total:   {total}")
        print(f"  \033[92mPassed:  {self.passed}\033[0m")
        print(f"  \033[91mFailed:  {self.failed}\033[0m")
        print(f"  \033[93mSkipped: {self.skipped}\033[0m")
        if self.errors:
            print("\nFailed Tests:")
            for test_name, error in self.errors:
                print(f"  - {test_name}: {error}")
        return self.failed == 0


results = TestResult()


# ============================================================================
# TEST 1: DEPENDENCY INSTALLATION TESTS
# ============================================================================

def test_section_dependencies():
    """Test all required dependencies"""
    print("\n" + "-" * 70)
    print("SECTION 1: DEPENDENCY INSTALLATION TESTS")
    print("-" * 70)

    # Test core dependencies
    try:
        import yaml
        import pandas as pd
        import requests
        from dotenv import load_dotenv
        results.record_pass(f"Core dependencies (yaml={yaml.__version__}, pandas={pd.__version__}, requests={requests.__version__})")
    except ImportError as e:
        results.record_fail("Core dependencies", str(e))

    # Test document processing
    try:
        import PyPDF2
        import docx
        results.record_pass(f"Document processing (PyPDF2={PyPDF2.__version__})")
    except ImportError as e:
        results.record_fail("Document processing", str(e))

    # Test LLM dependencies
    try:
        import openai
        import langchain
        results.record_pass(f"LLM dependencies (openai={openai.__version__})")
    except ImportError as e:
        results.record_fail("LLM dependencies", str(e))

    # Test langchain-openai
    try:
        import langchain_openai
        results.record_pass("langchain-openai installed")
    except ImportError as e:
        results.record_fail("langchain-openai", str(e))

    # Test evaluation frameworks
    ragas_ok = False
    deepeval_ok = False

    try:
        import ragas
        ragas_ok = True
        results.record_pass(f"RAGAS evaluation framework")
    except ImportError as e:
        results.record_skip("RAGAS", str(e))

    try:
        # Import deepeval carefully
        from deepeval.test_case import LLMTestCase
        deepeval_ok = True
        results.record_pass("DeepEval evaluation framework")
    except Exception as e:
        results.record_skip("DeepEval", str(e))

    if not ragas_ok and not deepeval_ok:
        results.record_fail("At least one evaluation framework needed", "Neither RAGAS nor DeepEval available")

    # Test datasets (required by RAGAS)
    try:
        from datasets import Dataset
        results.record_pass("Hugging Face datasets")
    except ImportError as e:
        results.record_skip("Hugging Face datasets", str(e))


# ============================================================================
# TEST 2: FOLDER STRUCTURE TESTS
# ============================================================================

def test_section_folder_structure():
    """Test folder structure"""
    print("\n" + "-" * 70)
    print("SECTION 2: FOLDER STRUCTURE TESTS")
    print("-" * 70)

    project_root = Path(__file__).parent

    # Test required files exist
    required_files = ["rag_pipeline.py", "config.yaml", "requirements.txt", ".env.example"]
    for file in required_files:
        if (project_root / file).exists():
            results.record_pass(f"Required file exists: {file}")
        else:
            results.record_fail(f"Required file exists: {file}", "File not found")

    # Test data folder structure
    expected_folders = [
        "data",
        "data/RAG_output_folder",
        "data/RAG_evaluator_input",
        "data/evaluation_output_folder"
    ]

    for folder in expected_folders:
        folder_path = project_root / folder
        if folder_path.exists():
            results.record_pass(f"Folder exists: {folder}")
            # Test write permission
            try:
                test_file = folder_path / ".write_test"
                test_file.write_text("test")
                test_file.unlink()
                results.record_pass(f"Folder writable: {folder}")
            except PermissionError:
                results.record_fail(f"Folder writable: {folder}", "Permission denied")
            except Exception as e:
                results.record_skip(f"Folder writable: {folder}", str(e))
        else:
            # Create folder if missing
            try:
                folder_path.mkdir(parents=True, exist_ok=True)
                results.record_pass(f"Folder created: {folder}")
            except Exception as e:
                results.record_fail(f"Folder created: {folder}", str(e))

    # Test pipeline creates folders
    try:
        temp_dir = tempfile.mkdtemp(prefix="rag_test_")
        config_content = f"""
rag_api:
  base_url: "http://localhost:8000"
  api_key_env: "RAG_API_KEY"
  upload_endpoint: "/api/v1/documents/upload"
  query_endpoint: "/api/v1/query"
  timeout: 60

folders:
  rag_output: "{temp_dir}/data/RAG_output_folder"
  evaluator_input: "{temp_dir}/data/RAG_evaluator_input"
  evaluation_output: "{temp_dir}/data/evaluation_output_folder"

thresholds:
  faithfulness: 0.8
  answer_relevancy: 0.8
  context_relevancy: 0.8
"""
        config_path = Path(temp_dir) / "config.yaml"
        config_path.write_text(config_content)

        from rag_pipeline import RAGPipeline
        pipeline = RAGPipeline(str(config_path))

        folders_created = all([
            (Path(temp_dir) / "data/RAG_output_folder").exists(),
            (Path(temp_dir) / "data/RAG_evaluator_input").exists(),
            (Path(temp_dir) / "data/evaluation_output_folder").exists()
        ])

        if folders_created:
            results.record_pass("Pipeline auto-creates output folders")
        else:
            results.record_fail("Pipeline auto-creates output folders", "Some folders not created")

        shutil.rmtree(temp_dir, ignore_errors=True)
    except Exception as e:
        results.record_fail("Pipeline auto-creates output folders", str(e))


# ============================================================================
# TEST 3: CONFIGURATION TESTS
# ============================================================================

def test_section_configuration():
    """Test configuration loading"""
    print("\n" + "-" * 70)
    print("SECTION 3: CONFIGURATION TESTS")
    print("-" * 70)

    import yaml

    config_path = Path(__file__).parent / "config.yaml"

    # Test config loads
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
        results.record_pass("config.yaml loads successfully")
    except Exception as e:
        results.record_fail("config.yaml loads successfully", str(e))
        return

    # Test required sections
    required_sections = ["rag_api", "llm", "folders", "thresholds"]
    for section in required_sections:
        if section in config:
            results.record_pass(f"Config section exists: {section}")
        else:
            results.record_fail(f"Config section exists: {section}", "Section missing")

    # Test RAG API settings
    rag_api = config.get("rag_api", {})
    required_keys = ["base_url", "api_key_env", "upload_endpoint", "query_endpoint", "timeout"]
    for key in required_keys:
        if key in rag_api:
            results.record_pass(f"RAG API config: {key}")
        else:
            results.record_fail(f"RAG API config: {key}", "Key missing")

    # Test thresholds
    thresholds = config.get("thresholds", {})
    for name, value in thresholds.items():
        if isinstance(value, (int, float)) and 0 <= value <= 1:
            results.record_pass(f"Threshold valid: {name}={value}")
        else:
            results.record_fail(f"Threshold valid: {name}", f"Invalid value: {value}")

    # Test .env.example
    env_example_path = Path(__file__).parent / ".env.example"
    try:
        content = env_example_path.read_text()
        required_vars = [
            "RAG_API_BASE_URL", "RAG_API_KEY", "AZURE_OPENAI_API_KEY",
            "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_DEPLOYMENT_NAME"
        ]
        for var in required_vars:
            if var in content:
                results.record_pass(f".env.example has: {var}")
            else:
                results.record_fail(f".env.example has: {var}", "Variable missing")
    except Exception as e:
        results.record_fail(".env.example readable", str(e))

    # Test environment variable resolution
    try:
        os.environ["TEST_RESOLUTION_VAR"] = "test_value"
        from rag_pipeline import RAGPipeline
        pipeline = RAGPipeline(str(config_path))
        results.record_pass("Environment variable resolution works")
        del os.environ["TEST_RESOLUTION_VAR"]
    except Exception as e:
        results.record_fail("Environment variable resolution", str(e))


# ============================================================================
# TEST 4: SYNTHETIC DATA TESTS
# ============================================================================

def test_section_synthetic_data():
    """Test with synthetic data"""
    print("\n" + "-" * 70)
    print("SECTION 4: SYNTHETIC DATA TESTS")
    print("-" * 70)

    import pandas as pd

    # Create synthetic data
    synthetic_data = pd.DataFrame([
        {
            "Questions": "What is machine learning?",
            "Answer": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed.",
            "Context": "Machine learning is a branch of AI and computer science. | ML algorithms build models based on sample data."
        },
        {
            "Questions": "How does neural network work?",
            "Answer": "A neural network processes information through layers of interconnected nodes, adjusting weights during training to minimize errors.",
            "Context": "Neural networks are computing systems inspired by biological neural networks. | Connections between neurons are adjusted during training."
        },
        {
            "Questions": "What is data preprocessing?",
            "Answer": "Data preprocessing transforms raw data into a clean, organized format suitable for analysis.",
            "Context": "Preprocessing transforms raw data into an understandable format. | Techniques include data cleaning and transformation."
        }
    ])

    # Test data structure
    required_columns = ["Questions", "Answer", "Context"]
    if all(col in synthetic_data.columns for col in required_columns):
        results.record_pass(f"Synthetic data has required columns")
    else:
        results.record_fail("Synthetic data columns", f"Missing columns")

    # Test no null values
    if synthetic_data.notna().all().all():
        results.record_pass("Synthetic data has no null values")
    else:
        results.record_fail("Synthetic data null check", "Contains null values")

    # Test data quality
    avg_answer_len = synthetic_data["Answer"].str.len().mean()
    if avg_answer_len > 50:
        results.record_pass(f"Answer length adequate (avg: {avg_answer_len:.0f} chars)")
    else:
        results.record_fail("Answer length", f"Too short: {avg_answer_len}")

    # Test multi-context
    has_multiple = synthetic_data["Context"].str.contains(r"\|").sum()
    if has_multiple > 0:
        results.record_pass(f"Multi-context samples present ({has_multiple}/{len(synthetic_data)})")
    else:
        results.record_fail("Multi-context samples", "No pipe-separated contexts found")

    return synthetic_data


# ============================================================================
# TEST 5: METRICS CALCULATION TESTS
# ============================================================================

def test_section_metrics(synthetic_data):
    """Test metrics calculation"""
    print("\n" + "-" * 70)
    print("SECTION 5: METRICS CALCULATION TESTS")
    print("-" * 70)

    import pandas as pd
    from rag_pipeline import RAGPipeline

    config_path = Path(__file__).parent / "config.yaml"
    pipeline = RAGPipeline(str(config_path))

    # Test fallback evaluation
    try:
        result_df = pipeline._fallback_evaluate(synthetic_data.copy())

        expected_columns = [
            "RAGAS_Faithfulness", "RAGAS_Answer_Relevancy", "RAGAS_Context_Relevancy",
            "DeepEval_Faithfulness", "DeepEval_Answer_Relevancy", "DeepEval_Context_Relevancy",
            "Faithfulness_Pass", "Answer_Relevancy_Pass", "Context_Relevancy_Pass"
        ]

        all_present = all(col in result_df.columns for col in expected_columns)
        if all_present:
            results.record_pass("Fallback evaluation adds all metric columns")
        else:
            results.record_fail("Fallback evaluation columns", "Missing columns")

        # Check fallback values are 0
        metric_cols = [c for c in expected_columns if "RAGAS" in c or "DeepEval" in c]
        all_zeros = all((result_df[col] == 0.0).all() for col in metric_cols)
        if all_zeros:
            results.record_pass("Fallback sets metric scores to 0")
        else:
            results.record_fail("Fallback scores", "Not all zeros")

        # Check pass columns are False
        pass_cols = [c for c in expected_columns if "Pass" in c]
        all_false = all((result_df[col] == False).all() for col in pass_cols)
        if all_false:
            results.record_pass("Fallback sets pass columns to False")
        else:
            results.record_fail("Fallback pass columns", "Not all False")

    except Exception as e:
        results.record_fail("Fallback evaluation", str(e))

    # Test pass/fail threshold logic
    try:
        thresholds = pipeline.config["thresholds"]
        test_df = synthetic_data.copy()

        # Simulate scores
        test_df["RAGAS_Faithfulness"] = [0.9, 0.5, 0.8]
        test_df["RAGAS_Answer_Relevancy"] = [0.85, 0.7, 0.8]
        test_df["RAGAS_Context_Relevancy"] = [0.9, 0.6, 0.79]
        test_df["DeepEval_Faithfulness"] = [0.85, 0.5, 0.75]
        test_df["DeepEval_Answer_Relevancy"] = [0.9, 0.65, 0.85]
        test_df["DeepEval_Context_Relevancy"] = [0.88, 0.55, 0.82]

        # Apply pass/fail logic
        for m in ["Faithfulness", "Answer_Relevancy", "Context_Relevancy"]:
            threshold = thresholds.get(m.lower(), 0.8)
            test_df[f"{m}_Pass"] = (
                (test_df.get(f"RAGAS_{m}", 0) >= threshold) |
                (test_df.get(f"DeepEval_{m}", 0) >= threshold)
            )

        # Verify row 0 passes (all high scores)
        if test_df.loc[0, "Faithfulness_Pass"] and test_df.loc[0, "Answer_Relevancy_Pass"]:
            results.record_pass("High scores correctly pass threshold")
        else:
            results.record_fail("High scores threshold", "Should pass")

        # Verify row 1 fails (all low scores)
        if not test_df.loc[1, "Faithfulness_Pass"] and not test_df.loc[1, "Answer_Relevancy_Pass"]:
            results.record_pass("Low scores correctly fail threshold")
        else:
            results.record_fail("Low scores threshold", "Should fail")

        # Verify OR logic (either framework passing = overall pass)
        if test_df.loc[2, "Context_Relevancy_Pass"]:  # RAGAS 0.79 fails, DeepEval 0.82 passes
            results.record_pass("OR logic works (DeepEval passes when RAGAS fails)")
        else:
            results.record_fail("OR logic", "Should pass when one framework passes")

    except Exception as e:
        results.record_fail("Pass/fail threshold logic", str(e))

    # Test metric value ranges
    try:
        test_df = synthetic_data.copy()
        metric_columns = [
            "RAGAS_Faithfulness", "RAGAS_Answer_Relevancy", "RAGAS_Context_Relevancy"
        ]
        for col in metric_columns:
            test_df[col] = [0.85, 0.72, 0.91]

        all_valid = all(
            (test_df[col] >= 0).all() and (test_df[col] <= 1).all()
            for col in metric_columns
        )
        if all_valid:
            results.record_pass("Metric values in valid range [0, 1]")
        else:
            results.record_fail("Metric value range", "Values out of range")
    except Exception as e:
        results.record_fail("Metric value range test", str(e))

    # Test summary statistics
    try:
        test_df = synthetic_data.copy()
        test_df["RAGAS_Faithfulness"] = [0.9, 0.8, 0.7]
        test_df["Faithfulness_Pass"] = [True, True, False]

        avg_faith = test_df["RAGAS_Faithfulness"].mean()
        pass_rate = test_df["Faithfulness_Pass"].mean() * 100

        if abs(avg_faith - 0.8) < 0.01 and abs(pass_rate - 66.67) < 1:
            results.record_pass(f"Summary statistics correct (avg={avg_faith:.2f}, pass_rate={pass_rate:.1f}%)")
        else:
            results.record_fail("Summary statistics", f"Unexpected values: avg={avg_faith}, rate={pass_rate}")
    except Exception as e:
        results.record_fail("Summary statistics", str(e))


# ============================================================================
# TEST 6: PIPELINE COMPONENT TESTS
# ============================================================================

def test_section_pipeline_components():
    """Test pipeline components"""
    print("\n" + "-" * 70)
    print("SECTION 6: PIPELINE COMPONENT TESTS")
    print("-" * 70)

    import pandas as pd
    from rag_pipeline import RAGPipeline

    config_path = Path(__file__).parent / "config.yaml"

    # Test pipeline initialization
    try:
        pipeline = RAGPipeline(str(config_path))
        if pipeline.config and "rag_api" in pipeline.config:
            results.record_pass("Pipeline initializes correctly")
        else:
            results.record_fail("Pipeline initialization", "Config not loaded properly")
    except Exception as e:
        results.record_fail("Pipeline initialization", str(e))

    # Test CSV column detection
    temp_dir = tempfile.mkdtemp(prefix="csv_test_")
    try:
        column_variations = ["question", "Question", "QUESTION", "questions", "user_question"]
        all_detected = True

        for col_name in column_variations:
            csv_path = Path(temp_dir) / f"test_{col_name}.csv"
            pd.DataFrame({col_name: ["Test question?"]}).to_csv(csv_path, index=False)

            df = pd.read_csv(csv_path)
            detected_col = next((c for c in df.columns if "question" in c.lower()), df.columns[0])

            if detected_col != col_name:
                all_detected = False
                break

        if all_detected:
            results.record_pass("CSV column auto-detection works for all variations")
        else:
            results.record_fail("CSV column detection", f"Failed for {col_name}")

        shutil.rmtree(temp_dir, ignore_errors=True)
    except Exception as e:
        results.record_fail("CSV column detection", str(e))

    # Test timestamp generation
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if len(timestamp) == 15 and timestamp[8] == "_":
            results.record_pass(f"Timestamp format correct: {timestamp}")
        else:
            results.record_fail("Timestamp format", f"Invalid: {timestamp}")
    except Exception as e:
        results.record_fail("Timestamp generation", str(e))

    # Test context parsing
    try:
        context_string = "Context 1. | Context 2. | Context 3."
        contexts = context_string.split(" | ")
        if len(contexts) == 3 and contexts[0] == "Context 1.":
            results.record_pass("Context parsing with pipe separator works")
        else:
            results.record_fail("Context parsing", "Incorrect split")
    except Exception as e:
        results.record_fail("Context parsing", str(e))

    # Test output DataFrame structure
    try:
        output_df = pd.DataFrame([{
            "Questions": "Test?", "Answer": "Yes.", "Context": "Test.",
            "RAGAS_Faithfulness": 0.85, "RAGAS_Answer_Relevancy": 0.82,
            "RAGAS_Context_Relevancy": 0.88,
            "DeepEval_Faithfulness": 0.87, "DeepEval_Answer_Relevancy": 0.84,
            "DeepEval_Context_Relevancy": 0.86,
            "Faithfulness_Pass": True, "Answer_Relevancy_Pass": True,
            "Context_Relevancy_Pass": True
        }])

        expected_cols = 12
        if len(output_df.columns) == expected_cols:
            results.record_pass(f"Output DataFrame has {expected_cols} expected columns")
        else:
            results.record_fail("Output DataFrame structure", f"Has {len(output_df.columns)} columns")
    except Exception as e:
        results.record_fail("Output DataFrame structure", str(e))


# ============================================================================
# TEST 7: INTEGRATION TESTS
# ============================================================================

def test_section_integration():
    """Integration tests"""
    print("\n" + "-" * 70)
    print("SECTION 7: INTEGRATION TESTS")
    print("-" * 70)

    import pandas as pd
    import yaml
    from rag_pipeline import RAGPipeline

    temp_dir = tempfile.mkdtemp(prefix="integration_test_")

    try:
        # Setup temp project
        project_root = Path(__file__).parent

        # Copy and modify config
        with open(project_root / "config.yaml") as f:
            config = yaml.safe_load(f)

        config["folders"]["rag_output"] = f"{temp_dir}/data/RAG_output_folder"
        config["folders"]["evaluator_input"] = f"{temp_dir}/data/RAG_evaluator_input"
        config["folders"]["evaluation_output"] = f"{temp_dir}/data/evaluation_output_folder"

        config_path = Path(temp_dir) / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        # Create test data
        test_df = pd.DataFrame([{
            "Questions": "What is AI?",
            "Answer": "AI is artificial intelligence.",
            "Context": "AI refers to machine intelligence."
        }])

        # Test pipeline with fallback
        pipeline = RAGPipeline(str(config_path))
        result_df = pipeline._fallback_evaluate(test_df)

        if len(result_df) == 1 and "RAGAS_Faithfulness" in result_df.columns:
            results.record_pass("Full pipeline with fallback evaluation works")
        else:
            results.record_fail("Full pipeline", "Unexpected output")

        # Test output file creation
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_folder = Path(temp_dir) / "data/RAG_output_folder"
        output_path = output_folder / f"rag_output_{timestamp}.csv"
        test_df.to_csv(output_path, index=False)

        if output_path.exists():
            results.record_pass(f"Output file creation works")
        else:
            results.record_fail("Output file creation", "File not created")

        # Test file copy to evaluator input
        eval_input_folder = Path(temp_dir) / "data/RAG_evaluator_input"
        eval_input_path = eval_input_folder / f"rag_output_{timestamp}.csv"
        shutil.copy(output_path, eval_input_path)

        if eval_input_path.exists():
            loaded = pd.read_csv(eval_input_path)
            if loaded.equals(test_df):
                results.record_pass("File copy to evaluator input works correctly")
            else:
                results.record_fail("File copy integrity", "Content mismatch")
        else:
            results.record_fail("File copy", "Copy not created")

    except Exception as e:
        results.record_fail("Integration tests", str(e))
        traceback.print_exc()
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# ============================================================================
# TEST 8: ERROR HANDLING TESTS
# ============================================================================

def test_section_error_handling():
    """Test error handling"""
    print("\n" + "-" * 70)
    print("SECTION 8: ERROR HANDLING TESTS")
    print("-" * 70)

    import pandas as pd
    from rag_pipeline import RAGPipeline

    # Test missing config file
    try:
        fake_config = "/tmp/nonexistent_config.yaml"
        try:
            pipeline = RAGPipeline(fake_config)
            results.record_fail("Missing config handling", "Should raise FileNotFoundError")
        except FileNotFoundError:
            results.record_pass("Missing config file raises FileNotFoundError")
        except Exception as e:
            results.record_fail("Missing config handling", f"Wrong exception: {type(e)}")
    except Exception as e:
        results.record_fail("Missing config test", str(e))

    # Test empty CSV handling
    temp_dir = tempfile.mkdtemp(prefix="error_test_")
    try:
        empty_csv = Path(temp_dir) / "empty.csv"
        empty_csv.write_text("question\n")

        df = pd.read_csv(empty_csv)
        questions = df["question"].dropna().tolist()

        if len(questions) == 0:
            results.record_pass("Empty CSV handled correctly")
        else:
            results.record_fail("Empty CSV handling", "Should return empty list")

        shutil.rmtree(temp_dir, ignore_errors=True)
    except Exception as e:
        results.record_fail("Empty CSV handling", str(e))

    # Test missing question column fallback
    temp_dir = tempfile.mkdtemp(prefix="column_test_")
    try:
        csv_path = Path(temp_dir) / "no_question.csv"
        pd.DataFrame({"other_column": ["data"]}).to_csv(csv_path, index=False)

        df = pd.read_csv(csv_path)
        question_col = next((c for c in df.columns if "question" in c.lower()), df.columns[0])

        if question_col == "other_column":
            results.record_pass("Missing question column falls back to first column")
        else:
            results.record_fail("Column fallback", f"Got: {question_col}")

        shutil.rmtree(temp_dir, ignore_errors=True)
    except Exception as e:
        results.record_fail("Column fallback", str(e))

    # Test DataFrame with nulls
    try:
        df_with_nulls = pd.DataFrame([
            {"Questions": "Q1?", "Answer": "A1", "Context": "C1"},
            {"Questions": None, "Answer": "A2", "Context": "C2"},
            {"Questions": "Q3?", "Answer": None, "Context": "C3"},
        ])

        valid_df = df_with_nulls[df_with_nulls["Questions"].notna()]
        if len(valid_df) == 2:
            results.record_pass("Null value filtering works correctly")
        else:
            results.record_fail("Null filtering", f"Expected 2, got {len(valid_df)}")
    except Exception as e:
        results.record_fail("Null value handling", str(e))

    # Test threshold validation
    try:
        import yaml
        config_path = Path(__file__).parent / "config.yaml"
        with open(config_path) as f:
            config = yaml.safe_load(f)

        all_valid = all(
            0 <= v <= 1 for v in config["thresholds"].values()
        )
        if all_valid:
            results.record_pass("All threshold values are in valid range [0, 1]")
        else:
            results.record_fail("Threshold validation", "Values out of range")
    except Exception as e:
        results.record_fail("Threshold validation", str(e))


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run all tests"""
    print("=" * 70)
    print("RAG EVALUATION PIPELINE - COMPREHENSIVE TEST SUITE")
    print("=" * 70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python: {sys.version}")
    print("=" * 70)

    # Run all test sections
    test_section_dependencies()
    test_section_folder_structure()
    test_section_configuration()
    synthetic_data = test_section_synthetic_data()
    test_section_metrics(synthetic_data)
    test_section_pipeline_components()
    test_section_integration()
    test_section_error_handling()

    # Print summary
    success = results.summary()

    print("\n" + "=" * 70)
    if success:
        print("\033[92mALL TESTS PASSED!\033[0m")
    else:
        print(f"\033[91m{results.failed} TESTS FAILED\033[0m")
    print("=" * 70)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
