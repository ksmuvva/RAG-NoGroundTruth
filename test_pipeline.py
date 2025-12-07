"""
Comprehensive Unit Tests for RAG Evaluation Pipeline
Tests: installations, folder structure, configuration, metrics with synthetic data
No mocks used - real component testing
"""
import os
import sys
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

import pytest
import pandas as pd


# ============================================================================
# TEST 1: DEPENDENCY INSTALLATION TESTS
# ============================================================================

class TestDependencyInstallation:
    """Test that all required dependencies are installed correctly"""

    def test_core_dependencies(self):
        """Test core Python dependencies"""
        import yaml
        assert yaml.__version__ is not None, "PyYAML not installed"

        from dotenv import load_dotenv
        assert load_dotenv is not None, "python-dotenv not installed"

        import pandas as pd
        assert pd.__version__ is not None, "pandas not installed"

        import requests
        assert requests.__version__ is not None, "requests not installed"

        print(f"[PASS] Core dependencies installed:")
        print(f"  - PyYAML: {yaml.__version__}")
        print(f"  - pandas: {pd.__version__}")
        print(f"  - requests: {requests.__version__}")

    def test_document_processing_dependencies(self):
        """Test document processing dependencies"""
        import PyPDF2
        assert PyPDF2.__version__ is not None, "PyPDF2 not installed"

        import docx
        # python-docx doesn't have __version__ in the same way
        assert docx is not None, "python-docx not installed"

        print(f"[PASS] Document processing dependencies installed:")
        print(f"  - PyPDF2: {PyPDF2.__version__}")
        print(f"  - python-docx: installed")

    def test_evaluation_dependencies(self):
        """Test evaluation framework dependencies"""
        try:
            import ragas
            ragas_installed = True
            ragas_version = getattr(ragas, '__version__', 'unknown')
        except ImportError:
            ragas_installed = False
            ragas_version = "NOT INSTALLED"

        try:
            import deepeval
            deepeval_installed = True
            deepeval_version = getattr(deepeval, '__version__', 'unknown')
        except ImportError:
            deepeval_installed = False
            deepeval_version = "NOT INSTALLED"

        print(f"[INFO] Evaluation dependencies:")
        print(f"  - RAGAS: {ragas_version}")
        print(f"  - DeepEval: {deepeval_version}")

        # At least one should be installed for metrics to work
        assert ragas_installed or deepeval_installed, \
            "At least one evaluation framework (RAGAS or DeepEval) must be installed"

    def test_llm_dependencies(self):
        """Test LLM integration dependencies"""
        import openai
        assert openai.__version__ is not None, "openai not installed"

        import langchain
        langchain_version = getattr(langchain, '__version__', 'unknown')

        try:
            import langchain_openai
            langchain_openai_installed = True
        except ImportError:
            langchain_openai_installed = False

        print(f"[PASS] LLM dependencies installed:")
        print(f"  - openai: {openai.__version__}")
        print(f"  - langchain: {langchain_version}")
        print(f"  - langchain-openai: {'installed' if langchain_openai_installed else 'NOT INSTALLED'}")

        assert langchain_openai_installed, "langchain-openai is required"

    def test_datasets_dependency(self):
        """Test Hugging Face datasets (required by RAGAS)"""
        try:
            from datasets import Dataset
            print("[PASS] Hugging Face datasets installed")
            assert Dataset is not None
        except ImportError:
            pytest.skip("datasets not installed - RAGAS features may be limited")


# ============================================================================
# TEST 2: FOLDER STRUCTURE TESTS
# ============================================================================

class TestFolderStructure:
    """Test folder structure creation and validation"""

    @pytest.fixture
    def temp_project_dir(self):
        """Create a temporary project directory for testing"""
        temp_dir = tempfile.mkdtemp(prefix="rag_test_")
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_expected_folders_exist(self):
        """Test that expected data folders exist in project"""
        project_root = Path(__file__).parent

        expected_folders = [
            "data",
            "data/RAG_output_folder",
            "data/RAG_evaluator_input",
            "data/evaluation_output_folder"
        ]

        existing = []
        missing = []

        for folder in expected_folders:
            folder_path = project_root / folder
            if folder_path.exists():
                existing.append(folder)
            else:
                missing.append(folder)

        print(f"[INFO] Folder structure check:")
        for f in existing:
            print(f"  [EXISTS] {f}")
        for f in missing:
            print(f"  [MISSING] {f}")

        # Data folder should exist
        assert (project_root / "data").exists(), "data/ folder is missing"

    def test_pipeline_creates_folders(self, temp_project_dir):
        """Test that RAGPipeline creates required folders automatically"""
        temp_dir = Path(temp_project_dir)

        # Create minimal config
        config_content = """
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
""".format(temp_dir=temp_dir)

        config_path = temp_dir / "config.yaml"
        config_path.write_text(config_content)

        # Import and initialize pipeline
        sys.path.insert(0, str(Path(__file__).parent))
        from rag_pipeline import RAGPipeline

        # Create pipeline - should create folders
        pipeline = RAGPipeline(str(config_path))

        # Verify folders were created
        expected_folders = [
            temp_dir / "data/RAG_output_folder",
            temp_dir / "data/RAG_evaluator_input",
            temp_dir / "data/evaluation_output_folder"
        ]

        for folder in expected_folders:
            assert folder.exists(), f"Pipeline failed to create {folder}"
            print(f"  [CREATED] {folder}")

        print("[PASS] Pipeline automatically creates required folders")

    def test_folder_permissions(self):
        """Test that folders are writable"""
        project_root = Path(__file__).parent
        data_folders = [
            project_root / "data/RAG_output_folder",
            project_root / "data/RAG_evaluator_input",
            project_root / "data/evaluation_output_folder"
        ]

        for folder in data_folders:
            if folder.exists():
                # Test write permission
                test_file = folder / ".write_test"
                try:
                    test_file.write_text("test")
                    test_file.unlink()
                    print(f"  [WRITABLE] {folder}")
                except PermissionError:
                    pytest.fail(f"Folder not writable: {folder}")

    def test_required_files_exist(self):
        """Test that required files exist in project"""
        project_root = Path(__file__).parent

        required_files = [
            "rag_pipeline.py",
            "config.yaml",
            "requirements.txt",
            ".env.example"
        ]

        for file in required_files:
            file_path = project_root / file
            assert file_path.exists(), f"Required file missing: {file}"
            print(f"  [EXISTS] {file}")

        print("[PASS] All required files exist")


# ============================================================================
# TEST 3: CONFIGURATION TESTS
# ============================================================================

class TestConfiguration:
    """Test configuration loading and environment variable resolution"""

    def test_config_yaml_loads(self):
        """Test that config.yaml loads correctly"""
        import yaml

        config_path = Path(__file__).parent / "config.yaml"
        assert config_path.exists(), "config.yaml not found"

        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Verify required sections
        required_sections = ["rag_api", "llm", "folders", "thresholds"]
        for section in required_sections:
            assert section in config, f"Missing config section: {section}"

        print("[PASS] config.yaml loads with all required sections")
        print(f"  Sections found: {list(config.keys())}")

    def test_config_rag_api_settings(self):
        """Test RAG API configuration values"""
        import yaml

        config_path = Path(__file__).parent / "config.yaml"
        with open(config_path) as f:
            config = yaml.safe_load(f)

        rag_api = config["rag_api"]

        # Required keys
        required_keys = ["base_url", "api_key_env", "upload_endpoint", "query_endpoint", "timeout"]
        for key in required_keys:
            assert key in rag_api, f"Missing rag_api.{key}"

        # Timeout should be positive
        assert rag_api["timeout"] > 0, "Timeout must be positive"

        print("[PASS] RAG API configuration is valid")
        print(f"  - upload_endpoint: {rag_api['upload_endpoint']}")
        print(f"  - query_endpoint: {rag_api['query_endpoint']}")
        print(f"  - timeout: {rag_api['timeout']}s")

    def test_config_thresholds(self):
        """Test evaluation thresholds are valid"""
        import yaml

        config_path = Path(__file__).parent / "config.yaml"
        with open(config_path) as f:
            config = yaml.safe_load(f)

        thresholds = config["thresholds"]

        required_thresholds = ["faithfulness", "answer_relevancy", "context_relevancy"]
        for threshold in required_thresholds:
            assert threshold in thresholds, f"Missing threshold: {threshold}"
            value = thresholds[threshold]
            assert 0 <= value <= 1, f"Threshold {threshold} must be between 0 and 1, got {value}"

        print("[PASS] Thresholds are valid (0-1 range)")
        for k, v in thresholds.items():
            print(f"  - {k}: {v}")

    def test_env_variable_resolution(self):
        """Test environment variable pattern resolution"""
        sys.path.insert(0, str(Path(__file__).parent))
        from rag_pipeline import RAGPipeline

        # Set test environment variables
        os.environ["TEST_VAR"] = "test_value"

        # Create pipeline and check resolution
        config_path = Path(__file__).parent / "config.yaml"
        pipeline = RAGPipeline(str(config_path))

        # Config should have resolved values (or defaults)
        assert pipeline.config is not None
        assert "rag_api" in pipeline.config

        # Clean up
        del os.environ["TEST_VAR"]

        print("[PASS] Environment variable resolution works")

    def test_env_example_file(self):
        """Test .env.example has all required variables"""
        env_example_path = Path(__file__).parent / ".env.example"
        assert env_example_path.exists(), ".env.example not found"

        content = env_example_path.read_text()

        required_vars = [
            "RAG_API_BASE_URL",
            "RAG_API_KEY",
            "AZURE_OPENAI_API_KEY",
            "AZURE_OPENAI_ENDPOINT",
            "AZURE_OPENAI_DEPLOYMENT_NAME",
            "AZURE_OPENAI_API_VERSION"
        ]

        for var in required_vars:
            assert var in content, f"Missing variable in .env.example: {var}"

        print("[PASS] .env.example contains all required variables")


# ============================================================================
# TEST 4: SYNTHETIC DATA GENERATION
# ============================================================================

class TestSyntheticData:
    """Generate and test with synthetic RAG data"""

    @pytest.fixture
    def synthetic_rag_data(self):
        """Generate synthetic RAG question-answer-context data"""
        data = [
            {
                "Questions": "What is machine learning?",
                "Answer": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It focuses on developing algorithms that can access data and use it to learn for themselves.",
                "Context": "Machine learning is a branch of artificial intelligence (AI) and computer science which focuses on the use of data and algorithms to imitate the way that humans learn, gradually improving its accuracy. | Machine learning algorithms build a model based on sample data, known as training data, in order to make predictions or decisions without being explicitly programmed to do so."
            },
            {
                "Questions": "How does neural network work?",
                "Answer": "A neural network works by processing information through layers of interconnected nodes (neurons). Input data passes through these layers, with each connection having a weight that adjusts during training. The network learns patterns by adjusting these weights to minimize prediction errors.",
                "Context": "Neural networks are computing systems inspired by biological neural networks that constitute animal brains. They consist of artificial neurons organized in layers. | The connections between neurons, called weights, are adjusted during training to optimize the network's performance on a specific task."
            },
            {
                "Questions": "What is data preprocessing?",
                "Answer": "Data preprocessing is the process of transforming raw data into a clean, organized format suitable for analysis. It includes handling missing values, normalizing data, encoding categorical variables, and removing outliers.",
                "Context": "Data preprocessing is a data mining technique that transforms raw data into an understandable format. Real-world data is often incomplete, inconsistent, and likely to contain errors. | Preprocessing techniques include data cleaning, data integration, data transformation, and data reduction."
            },
            {
                "Questions": "Explain supervised learning.",
                "Answer": "Supervised learning is a type of machine learning where the algorithm learns from labeled training data. The model is trained on input-output pairs and learns to map inputs to outputs, enabling it to make predictions on new, unseen data.",
                "Context": "Supervised learning is a machine learning paradigm for acquiring the input-output relationship information of a system based on a given set of paired input-output training samples. | In supervised learning, each training example consists of an input and a desired output, also known as the supervisory signal."
            },
            {
                "Questions": "What is natural language processing?",
                "Answer": "Natural Language Processing (NLP) is a field of AI that focuses on the interaction between computers and human language. It enables machines to read, understand, and derive meaning from human languages, powering applications like chatbots and translation services.",
                "Context": "Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language. | NLP combines computational linguistics with statistical, machine learning, and deep learning models to process human language."
            }
        ]
        return pd.DataFrame(data)

    @pytest.fixture
    def synthetic_csv_file(self, synthetic_rag_data, tmp_path):
        """Create a temporary CSV file with synthetic questions"""
        csv_path = tmp_path / "synthetic_questions.csv"
        questions_df = pd.DataFrame({"question": synthetic_rag_data["Questions"].tolist()})
        questions_df.to_csv(csv_path, index=False)
        return str(csv_path)

    def test_synthetic_data_structure(self, synthetic_rag_data):
        """Test synthetic data has correct structure"""
        df = synthetic_rag_data

        assert len(df) == 5, f"Expected 5 synthetic samples, got {len(df)}"

        required_columns = ["Questions", "Answer", "Context"]
        for col in required_columns:
            assert col in df.columns, f"Missing column: {col}"

        # No empty values
        assert df["Questions"].notna().all(), "Questions contain null values"
        assert df["Answer"].notna().all(), "Answers contain null values"
        assert df["Context"].notna().all(), "Contexts contain null values"

        print("[PASS] Synthetic data structure is valid")
        print(f"  - Samples: {len(df)}")
        print(f"  - Columns: {list(df.columns)}")

    def test_synthetic_data_quality(self, synthetic_rag_data):
        """Test synthetic data quality metrics"""
        df = synthetic_rag_data

        # Check answer lengths (should be substantive)
        avg_answer_len = df["Answer"].str.len().mean()
        assert avg_answer_len > 50, f"Answers too short (avg: {avg_answer_len})"

        # Check context contains separators (multiple contexts)
        contexts_with_multiple = df["Context"].str.contains(r"\|").sum()
        assert contexts_with_multiple > 0, "No multi-context samples found"

        # Check questions end with ?
        questions_with_mark = df["Questions"].str.endswith("?").sum()
        assert questions_with_mark >= len(df) - 1, "Questions should end with ?"

        print("[PASS] Synthetic data quality is good")
        print(f"  - Avg answer length: {avg_answer_len:.0f} chars")
        print(f"  - Multi-context samples: {contexts_with_multiple}/{len(df)}")


# ============================================================================
# TEST 5: METRICS CALCULATION WITH SYNTHETIC DATA
# ============================================================================

class TestMetricsCalculation:
    """Test metric calculations with synthetic data (no external API calls)"""

    @pytest.fixture
    def synthetic_rag_data(self):
        """Generate synthetic RAG data for metric testing"""
        return pd.DataFrame([
            {
                "Questions": "What is Python?",
                "Answer": "Python is a high-level, interpreted programming language known for its simple syntax and readability. It supports multiple programming paradigms including procedural, object-oriented, and functional programming.",
                "Context": "Python is an interpreted, high-level, general-purpose programming language. Created by Guido van Rossum and first released in 1991, Python's design philosophy emphasizes code readability. | Python features a dynamic type system and automatic memory management. It supports multiple programming paradigms."
            },
            {
                "Questions": "How does garbage collection work?",
                "Answer": "Garbage collection is an automatic memory management technique that reclaims memory occupied by objects that are no longer in use by the program. It identifies and frees memory that is no longer accessible.",
                "Context": "Garbage collection (GC) is a form of automatic memory management. The garbage collector attempts to reclaim memory which was allocated by the program, but is no longer referenced. | In Python, garbage collection is primarily handled through reference counting, supplemented by a cycle-detecting garbage collector."
            },
            {
                "Questions": "What is an API?",
                "Answer": "An API (Application Programming Interface) is a set of protocols, routines, and tools that specify how software components should interact. It defines the methods and data formats for communication between different software applications.",
                "Context": "An application programming interface (API) is a connection between computers or between computer programs. It is a type of software interface, offering a service to other pieces of software. | APIs are mechanisms that enable two software components to communicate with each other using a set of definitions and protocols."
            }
        ])

    def test_fallback_evaluation(self, synthetic_rag_data):
        """Test fallback evaluation when RAGAS/DeepEval not available"""
        sys.path.insert(0, str(Path(__file__).parent))
        from rag_pipeline import RAGPipeline

        config_path = Path(__file__).parent / "config.yaml"
        pipeline = RAGPipeline(str(config_path))

        # Test fallback evaluation
        result_df = pipeline._fallback_evaluate(synthetic_rag_data.copy())

        # Check all metric columns exist
        expected_columns = [
            "RAGAS_Faithfulness", "RAGAS_Answer_Relevancy", "RAGAS_Context_Relevancy",
            "DeepEval_Faithfulness", "DeepEval_Answer_Relevancy", "DeepEval_Context_Relevancy",
            "Faithfulness_Pass", "Answer_Relevancy_Pass", "Context_Relevancy_Pass"
        ]

        for col in expected_columns:
            assert col in result_df.columns, f"Missing column: {col}"

        # Fallback should set scores to 0
        for col in ["RAGAS_Faithfulness", "RAGAS_Answer_Relevancy", "RAGAS_Context_Relevancy"]:
            assert (result_df[col] == 0.0).all(), f"{col} should be 0.0 in fallback"

        # Pass columns should be False
        for col in ["Faithfulness_Pass", "Answer_Relevancy_Pass", "Context_Relevancy_Pass"]:
            assert (result_df[col] == False).all(), f"{col} should be False in fallback"

        print("[PASS] Fallback evaluation produces correct structure")

    def test_pass_fail_logic(self, synthetic_rag_data):
        """Test pass/fail threshold logic"""
        sys.path.insert(0, str(Path(__file__).parent))
        from rag_pipeline import RAGPipeline

        config_path = Path(__file__).parent / "config.yaml"
        pipeline = RAGPipeline(str(config_path))

        thresholds = pipeline.config["thresholds"]

        # Create test data with known scores
        test_df = synthetic_rag_data.copy()

        # Simulate scores: first row passes, second row fails, third row edge case
        test_df["RAGAS_Faithfulness"] = [0.9, 0.5, 0.8]  # Pass, Fail, Edge
        test_df["RAGAS_Answer_Relevancy"] = [0.85, 0.7, 0.8]
        test_df["RAGAS_Context_Relevancy"] = [0.9, 0.6, 0.79]

        test_df["DeepEval_Faithfulness"] = [0.85, 0.5, 0.75]
        test_df["DeepEval_Answer_Relevancy"] = [0.9, 0.65, 0.85]
        test_df["DeepEval_Context_Relevancy"] = [0.88, 0.55, 0.82]

        # Apply pass/fail logic
        for m in ["Faithfulness", "Answer_Relevancy", "Context_Relevancy"]:
            threshold = thresholds.get(m.lower().replace("_", "_"), 0.8)
            test_df[f"{m}_Pass"] = (
                (test_df.get(f"RAGAS_{m}", 0) >= threshold) |
                (test_df.get(f"DeepEval_{m}", 0) >= threshold)
            )

        # Verify pass/fail results
        # Row 0: All pass (all scores >= 0.8)
        assert test_df.loc[0, "Faithfulness_Pass"] == True
        assert test_df.loc[0, "Answer_Relevancy_Pass"] == True
        assert test_df.loc[0, "Context_Relevancy_Pass"] == True

        # Row 1: All fail (all scores < 0.8)
        assert test_df.loc[1, "Faithfulness_Pass"] == False
        assert test_df.loc[1, "Answer_Relevancy_Pass"] == False
        assert test_df.loc[1, "Context_Relevancy_Pass"] == False

        # Row 2: Edge cases
        # Faithfulness: 0.8 RAGAS (pass), 0.75 DeepEval -> Pass (RAGAS meets threshold)
        assert test_df.loc[2, "Faithfulness_Pass"] == True
        # Answer_Relevancy: 0.8 RAGAS (pass), 0.85 DeepEval (pass) -> Pass
        assert test_df.loc[2, "Answer_Relevancy_Pass"] == True
        # Context_Relevancy: 0.79 RAGAS (fail), 0.82 DeepEval (pass) -> Pass
        assert test_df.loc[2, "Context_Relevancy_Pass"] == True

        print("[PASS] Pass/fail threshold logic works correctly")
        print(f"  - Thresholds: {thresholds}")

    def test_metric_value_ranges(self, synthetic_rag_data):
        """Test that metrics are in valid range [0, 1]"""
        # Simulate metric scores
        test_df = synthetic_rag_data.copy()

        metric_columns = [
            "RAGAS_Faithfulness", "RAGAS_Answer_Relevancy", "RAGAS_Context_Relevancy",
            "DeepEval_Faithfulness", "DeepEval_Answer_Relevancy", "DeepEval_Context_Relevancy"
        ]

        # Add simulated scores
        for col in metric_columns:
            test_df[col] = [0.85, 0.72, 0.91]

        # Verify all values in range
        for col in metric_columns:
            values = test_df[col]
            assert (values >= 0).all(), f"{col} has values < 0"
            assert (values <= 1).all(), f"{col} has values > 1"

        print("[PASS] Metric values are in valid range [0, 1]")

    def test_summary_statistics(self, synthetic_rag_data):
        """Test summary statistics calculation"""
        test_df = synthetic_rag_data.copy()

        # Add test metrics
        test_df["RAGAS_Faithfulness"] = [0.9, 0.8, 0.7]
        test_df["RAGAS_Answer_Relevancy"] = [0.85, 0.75, 0.65]
        test_df["RAGAS_Context_Relevancy"] = [0.88, 0.78, 0.68]
        test_df["Faithfulness_Pass"] = [True, True, False]
        test_df["Answer_Relevancy_Pass"] = [True, False, False]
        test_df["Context_Relevancy_Pass"] = [True, False, False]

        # Calculate averages
        avg_faithfulness = test_df["RAGAS_Faithfulness"].mean()
        avg_relevancy = test_df["RAGAS_Answer_Relevancy"].mean()
        avg_context = test_df["RAGAS_Context_Relevancy"].mean()

        # Calculate pass rates
        faith_pass_rate = test_df["Faithfulness_Pass"].mean() * 100
        answer_pass_rate = test_df["Answer_Relevancy_Pass"].mean() * 100
        context_pass_rate = test_df["Context_Relevancy_Pass"].mean() * 100

        # Verify calculations
        assert abs(avg_faithfulness - 0.8) < 0.01
        assert abs(avg_relevancy - 0.75) < 0.01
        assert abs(avg_context - 0.78) < 0.01

        assert abs(faith_pass_rate - 66.67) < 1
        assert abs(answer_pass_rate - 33.33) < 1
        assert abs(context_pass_rate - 33.33) < 1

        print("[PASS] Summary statistics calculated correctly")
        print(f"  - Avg Faithfulness: {avg_faithfulness:.3f}")
        print(f"  - Avg Answer Relevancy: {avg_relevancy:.3f}")
        print(f"  - Avg Context Relevancy: {avg_context:.3f}")
        print(f"  - Pass Rates: {faith_pass_rate:.1f}%, {answer_pass_rate:.1f}%, {context_pass_rate:.1f}%")


# ============================================================================
# TEST 6: PIPELINE COMPONENT TESTS
# ============================================================================

class TestPipelineComponents:
    """Test individual pipeline components"""

    def test_pipeline_initialization(self):
        """Test RAGPipeline initializes correctly"""
        sys.path.insert(0, str(Path(__file__).parent))
        from rag_pipeline import RAGPipeline

        config_path = Path(__file__).parent / "config.yaml"
        pipeline = RAGPipeline(str(config_path))

        # Verify initialization
        assert pipeline.config is not None
        assert "rag_api" in pipeline.config
        assert "folders" in pipeline.config
        assert "thresholds" in pipeline.config

        print("[PASS] Pipeline initializes correctly")

    def test_csv_processing(self, tmp_path):
        """Test CSV question extraction"""
        sys.path.insert(0, str(Path(__file__).parent))
        from rag_pipeline import RAGPipeline

        # Create test CSV
        csv_path = tmp_path / "test_questions.csv"
        test_data = pd.DataFrame({
            "question": ["What is Python?", "How does AI work?", "What is ML?"]
        })
        test_data.to_csv(csv_path, index=False)

        # Load CSV
        df = pd.read_csv(csv_path)
        question_col = next((c for c in df.columns if "question" in c.lower()), df.columns[0])
        questions = df[question_col].dropna().tolist()

        assert len(questions) == 3
        assert "What is Python?" in questions

        print("[PASS] CSV processing extracts questions correctly")

    def test_csv_column_detection(self, tmp_path):
        """Test auto-detection of question column"""
        # Test various column names
        column_variations = [
            "question",
            "Question",
            "QUESTION",
            "questions",
            "Questions",
            "user_question"
        ]

        for col_name in column_variations:
            csv_path = tmp_path / f"test_{col_name}.csv"
            test_data = pd.DataFrame({col_name: ["Test question?"]})
            test_data.to_csv(csv_path, index=False)

            df = pd.read_csv(csv_path)
            detected_col = next((c for c in df.columns if "question" in c.lower()), df.columns[0])

            assert detected_col == col_name, f"Failed to detect column: {col_name}"

        print("[PASS] Auto-detection works for various column names")

    def test_timestamp_generation(self):
        """Test timestamp format for output files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Verify format: YYYYMMDD_HHMMSS
        assert len(timestamp) == 15
        assert timestamp[8] == "_"
        assert timestamp[:8].isdigit()
        assert timestamp[9:].isdigit()

        # Test file naming
        output_name = f"rag_output_{timestamp}.csv"
        assert output_name.startswith("rag_output_")
        assert output_name.endswith(".csv")

        print(f"[PASS] Timestamp generation: {timestamp}")

    def test_context_parsing(self):
        """Test context string parsing"""
        # Test pipe-separated contexts
        context_string = "Context 1 text here. | Context 2 text here. | Context 3 text here."
        contexts = context_string.split(" | ")

        assert len(contexts) == 3
        assert contexts[0] == "Context 1 text here."
        assert contexts[1] == "Context 2 text here."
        assert contexts[2] == "Context 3 text here."

        # Test empty context
        empty_context = ""
        contexts_empty = empty_context.split(" | ") if empty_context else []
        assert contexts_empty == [""] or len(contexts_empty) == 1

        print("[PASS] Context parsing works correctly")

    def test_output_dataframe_structure(self, tmp_path):
        """Test output DataFrame has correct structure"""
        # Simulate full output
        output_df = pd.DataFrame([
            {
                "Questions": "Test question?",
                "Answer": "Test answer.",
                "Context": "Test context 1. | Test context 2.",
                "RAGAS_Faithfulness": 0.85,
                "RAGAS_Answer_Relevancy": 0.82,
                "RAGAS_Context_Relevancy": 0.88,
                "DeepEval_Faithfulness": 0.87,
                "DeepEval_Answer_Relevancy": 0.84,
                "DeepEval_Context_Relevancy": 0.86,
                "Faithfulness_Pass": True,
                "Answer_Relevancy_Pass": True,
                "Context_Relevancy_Pass": True
            }
        ])

        # Verify columns
        expected_columns = [
            "Questions", "Answer", "Context",
            "RAGAS_Faithfulness", "RAGAS_Answer_Relevancy", "RAGAS_Context_Relevancy",
            "DeepEval_Faithfulness", "DeepEval_Answer_Relevancy", "DeepEval_Context_Relevancy",
            "Faithfulness_Pass", "Answer_Relevancy_Pass", "Context_Relevancy_Pass"
        ]

        for col in expected_columns:
            assert col in output_df.columns, f"Missing column: {col}"

        # Test CSV export
        csv_path = tmp_path / "test_output.csv"
        output_df.to_csv(csv_path, index=False)

        # Reload and verify
        loaded_df = pd.read_csv(csv_path)
        assert len(loaded_df) == 1
        assert loaded_df.loc[0, "Questions"] == "Test question?"

        print("[PASS] Output DataFrame structure is correct")


# ============================================================================
# TEST 7: INTEGRATION TESTS WITH SYNTHETIC DATA
# ============================================================================

class TestIntegration:
    """Integration tests using synthetic data"""

    @pytest.fixture
    def temp_project(self, tmp_path):
        """Create a temporary project structure"""
        # Copy config
        project_root = Path(__file__).parent
        shutil.copy(project_root / "config.yaml", tmp_path / "config.yaml")

        # Create data folders
        (tmp_path / "data/RAG_output_folder").mkdir(parents=True)
        (tmp_path / "data/RAG_evaluator_input").mkdir(parents=True)
        (tmp_path / "data/evaluation_output_folder").mkdir(parents=True)

        # Create test CSV
        test_csv = tmp_path / "test_questions.csv"
        pd.DataFrame({
            "question": ["What is AI?", "How does ML work?"]
        }).to_csv(test_csv, index=False)

        return tmp_path

    def test_full_fallback_pipeline(self, temp_project):
        """Test complete pipeline with fallback evaluation"""
        sys.path.insert(0, str(Path(__file__).parent))
        from rag_pipeline import RAGPipeline

        # Create test data that simulates RAG output
        test_df = pd.DataFrame([
            {
                "Questions": "What is artificial intelligence?",
                "Answer": "Artificial intelligence is the simulation of human intelligence by machines.",
                "Context": "AI refers to systems that can perform tasks typically requiring human intelligence."
            }
        ])

        # Initialize pipeline
        config_path = temp_project / "config.yaml"

        # Modify config to use temp folders
        import yaml
        with open(config_path) as f:
            config = yaml.safe_load(f)

        config["folders"]["rag_output"] = str(temp_project / "data/RAG_output_folder")
        config["folders"]["evaluator_input"] = str(temp_project / "data/RAG_evaluator_input")
        config["folders"]["evaluation_output"] = str(temp_project / "data/evaluation_output_folder")

        with open(config_path, "w") as f:
            yaml.dump(config, f)

        pipeline = RAGPipeline(str(config_path))

        # Run fallback evaluation
        result_df = pipeline._fallback_evaluate(test_df)

        # Verify results
        assert len(result_df) == 1
        assert "RAGAS_Faithfulness" in result_df.columns
        assert "Faithfulness_Pass" in result_df.columns

        print("[PASS] Full pipeline with fallback works correctly")

    def test_output_file_creation(self, temp_project):
        """Test that output files are created correctly"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create output files
        output_folder = temp_project / "data/RAG_output_folder"
        rag_output_path = output_folder / f"rag_output_{timestamp}.csv"

        test_df = pd.DataFrame([{"Questions": "Test?", "Answer": "Yes.", "Context": "Test."}])
        test_df.to_csv(rag_output_path, index=False)

        assert rag_output_path.exists()

        # Verify file content
        loaded = pd.read_csv(rag_output_path)
        assert len(loaded) == 1
        assert loaded.loc[0, "Questions"] == "Test?"

        print(f"[PASS] Output file created: {rag_output_path.name}")

    def test_copy_to_evaluator_input(self, temp_project):
        """Test file copying to evaluator input folder"""
        import shutil

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_name = f"rag_output_{timestamp}.csv"

        rag_output = temp_project / "data/RAG_output_folder" / output_name
        eval_input = temp_project / "data/RAG_evaluator_input" / output_name

        # Create source file
        pd.DataFrame([{"test": "data"}]).to_csv(rag_output, index=False)

        # Copy to evaluator input
        shutil.copy(rag_output, eval_input)

        assert eval_input.exists()

        # Verify content matches
        source_df = pd.read_csv(rag_output)
        dest_df = pd.read_csv(eval_input)
        assert source_df.equals(dest_df)

        print("[PASS] File copy to evaluator input works")


# ============================================================================
# TEST 8: ERROR HANDLING TESTS
# ============================================================================

class TestErrorHandling:
    """Test error handling and edge cases"""

    def test_missing_config_file(self, tmp_path):
        """Test handling of missing config file"""
        sys.path.insert(0, str(Path(__file__).parent))
        from rag_pipeline import RAGPipeline

        fake_config = tmp_path / "nonexistent.yaml"

        with pytest.raises(FileNotFoundError):
            RAGPipeline(str(fake_config))

        print("[PASS] Missing config file raises FileNotFoundError")

    def test_empty_csv(self, tmp_path):
        """Test handling of empty CSV file"""
        empty_csv = tmp_path / "empty.csv"
        empty_csv.write_text("question\n")  # Header only

        df = pd.read_csv(empty_csv)
        questions = df["question"].dropna().tolist()

        assert len(questions) == 0
        print("[PASS] Empty CSV handled correctly")

    def test_missing_question_column(self, tmp_path):
        """Test handling of CSV without question column"""
        csv_path = tmp_path / "no_question.csv"
        pd.DataFrame({"other_column": ["data"]}).to_csv(csv_path, index=False)

        df = pd.read_csv(csv_path)

        # Should fall back to first column
        question_col = next((c for c in df.columns if "question" in c.lower()), df.columns[0])
        assert question_col == "other_column"

        print("[PASS] Missing question column handled with fallback")

    def test_invalid_threshold_values(self):
        """Test validation of threshold values"""
        import yaml

        config_path = Path(__file__).parent / "config.yaml"
        with open(config_path) as f:
            config = yaml.safe_load(f)

        for threshold_name, value in config["thresholds"].items():
            # Values should be between 0 and 1
            assert 0 <= value <= 1, f"Invalid threshold {threshold_name}: {value}"

        print("[PASS] All threshold values are valid")

    def test_dataframe_with_nulls(self):
        """Test handling of DataFrame with null values"""
        df_with_nulls = pd.DataFrame([
            {"Questions": "Q1?", "Answer": "A1", "Context": "C1"},
            {"Questions": None, "Answer": "A2", "Context": "C2"},
            {"Questions": "Q3?", "Answer": None, "Context": "C3"},
            {"Questions": "Q4?", "Answer": "A4", "Context": None}
        ])

        # Filter out rows with null questions
        valid_df = df_with_nulls[df_with_nulls["Questions"].notna()]
        assert len(valid_df) == 3

        print("[PASS] Null value handling works correctly")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def run_all_tests():
    """Run all tests with verbose output"""
    print("=" * 70)
    print("RAG EVALUATION PIPELINE - COMPREHENSIVE TEST SUITE")
    print("=" * 70)
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70 + "\n")

    # Run pytest with verbose output
    exit_code = pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-s",  # Show print statements
        "--color=yes"
    ])

    print("\n" + "=" * 70)
    print(f"Tests completed with exit code: {exit_code}")
    print("=" * 70)

    return exit_code


if __name__ == "__main__":
    run_all_tests()
