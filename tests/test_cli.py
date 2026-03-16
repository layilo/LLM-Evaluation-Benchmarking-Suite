import json
from pathlib import Path

from click.testing import CliRunner

from llm_benchmark_suite.cli import main
from llm_benchmark_suite.schemas.models import BenchmarkSummary


def test_demo_command_runs() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["demo", "--output-dir", "artifacts/generated/test-demo"])
    assert result.exit_code == 0, result.output
    assert "Benchmark Summary" in result.output


def test_export_baseline_creates_parent_directory(tmp_path) -> None:
    runner = CliRunner()
    output_path = tmp_path / "nested" / "baseline.json"
    result = runner.invoke(
        main,
        [
            "export-baseline",
            "--input",
            "artifacts/sample_run/summary.json",
            "--output",
            str(output_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert output_path.exists()


def test_compare_output_includes_backend_and_dataset_for_multi_pair_results(tmp_path) -> None:
    sample = BenchmarkSummary.model_validate(
        json.loads(Path("artifacts/sample_run/summary.json").read_text(encoding="utf-8"))
    )
    current = sample.model_copy(deep=True)
    baseline = sample.model_copy(deep=True)

    current.backend_metrics.append(current.backend_metrics[0].model_copy(deep=True))
    current.backend_metrics[1].backend_name = "onnx_runtime"
    current.accuracy_metrics.append(current.accuracy_metrics[0].model_copy(deep=True))
    current.accuracy_metrics[1].backend_name = "onnx_runtime"
    current.cost_metrics.append(current.cost_metrics[0].model_copy(deep=True))
    current.cost_metrics[1].backend_name = "onnx_runtime"

    baseline.backend_metrics.append(baseline.backend_metrics[0].model_copy(deep=True))
    baseline.backend_metrics[1].backend_name = "onnx_runtime"
    baseline.accuracy_metrics.append(baseline.accuracy_metrics[0].model_copy(deep=True))
    baseline.accuracy_metrics[1].backend_name = "onnx_runtime"
    baseline.cost_metrics.append(baseline.cost_metrics[0].model_copy(deep=True))
    baseline.cost_metrics[1].backend_name = "onnx_runtime"

    current_path = tmp_path / "current.json"
    baseline_path = tmp_path / "baseline.json"
    current_path.write_text(current.model_dump_json(indent=2), encoding="utf-8")
    baseline_path.write_text(baseline.model_dump_json(indent=2), encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "compare",
            "--current",
            str(current_path),
            "--baseline",
            str(baseline_path),
            "--thresholds",
            "configs/thresholds/default.yaml",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "onnx_runtime/summarization p95_latency" in result.output
