from pathlib import Path

from llm_benchmark_suite.config import load_run_config


def test_load_local_demo_config() -> None:
    config = load_run_config("configs/profiles/local-demo.yaml")
    assert config.profile_name == "local-demo"
    assert "vllm" in config.selected_backends


def test_load_run_config_resolves_paths_relative_to_profile(tmp_path) -> None:
    profile_dir = tmp_path / "profiles"
    data_dir = tmp_path / "data"
    costs_dir = tmp_path / "costs"
    thresholds_dir = tmp_path / "thresholds"
    profile_dir.mkdir()
    data_dir.mkdir()
    costs_dir.mkdir()
    thresholds_dir.mkdir()

    (data_dir / "sample.jsonl").write_text(
        '{"id":"1","prompt":"Hello","reference":"Hello","task_type":"qa"}\n',
        encoding="utf-8",
    )
    (costs_dir / "default.yaml").write_text(
        "gpu_hourly_cost_usd: 1\ncpu_hourly_cost_usd: 1\nmemory_gb_hourly_cost_usd: 1\n",
        encoding="utf-8",
    )
    (thresholds_dir / "default.yaml").write_text(
        "\n".join(
            [
                "p95_latency_regression_pct: 10",
                "ttft_regression_pct: 10",
                "throughput_regression_pct: 10",
                "accuracy_min: 0.5",
                "cost_per_million_tokens_regression_pct: 10",
                "error_rate_max: 0.1",
                "",
            ]
        ),
        encoding="utf-8",
    )
    profile_path = profile_dir / "custom.yaml"
    profile_path.write_text(
        "\n".join(
            [
                "profile_name: custom",
                "mock_mode: true",
                "output_dir: ../artifacts/custom",
                "report_formats: [json]",
                "selected_backends: [vllm]",
                "datasets:",
                "  - name: sample",
                "    path: ../data/sample.jsonl",
                "    task_type: qa",
                "backend_defaults:",
                "  model_name: demo",
                "  precision: fp16",
                "cost_profile: ../costs/default.yaml",
                "thresholds_profile: ../thresholds/default.yaml",
            ]
        ),
        encoding="utf-8",
    )

    config = load_run_config(profile_path)

    assert Path(config.output_dir).is_absolute()
    assert Path(config.datasets[0].path) == (data_dir / "sample.jsonl").resolve()
    assert Path(config.cost_profile) == (costs_dir / "default.yaml").resolve()
    assert Path(config.thresholds_profile) == (thresholds_dir / "default.yaml").resolve()
