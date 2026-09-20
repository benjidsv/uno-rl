import math
import re
import mlflow
import pandas as pd


def count_tokens(text: str, model: str = "gpt-4o") -> tuple[int, str]:
    """Returns (token_count, method_used).

    Uses tiktoken if available, with a 4-chars-per-token fallback.
    """
    try:
        import tiktoken

        try:
            enc = tiktoken.encoding_for_model(model)
        except KeyError:
            enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text)), f"exact ({enc.name})"
    except ImportError:
        return max(1, len(text) // 4), "estimated (~4 chars/token)"


def fetch_mlflow_run_data(
    run_id: str,
    client: mlflow.tracking.MlflowClient = None,
    include_pattern: str = None,
    exclude_pattern: str = None,
    exclude_metrics: list[str] | set[str] = None,
) -> dict[str, pd.DataFrame]:
    """Fetches metric histories from MLflow.

    Filtering order:
      1. exclude_metrics (exact match from a list/set)
      2. include_pattern (regex match)
      3. exclude_pattern (regex match)
    """
    if client is None:
        client = mlflow.tracking.MlflowClient()

    exclude_set = set(exclude_metrics) if exclude_metrics else set()

    run = client.get_run(run_id)
    metrics_data = {}

    for metric_name in run.data.metrics.keys():
        # 1. Exact metric name exclusion
        if metric_name in exclude_set:
            continue

        # 2. Regex inclusion (e.g. include "actor" group)
        if include_pattern and not re.search(include_pattern, metric_name):
            continue

        # 3. Regex pattern exclusion (e.g. general terms)
        if exclude_pattern and re.search(exclude_pattern, metric_name):
            continue

        history = client.get_metric_history(run_id, metric_name)
        if not history:
            continue

        steps = [m.step for m in history]
        vals = [m.value for m in history]

        df = (
            pd
            .DataFrame({"step": steps, "value": vals})
            .sort_values("step")
            .reset_index(drop=True)
        )
        metrics_data[metric_name] = df

    return metrics_data


def min_max_downsample(df: pd.DataFrame, num_windows: int) -> list[dict]:
    """Splits timeline into windows and extracts tagged local min and max points."""
    if len(df) == 0:
        return []

    window_size = math.ceil(len(df) / max(1, num_windows))
    tagged_points = {}

    for i in range(num_windows):
        start_idx = i * window_size
        end_idx = min((i + 1) * window_size, len(df))

        window = df.iloc[start_idx:end_idx]
        if window.empty:
            continue

        min_idx = window["value"].idxmin()
        max_idx = window["value"].idxmax()

        if min_idx == max_idx:
            tagged_points[min_idx] = "flat"
        else:
            tagged_points.setdefault(min_idx, set()).add("min")
            tagged_points.setdefault(max_idx, set()).add("max")

    results = []
    for idx in sorted(tagged_points.keys()):
        row = df.loc[idx]
        tags = tagged_points[idx]
        tag_str = "-" if tags == "flat" else "/".join(sorted(tags))

        results.append({"step": row["step"], "value": row["value"], "tag": tag_str})

    return results


def export_run_for_llm(
    metrics_data: dict[str, pd.DataFrame],
    output_path: str = "rl_run_analysis.txt",
    windows_per_metric: int = 10,
):
    """Formats metrics as minimal CSV with an LLM system prompt, saves to a file,

    and displays only the summary in the terminal.
    """
    if not metrics_data:
        print("No metrics matched the criteria. Nothing written.")
        return

    # Prompt explaining data format to the LLM
    output_lines = [
        "# RL Training Run Metrics Analysis",
        "",
        "## Context & Data Schema for Assistant",
        (
            "The following data contains reinforcement learning training"
            " metrics downsampled via Min-Max Windowing."
        ),
        (
            "- Each metric is divided into uniform time windows; the absolute"
            " minimum (`min`) and maximum (`max`) within each window are"
            " preserved chronologically."
        ),
        "- Format under each metric header is CSV: `step,value,tag`.",
        "- Column semantics:",
        (
            "  * `step`: Training environment step (`k` = 1,000 steps, `M` ="
            " 1,000,000 steps)."
        ),
        "  * `value`: Metric value formatted up to 4 significant digits.",
        "  * `tag`:",
        "    - `min`: Local minimum within that time window.",
        (
            "    - `max`: Local maximum within that time window (captures"
            " gradient blowups, reward spikes, etc.)."
        ),
        ("    - `-`: Metric stayed flat/constant throughout the entire window."),
        (
            "When analyzing: Cross-reference steps across metrics to detect"
            " correlations, policy instability (e.g., KL divergence or critic"
            " loss spikes), and learning trends."
        ),
        "---",
        "",
    ]

    total_points = 0

    for metric_name, df in metrics_data.items():
        output_lines.append(f"## {metric_name}")
        output_lines.append("step,value,tag")

        sampled_points = min_max_downsample(df, num_windows=windows_per_metric)
        total_points += len(sampled_points)

        for pt in sampled_points:
            step_val = pt["step"]
            if step_val >= 1_000_000:
                step_str = f"{step_val / 1_000_000:g}M"
            elif step_val >= 1_000:
                step_str = f"{step_val / 1_000:g}k"
            else:
                step_str = f"{step_val}"

            val_str = f"{pt['value']:.4g}"
            output_lines.append(f"{step_str},{val_str},{pt['tag']}")

        output_lines.append("")

    final_text = "\n".join(output_lines)

    # Write directly to file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(final_text)

    # Terminal summary output only
    tokens, method = count_tokens(final_text)
    num_metrics = len(metrics_data)
    avg_points = total_points / max(1, num_metrics)

    print(f"\n{'=' * 45}")
    print(f" MLflow RL Data Export Complete")
    print(f"{'=' * 45}")
    print(f" - Output file       : {output_path}")
    print(f" - Metrics written   : {num_metrics}")
    print(f" - Total data points : {total_points} (avg {avg_points:.1f} / metric)")
    print(f" - Characters        : {len(final_text):,}")
    print(f" - Total tokens      : ~{tokens:,} [{method}]")
    print(f"{'=' * 45}\n")


# --- Execution Example ---
if __name__ == "__main__":
    RUN_ID = "15cc37bf5c2742d48beadd6776741b21"

    # Fetch data with regex filtering
    metrics = fetch_mlflow_run_data(
        run_id=RUN_ID,
        include_pattern=r"^(actor|critic|batch|training|evaluation)/",
        exclude_pattern=r"behavior",
        # remove duplicated metrics (eg surrogate is also value loss, critic mse is policy loss, etc.)
        exclude_metrics={"actor/surrogate", "critic/mse"},
        # add back the few useful behavior metrics
        # include_metrics={"evaluation/behavior_mean_game_length"},
    )

    # Process, write to file, and print terminal summary
    export_run_for_llm(
        metrics_data=metrics,
        output_path="rl_run_analysis.txt",
        windows_per_metric=20,  # 10 windows = max ~20 points per metric
    )
