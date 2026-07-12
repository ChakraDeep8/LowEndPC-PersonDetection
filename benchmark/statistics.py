import csv
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from datetime import datetime
from zoneinfo import ZoneInfo
import config


ISOLATED_NOTE = "Isolated Baseline | Trial"

BACKEND_ORDER = [
    "PyTorch",
    "ONNX Runtime",
    "OpenVINO",
    "OpenCV DNN",
]

NUMERIC_COLUMNS = {
    "model_load": "Model Load(ms)",
    "image_read": "Image Read(ms)",
    "preprocess": "Preprocess(ms)",
    "inference": "Inference(ms)",
    "postprocess": "Postprocess(ms)",
    "draw": "Draw(ms)",
    "save": "Save(ms)",
    "total": "Total(ms)",
    "fps": "FPS",
    "cpu": "CPU Usage(%)",
    "ram": "RAM Usage(%)",
    "memory": "Process Memory(MB)",
    "threads": "Threads",
    "detections": "Detections",
}

class BenchmarkStatistics:
    def __init__(self, csv_path):
        self.csv_path = Path(csv_path)
        self.rows = []
        self.statistics = {}

    def load(self):
        """
        Load repeated isolated benchmark trials.
        """

        if not self.csv_path.exists():
            raise FileNotFoundError(
                f"Benchmark CSV not found: {self.csv_path}"
            )

        with self.csv_path.open(
            "r",
            encoding="utf-8",
            newline="",
        ) as file:

            reader = csv.DictReader(file)

            self.rows = [
                row
                for row in reader
                if ISOLATED_NOTE in row["Notes"]
            ]

        if not self.rows:
            raise RuntimeError(
                "No repeated isolated benchmark trials found."
            )

    def calculate(self):
        """
        Calculate statistics for every backend and every numeric metric.
        """

        grouped = defaultdict(list)

        for row in self.rows:
            grouped[row["Engine"]].append(row)

        self.statistics.clear()

        for backend in BACKEND_ORDER:

            if backend not in grouped:
                continue

            rows = grouped[backend]

            backend_statistics = {
                "trials": len(rows)
            }

            for metric_name, csv_column in NUMERIC_COLUMNS.items():

                values = [
                    float(row[csv_column])
                    for row in rows
                ]

                mean_value = statistics.mean(values)

                if len(values) > 1:
                    std_value = statistics.stdev(values)
                else:
                    std_value = 0.0

                if mean_value != 0:
                    cv_value = (
                        std_value / mean_value
                    ) * 100
                else:
                    cv_value = 0.0

                backend_statistics[metric_name] = {
                    "values": values,
                    "mean": mean_value,
                    "median": statistics.median(values),
                    "std": std_value,
                    "min": min(values),
                    "max": max(values),
                    "cv": cv_value,
                }

            self.statistics[backend] = backend_statistics

    def save_statistics(self):
        """
        Save detailed backend statistics.
        """

        

        today = datetime.now(
            ZoneInfo(config.TIMEZONE)
        ).strftime("%d-%m-%Y")

        output_path = (
            config.BENCHMARK_STAT_CSV
            / f"stat_{today}.csv"
        )

        fieldnames = [
            "Engine",
            "Trials",
        ]

        for metric in NUMERIC_COLUMNS.keys():

            metric_name = (
                metric.replace("_", " ")
                .title()
            )

            fieldnames.extend(
                [
                    f"{metric_name} Mean",
                    f"{metric_name} Median",
                    f"{metric_name} Std",
                    f"{metric_name} Min",
                    f"{metric_name} Max",
                    f"{metric_name} CV(%)",
                ]
            )

        with output_path.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames,
            )

            writer.writeheader()

            for backend in BACKEND_ORDER:

                if backend not in self.statistics:
                    continue

                data = self.statistics[backend]

                row = {
                    "Engine": backend,
                    "Trials": data["trials"],
                }

                for metric in NUMERIC_COLUMNS.keys():

                    metric_name = (
                        metric.replace("_", " ")
                        .title()
                    )

                    row[
                        f"{metric_name} Mean"
                    ] = round(
                        data[metric]["mean"],
                        3,
                    )

                    row[
                        f"{metric_name} Median"
                    ] = round(
                        data[metric]["median"],
                        3,
                    )

                    row[
                        f"{metric_name} Std"
                    ] = round(
                        data[metric]["std"],
                        3,
                    )

                    row[
                        f"{metric_name} Min"
                    ] = round(
                        data[metric]["min"],
                        3,
                    )

                    row[
                        f"{metric_name} Max"
                    ] = round(
                        data[metric]["max"],
                        3,
                    )

                    row[
                        f"{metric_name} CV(%)"
                    ] = round(
                        data[metric]["cv"],
                        2,
                    )

                writer.writerow(row)

        print(f"Statistics saved: {output_path}")

    def save_rankings(self):
        """
        Save backend rankings for key benchmark metrics.
        """

        today = datetime.now(
            ZoneInfo(config.TIMEZONE)
        ).strftime("%d-%m-%Y")

        output_path = (
            config.BENCHMARK_STAT_CSV
            / f"backend_ranking_{today}.csv"
        )
        ranking_metrics = [
            ("Inference", "inference", False),
            ("Total", "total", False),
            ("FPS", "fps", True),
            ("Process Memory", "memory", False),
            ("CPU Usage", "cpu", False),
            ("Model Load", "model_load", False),
        ]

        with output_path.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as file:

            writer = csv.writer(file)

            writer.writerow(
                [
                    "Metric",
                    "Rank",
                    "Backend",
                    "Mean Value",
                ]
            )

            for (
                metric_title,
                metric_key,
                descending,
            ) in ranking_metrics:

                ranking = sorted(
                    self.statistics.items(),
                    key=lambda item:
                        item[1][metric_key]["mean"],
                    reverse=descending,
                )

                for rank, (
                    backend,
                    values,
                ) in enumerate(
                    ranking,
                    start=1,
                ):

                    writer.writerow(
                        [
                            metric_title,
                            rank,
                            backend,
                            round(
                                values[metric_key]["mean"],
                                3,
                            ),
                        ]
                    )

        print(f"Rankings saved: {output_path}")

    def setup_plot(
        self,
        title,
        xlabel,
        ylabel,
        figsize=(9, 6),
    ):
        """
        Configure a consistent plot style.
        """

        plt.figure(figsize=figsize)

        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        plt.grid(
            axis="y",
            linestyle="--",
            alpha=0.3,
        )

    def annotate_bars(self, bars):
        """
        Display values above each bar.
        """

        for bar in bars:

            height = bar.get_height()

            plt.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    def save_bar_chart(
        self,
        filename,
        title,
        ylabel,
        metric,
    ):
        """
        Save a backend comparison bar chart.
        """

        backends = [
            backend
            for backend in BACKEND_ORDER
            if backend in self.statistics
        ]

        values = [
            self.statistics[backend][metric]["mean"]
            for backend in backends
        ]

        self.setup_plot(
            title,
            "Backend",
            ylabel,
        )

        bars = plt.bar(
            backends,
            values,
        )

        self.annotate_bars(bars)

        output_path = (
            config.BENCHMARK_STAT
            / filename
        )

        plt.tight_layout()

        plt.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()

        print(f"Graph saved: {output_path}")

    def save_variability_chart(self):
        """
        Plot inference latency mean with
        standard deviation error bars.
        """

        backends = [
            backend
            for backend in BACKEND_ORDER
            if backend in self.statistics
        ]

        means = [
            self.statistics[backend]["inference"]["mean"]
            for backend in backends
        ]

        errors = [
            self.statistics[backend]["inference"]["std"]
            for backend in backends
        ]

        self.setup_plot(
            "Inference Latency Variability",
            "Backend",
            "Inference Latency (ms)",
        )

        bars = plt.bar(
            backends,
            means,
            yerr=errors,
            capsize=6,
        )

        self.annotate_bars(bars)

        output_path = (
            config.BENCHMARK_STAT
            / "inference_variability.png"
        )

        plt.tight_layout()

        plt.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()

        print(f"Graph saved: {output_path}")

    def save_trial_chart(self):
        """
        Plot inference latency
        across all isolated trials.
        """

        self.setup_plot(
            "Per-Trial Inference Latency",
            "Trial",
            "Inference (ms)",
            figsize=(10, 6),
        )

        max_trials = 0

        for backend in BACKEND_ORDER:

            if backend not in self.statistics:
                continue

            values = self.statistics[
                backend
            ]["inference"]["values"]

            trials = list(
                range(
                    1,
                    len(values) + 1,
                )
            )

            max_trials = max(
                max_trials,
                len(values),
            )

            plt.plot(
                trials,
                values,
                marker="o",
                linewidth=2,
                label=backend,
            )

        plt.xticks(
            range(
                1,
                max_trials + 1,
            )
        )

        plt.legend()

        output_path = (
            config.BENCHMARK_STAT
            / "trial_latency_comparison.png"
        )

        plt.tight_layout()

        plt.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()

        print(f"Graph saved: {output_path}")

    def save_total_trial_chart(self):
        """
        Plot total latency
        across all isolated trials.
        """

        self.setup_plot(
            "Per-Trial Total Latency",
            "Trial",
            "Total Latency (ms)",
            figsize=(10, 6),
        )

        max_trials = 0

        for backend in BACKEND_ORDER:

            if backend not in self.statistics:
                continue

            values = self.statistics[
                backend
            ]["total"]["values"]

            trials = list(
                range(
                    1,
                    len(values) + 1,
                )
            )

            max_trials = max(
                max_trials,
                len(values),
            )

            plt.plot(
                trials,
                values,
                marker="o",
                linewidth=2,
                label=backend,
            )

        plt.xticks(
            range(
                1,
                max_trials + 1,
            )
        )

        plt.legend()

        output_path = (
            config.BENCHMARK_STAT
            / "total_latency_trial_comparison.png"
        )

        plt.tight_layout()

        plt.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()

        print(f"Graph saved: {output_path}")

    def print_summary(self):
        """
        Print a concise benchmark summary.
        """

        print("\n" + "=" * 60)
        print("Statistical Benchmark Summary")
        print("=" * 60)

        summary_metrics = [
            ("Fastest Inference", "inference", False),
            ("Lowest Total Time", "total", False),
            ("Highest FPS", "fps", True),
            ("Lowest Process Memory", "memory", False),
            ("Lowest CPU Usage", "cpu", False),
            ("Fastest Model Load", "model_load", False),
        ]

        for title, metric, reverse in summary_metrics:

            ranking = sorted(
                self.statistics.items(),
                key=lambda item: item[1][metric]["mean"],
                reverse=reverse,
            )

            backend, values = ranking[0]

            print(
                f"{title:<24}: "
                f"{backend:<15}"
                f"{values[metric]['mean']:.3f}"
            )

        print("=" * 60)

    def generate(self):
        """
        Execute the complete statistical analysis.
        """

        self.load()

        self.calculate()

        self.save_statistics()

        self.save_rankings()

        self.save_bar_chart(
            "inference_latency.png",
            "Mean Inference Latency by Backend",
            "Inference Latency (ms)",
            "inference",
        )

        self.save_bar_chart(
            "total_latency.png",
            "Mean Total Latency by Backend",
            "Total Latency (ms)",
            "total",
        )

        self.save_bar_chart(
            "fps_comparison.png",
            "Mean FPS by Backend",
            "FPS",
            "fps",
        )

        self.save_bar_chart(
            "memory_comparison.png",
            "Mean Process Memory by Backend",
            "Process Memory (MB)",
            "memory",
        )

        self.save_variability_chart()

        self.save_trial_chart()

        self.save_total_trial_chart()

        self.print_summary()

        
def find_latest_benchmark_csv():
    csv_files = list(config.CSV.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No benchmark CSV files found in: {config.CSV}"
        )

    return max(
        csv_files,
        key=lambda path: path.stat().st_mtime,
    )


def main():
    csv_path = find_latest_benchmark_csv()

    print(f"Benchmark CSV: {csv_path}")

    analyzer = BenchmarkStatistics(csv_path)
    analyzer.generate()

    print("Statistical benchmark analysis complete.")


if __name__ == "__main__":
    main()