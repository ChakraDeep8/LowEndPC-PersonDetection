import csv
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

import config


ISOLATED_NOTE = "Isolated Baseline | Trial"


class BenchmarkStatistics:
    def __init__(self, csv_path):
        self.csv_path = Path(csv_path)
        self.rows = []
        self.statistics = {}

    def load(self):
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
        grouped = defaultdict(list)

        for row in self.rows:
            grouped[row["Engine"]].append(row)

        for engine, rows in grouped.items():
            inference = [
                float(row["Inference(ms)"])
                for row in rows
            ]
            total = [
                float(row["Total(ms)"])
                for row in rows
            ]
            fps = [
                float(row["FPS"])
                for row in rows
            ]
            memory = [
                float(row["Process Memory(MB)"])
                for row in rows
            ]

            inference_mean = statistics.mean(inference)
            inference_std = statistics.stdev(inference)

            self.statistics[engine] = {
                "trials": len(rows),
                "inference": inference,
                "inference_mean": inference_mean,
                "inference_median": statistics.median(inference),
                "inference_std": inference_std,
                "inference_min": min(inference),
                "inference_max": max(inference),
                "inference_cv": (
                    inference_std / inference_mean
                ) * 100,
                "total_mean": statistics.mean(total),
                "fps_mean": statistics.mean(fps),
                "memory_mean": statistics.mean(memory),
            }

    def save_statistics(self):
        output_path = (
            config.BENCHMARK_STAT
            / "backend_statistics.csv"
        )

        fieldnames = [
            "Engine",
            "Trials",
            "Inference Mean(ms)",
            "Inference Median(ms)",
            "Inference Std(ms)",
            "Inference Min(ms)",
            "Inference Max(ms)",
            "Inference CV(%)",
            "Total Mean(ms)",
            "FPS Mean",
            "Process Memory Mean(MB)",
        ]

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

            for engine, data in self.statistics.items():
                writer.writerow(
                    {
                        "Engine": engine,
                        "Trials": data["trials"],
                        "Inference Mean(ms)": round(
                            data["inference_mean"], 3
                        ),
                        "Inference Median(ms)": round(
                            data["inference_median"], 3
                        ),
                        "Inference Std(ms)": round(
                            data["inference_std"], 3
                        ),
                        "Inference Min(ms)": round(
                            data["inference_min"], 3
                        ),
                        "Inference Max(ms)": round(
                            data["inference_max"], 3
                        ),
                        "Inference CV(%)": round(
                            data["inference_cv"], 2
                        ),
                        "Total Mean(ms)": round(
                            data["total_mean"], 3
                        ),
                        "FPS Mean": round(
                            data["fps_mean"], 3
                        ),
                        "Process Memory Mean(MB)": round(
                            data["memory_mean"], 3
                        ),
                    }
                )

        print(f"Statistics saved: {output_path}")

    def save_bar_chart(
        self,
        filename,
        title,
        ylabel,
        key,
    ):
        engines = list(self.statistics.keys())
        values = [
            self.statistics[engine][key]
            for engine in engines
        ]

        plt.figure(figsize=(9, 6))
        plt.bar(engines, values)
        plt.title(title)
        plt.xlabel("Backend")
        plt.ylabel(ylabel)
        plt.tight_layout()

        output_path = (
            config.BENCHMARK_STAT / filename
        )

        plt.savefig(output_path, dpi=300)
        plt.close()

        print(f"Graph saved: {output_path}")

    def save_variability_chart(self):
        engines = list(self.statistics.keys())

        means = [
            self.statistics[engine]["inference_mean"]
            for engine in engines
        ]

        errors = [
            self.statistics[engine]["inference_std"]
            for engine in engines
        ]

        plt.figure(figsize=(9, 6))

        plt.bar(
            engines,
            means,
            yerr=errors,
            capsize=5,
        )

        plt.title(
            "Inference Latency Mean and Standard Deviation"
        )
        plt.xlabel("Backend")
        plt.ylabel("Inference Latency (ms)")
        plt.tight_layout()

        output_path = (
            config.BENCHMARK_STAT
            / "inference_variability.png"
        )

        plt.savefig(output_path, dpi=300)
        plt.close()

        print(f"Graph saved: {output_path}")

    def save_trial_chart(self):
        plt.figure(figsize=(10, 6))

        for engine, data in self.statistics.items():
            trials = range(
                1,
                len(data["inference"]) + 1,
            )

            plt.plot(
                trials,
                data["inference"],
                marker="o",
                label=engine,
            )

        plt.title("Per-Trial Inference Latency")
        plt.xlabel("Trial")
        plt.ylabel("Inference Latency (ms)")
        plt.xticks(
            range(
                1,
                max(
                    len(data["inference"])
                    for data in self.statistics.values()
                )
                + 1,
            )
        )
        plt.legend()
        plt.tight_layout()

        output_path = (
            config.BENCHMARK_STAT
            / "trial_latency_comparison.png"
        )

        plt.savefig(output_path, dpi=300)
        plt.close()

        print(f"Graph saved: {output_path}")

    def generate(self):
        self.load()
        self.calculate()
        self.save_statistics()

        self.save_bar_chart(
            "inference_latency.png",
            "Mean Inference Latency by Backend",
            "Inference Latency (ms)",
            "inference_mean",
        )

        self.save_bar_chart(
            "total_latency.png",
            "Mean Total Latency by Backend",
            "Total Latency (ms)",
            "total_mean",
        )

        self.save_bar_chart(
            "fps_comparison.png",
            "Mean FPS by Backend",
            "FPS",
            "fps_mean",
        )

        self.save_bar_chart(
            "memory_comparison.png",
            "Mean Process Memory by Backend",
            "Process Memory (MB)",
            "memory_mean",
        )

        self.save_variability_chart()
        self.save_trial_chart()

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