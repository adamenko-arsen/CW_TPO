"""
plot_results.py - побудова графіків для курсової роботи.
Зчитує benchmark_results.csv і генерує три графіки:
  - plot_time.png    - час виконання vs кількість потоків (log-scale)
  - plot_speedup.png - коефіцієнт прискорення vs кількість потоків
  - plot_size.png    - вплив розміру матриці на ефективність паралелізації

Запуск:
  pip install matplotlib
  python plot_results.py
"""
import csv
import os
import sys

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("Помилка: не встановлено matplotlib.")
    print("Встановіть: pip install matplotlib")
    sys.exit(1)


def find_csv():
    candidates = [
        "benchmark_results.csv",
        "../benchmark_results.csv",
        "../../benchmark_results.csv",
        os.path.join(os.path.dirname(__file__), "benchmark_results.csv"),
        os.path.join(os.path.dirname(__file__), "..", "benchmark_results.csv"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)
    return None


def read_data(csv_path):
    """Парсить CSV у словник {dataset: {size, sequential_ms, times, speedups}}."""
    data = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ds = row["dataset"]
            data[ds] = {
                "size": int(row["size"]),
                "sequential_ms": float(row["sequential_ms"]),
                "times": {},
                "speedups": {},
            }
            for key, value in row.items():
                if key.startswith("parallel_") and key.endswith("_ms"):
                    workers = int(key[len("parallel_"):-len("_ms")])
                    data[ds]["times"][workers] = float(value)
                elif key.startswith("speedup_"):
                    workers = int(key[len("speedup_"):])
                    data[ds]["speedups"][workers] = float(value)
    return data


def plot_time(data, out_path):
    """Час виконання vs кількість потоків (логарифмічна шкала по Y)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    styles = {
        "small":  ("o-", "#1f77b4"),
        "medium": ("s-", "#ff7f0e"),
        "large":  ("^-", "#d62728"),
        "xlarge": ("D-", "#2ca02c"),
    }

    for ds_name, ds_data in data.items():
        if ds_name not in styles: continue
        marker, color = styles[ds_name]
        workers = sorted(ds_data["times"].keys())
        times = [ds_data["times"][w] for w in workers]
        label = f"{ds_name} (N={ds_data['size']})"
        ax.plot(workers, times, marker, label=label, color=color,
                linewidth=2, markersize=8)
        ax.axhline(y=ds_data["sequential_ms"], color=color, linestyle=":", alpha=0.4)

    ax.set_xlabel("Кількість робочих потоків", fontsize=12)
    ax.set_ylabel("Час виконання, мс (логарифмічна шкала)", fontsize=12)
    ax.set_title("Час виконання паралельного алгоритму\nна матрицях різного розміру", fontsize=13)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks([1, 2, 4, 8, 16, 32])
    ax.set_xticklabels([1, 2, 4, 8, 16, 32])
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(loc="best", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Збережено: {out_path}")


def plot_speedup(data, out_path):
    """Коефіцієнт прискорення vs кількість потоків."""
    fig, ax = plt.subplots(figsize=(10, 6))
    styles = {
        "small":  ("o-", "#1f77b4"),
        "medium": ("s-", "#ff7f0e"),
        "large":  ("^-", "#d62728"),
        "xlarge": ("D-", "#2ca02c"),
    }

    for ds_name, ds_data in data.items():
        if ds_name not in styles: continue
        marker, color = styles[ds_name]
        workers = sorted(ds_data["speedups"].keys())
        speedups = [ds_data["speedups"][w] for w in workers]
        ax.plot(workers, speedups, marker, label=ds_name, color=color,
                linewidth=2, markersize=8)

    ax.axhline(y=1.2, color="red", linestyle=":", alpha=0.6, label="Ціль ТЗ: 1,2x")

    ax.set_xlabel("Кількість робочих потоків", fontsize=12)
    ax.set_ylabel("Коефіцієнт прискорення (S = T_seq / T_par)", fontsize=12)
    ax.set_title("Залежність коефіцієнта прискорення\nвід кількості потоків", fontsize=13)
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, 16, 32])
    ax.set_xticklabels([1, 2, 4, 8, 16, 32])
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(loc="best", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Збережено: {out_path}")


def plot_size(data, out_path):
    """Стовпчаста діаграма: прискорення в залежності від розміру матриці."""
    targets = ["small", "medium", "large", "xlarge"]
    available = [t for t in targets if t in data]
    if len(available) < 2:
        print(f"  Пропущено plot_size: потрібно >= 2 з {targets}, наявні: {available}")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    workers_set = sorted(data[available[0]]["speedups"].keys())
    width = 0.2
    x_positions = list(range(len(workers_set)))
    colors = {
        "small":  "#1f77b4",
        "medium": "#ff7f0e",
        "large":  "#d62728",
        "xlarge": "#2ca02c",
    }

    for i, ds_name in enumerate(available):
        offset = (i - (len(available) - 1) / 2) * width
        speedups = [data[ds_name]["speedups"][w] for w in workers_set]
        bars = ax.bar(
            [x + offset for x in x_positions],
            speedups, width,
            label=f"{ds_name} (N={data[ds_name]['size']})",
            color=colors.get(ds_name, "gray"),
        )
        for bar, val in zip(bars, speedups):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                    f"{val:.1f}x", ha="center", fontsize=8)

    ax.set_xlabel("Кількість робочих потоків", fontsize=12)
    ax.set_ylabel("Коефіцієнт прискорення", fontsize=12)
    ax.set_title("Вплив розміру матриці на ефективність паралелізації", fontsize=13)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(workers_set)
    ax.axhline(y=1.0, color="gray", linestyle="-", alpha=0.4)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="best", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Збережено: {out_path}")


def main():
    print("─" * 50)
    print("  Побудова графіків для курсової")
    print("─" * 50)

    csv_path = find_csv()
    if not csv_path:
        print()
        print("ПОМИЛКА: не знайдено benchmark_results.csv.")
        print("Запустіть бенчмарк через CLI спочатку (опція 4).")
        sys.exit(1)

    data = read_data(csv_path)
    out_dir = os.path.dirname(csv_path)

    plot_time(data, os.path.join(out_dir, "plot_time.png"))
    plot_speedup(data, os.path.join(out_dir, "plot_speedup.png"))
    plot_size(data, os.path.join(out_dir, "plot_size.png"))

    print()
    print("  Готово! Графіки збережено.")


if __name__ == "__main__":
    main()
