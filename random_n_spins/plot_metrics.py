#!/usr/bin/env python3
"""
Rysuje porównanie przyspieszenia, efektywności i współczynnika sekwencyjnego
(Karp-Flatt) dla dwóch zestawów pomiarów zapisanych w plikach CSV.

Użycie:
	python plot_metrics.py run1.csv run2.csv --labels "GPU" "CPU" --output wykres.png
"""

from __future__ import annotations

import argparse
import csv
import pathlib
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


def load_metrics(csv_path: pathlib.Path) -> List[Tuple[int, float]]:
	"""Zwraca listę (n_proc, time) posortowaną po liczbie procesów."""
	rows: List[Tuple[int, float]] = []
	with csv_path.open(newline="", encoding="utf-8") as handle:
		reader = csv.DictReader(handle)
		for row in reader:
			try:
				n_proc = int(row["n_proc"])
				wall_time = float(row["time"])
			except KeyError as exc:
				raise ValueError(
					f"Brakuje kolumny {exc} w pliku {csv_path}"
				) from exc
			rows.append((n_proc, wall_time))
	if not rows:
		raise ValueError(f"Plik {csv_path} nie zawiera żadnych danych")
	rows.sort(key=lambda item: item[0])
	return rows


def compute_scaling(series: List[Tuple[int, float]]) -> Dict[str, List[float]]:
	"""Wylicza metryki skalowania dla podanego szeregu czasów."""
	baseline = next((t for n, t in series if n == 1), None)
	if baseline is None:
		raise ValueError("Brakuje pomiaru dla n_proc=1 potrzebnego do referencji")

	n_procs: List[int] = []
	speedups: List[float] = []
	efficiencies: List[float] = []
	serial_fracs: List[float] = []

	for n, wall_time in series:
		speedup = baseline / wall_time
		efficiency = speedup / n
		# Karp-Flatt: e(p) = (1/S_p - 1/p) / (1 - 1/p)
		serial = 0.0 if n == 1 else (1.0 / speedup - 1.0 / n) / (1.0 - 1.0 / n)

		n_procs.append(n)
		speedups.append(speedup)
		efficiencies.append(efficiency)
		serial_fracs.append(serial)

	return {
		"n_procs": n_procs,
		"speedup": speedups,
		"efficiency": efficiencies,
		"serial_frac": serial_fracs,
	}


def plot_comparison(
	series_a: Dict[str, List[float]],
	series_b: Dict[str, List[float]],
	label_a: str,
	label_b: str,
	title: str,
	output: pathlib.Path,
) -> None:
	fig, axes = plt.subplots(1, 3, figsize=(13, 4))

	ideal_x = series_a["n_procs"]

	axes[0].plot(ideal_x, ideal_x, "k--", label="idealna")
	axes[0].plot(series_a["n_procs"], series_a["speedup"], "o-", label=label_a)
	axes[0].plot(series_b["n_procs"], series_b["speedup"], "s-", label=label_b)
	axes[0].set_xlabel("Liczba procesów")
	axes[0].set_ylabel("Przyspieszenie")
	axes[0].set_title("Przyspieszenie")
	axes[0].grid(True, linestyle=":", alpha=0.6)
	axes[0].legend()

	axes[1].axhline(1.0, color="k", linestyle="--", label="idealna")
	axes[1].plot(series_a["n_procs"], series_a["efficiency"], "o-", label=label_a)
	axes[1].plot(series_b["n_procs"], series_b["efficiency"], "s-", label=label_b)
	axes[1].set_xlabel("Liczba procesów")
	axes[1].set_ylabel("Efektywność")
	axes[1].set_ylim(bottom=0)
	axes[1].set_title("Efektywność")
	axes[1].grid(True, linestyle=":", alpha=0.6)
	axes[1].legend()

	axes[2].axhline(0.0, color="k", linestyle="--", label="idealna")
	axes[2].plot(series_a["n_procs"], series_a["serial_frac"], "o-", label=label_a)
	axes[2].plot(series_b["n_procs"], series_b["serial_frac"], "s-", label=label_b)
	axes[2].set_xlabel("Liczba procesów")
	axes[2].set_ylabel("Serial fraction (Karp-Flatt)")
	axes[2].set_title("Serial fraction")
	axes[2].grid(True, linestyle=":", alpha=0.6)
	axes[2].legend()

	fig.suptitle(title)
	fig.tight_layout()
	fig.savefig(output, dpi=300)
	print(f"Zapisano wykres do {output}")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Porównanie metryk skalowania z dwóch plików CSV",
		formatter_class=argparse.ArgumentDefaultsHelpFormatter,
	)
	parser.add_argument("csv_a", type=pathlib.Path, help="pierwszy plik z metrykami")
	parser.add_argument("csv_b", type=pathlib.Path, help="drugi plik z metrykami")
	parser.add_argument(
		"--labels",
		nargs=2,
		metavar=("A", "B"),
		help="etykiety dla serii (domyślnie nazwy plików)",
	)
	parser.add_argument(
		"--output",
		type=pathlib.Path,
		default=pathlib.Path("metrics_comparison.png"),
		help="ścieżka do pliku wyjściowego",
	)
	parser.add_argument(
		"--title",
		default="Porównanie skalowania",
		help="tytuł wykresu",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()

	label_a, label_b = (
		args.labels if args.labels else (args.csv_a.stem, args.csv_b.stem)
	)

	series_a = compute_scaling(load_metrics(args.csv_a))
	series_b = compute_scaling(load_metrics(args.csv_b))

	plot_comparison(series_a, series_b, label_a, label_b, args.title, args.output)


if __name__ == "__main__":
	main()
