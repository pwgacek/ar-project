#!/usr/bin/env python3
"""
Rysuje wykres magnetyzacji dla dwóch metod (rng, subnets):
- Średnia magnetyzacja z 5 powtórzeń dla każdego sweepa
- "Tunel" jako odchylenie standardowe wokół średniej (±std)

Oczekiwane pliki wejściowe w katalogu:
  magnetization_rep0_rng.txt ... magnetization_rep4_rng.txt
  magnetization_rep0_subnets.txt ... magnetization_rep4_subnets.txt

Każdy plik powinien mieć dwie kolumny (whitespace):
  sweep  magnetization
lub w przypadku subnets:  sweep  suma_spinów (wtedy użyj opcji --N-subnets).

Przykład:
  python plot_magnetization.py --dir T_1.0 --output mag_rng_vs_subnets.png --N-subnets 1048576
"""

from __future__ import annotations

import argparse
import pathlib
import statistics
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt


def parse_line(line: str) -> Tuple[int, float]:
	parts = line.strip().split()
	if len(parts) < 2:
		raise ValueError(f"Nieprawidłowy wiersz: '{line.strip()}'")
	return int(parts[0]), float(parts[1])


def read_series(path: pathlib.Path, divide_by: float | None = None) -> Dict[int, float]:
	"""Czyta plik i zwraca mapę sweep->wartość (ew. dzieloną przez divide_by)."""
	data: Dict[int, float] = {}
	with path.open("r", encoding="utf-8") as f:
		for raw in f:
			raw = raw.strip()
			if not raw or raw.startswith("#"):
				continue
			s, v = parse_line(raw)
			if divide_by and divide_by != 0:
				v = v / divide_by
			data[s] = v
	if not data:
		raise ValueError(f"Plik {path} nie zawiera danych")
	return data


def aggregate_method(
	files: List[pathlib.Path],
	divide_by: float | None = None,
) -> Tuple[List[int], List[float], List[float]]:
	"""Zwraca (sweeps, mean, std) z wielu powtórzeń dla danej metody."""
	if not files:
		raise ValueError("Brak plików do agregacji")

	# Wczytaj wszystkie powtórzenia do listy map sweep->val
	reps: List[Dict[int, float]] = [read_series(p, divide_by) for p in files]

	# Wspólne sweepy (na wszelki wypadek)
	common_sweeps = set(reps[0].keys())
	for r in reps[1:]:
		common_sweeps &= set(r.keys())
	sweeps = sorted(common_sweeps)

	means: List[float] = []
	stds: List[float] = []
	for s in sweeps:
		vals = [r[s] for r in reps]
		mean = statistics.fmean(vals)
		std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
		means.append(mean)
		stds.append(std)

	return sweeps, means, stds


def find_files(base_dir: pathlib.Path, method: str) -> List[pathlib.Path]:
	return sorted(base_dir.glob(f"magnetization_rep*_" + method + ".txt"))


def plot_methods(
	sweeps_a: List[int], mean_a: List[float], std_a: List[float], label_a: str,
	sweeps_b: List[int], mean_b: List[float], std_b: List[float], label_b: str,
	title: str, output: pathlib.Path,
) -> None:
	fig, ax = plt.subplots(figsize=(10, 5))

	ax.plot(sweeps_a, mean_a, label=label_a, color="#1f77b4")
	ax.fill_between(sweeps_a, [m - s for m, s in zip(mean_a, std_a)], [m + s for m, s in zip(mean_a, std_a)],
					color="#1f77b4", alpha=0.2, label=f"{label_a} ±std")

	ax.plot(sweeps_b, mean_b, label=label_b, color="#ff7f0e")
	ax.fill_between(sweeps_b, [m - s for m, s in zip(mean_b, std_b)], [m + s for m, s in zip(mean_b, std_b)],
					color="#ff7f0e", alpha=0.2, label=f"{label_b} ±std")

	ax.set_xlabel("Sweep")
	ax.set_ylabel("Magnetyzacja")
	ax.set_title(title)
	ax.grid(True, linestyle=":", alpha=0.6)
	ax.legend()

	fig.tight_layout()
	fig.savefig(output, dpi=300)
	print(f"Zapisano wykres do: {output}")


def parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(
		description="Wykres średniej magnetyzacji z tunelami (±std) dla metod rng i subnets",
		formatter_class=argparse.ArgumentDefaultsHelpFormatter,
	)
	p.add_argument("--dir", type=pathlib.Path, default=pathlib.Path("."), help="folder z plikami magnetyzacji")
	p.add_argument("--output", type=pathlib.Path, default=pathlib.Path("mag_rng_vs_subnets.png"), help="plik wyjściowy PNG")
	p.add_argument("--title", type=str, default="Magnetyzacja: losowy wybór spinów vs podsieci", help="tytuł wykresu")
	p.add_argument("--N-subnets", type=float, default=None, help="jeśli subnets przechowuje sumę spinów, podaj N aby przeliczyć na magnetyzację")
	return p.parse_args()


def main() -> None:
	args = parse_args()
	base_dir = args.dir
	if not base_dir.exists():
		raise SystemExit(f"Folder nie istnieje: {base_dir}")

	files_rng = find_files(base_dir, "rng")
	files_sub = find_files(base_dir, "subnets")
	if not files_rng:
		raise SystemExit(f"Nie znaleziono plików rng w {base_dir}")
	if not files_sub:
		raise SystemExit(f"Nie znaleziono plików subnets w {base_dir}")

	sweeps_r, mean_r, std_r = aggregate_method(files_rng)
	sweeps_s, mean_s, std_s = aggregate_method(files_sub, divide_by=args.N_subnets)

	plot_methods(
		sweeps_r, mean_r, std_r, "rng",
		sweeps_s, mean_s, std_s, "subnets",
		args.title, args.output,
	)


if __name__ == "__main__":
	main()
