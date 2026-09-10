"""Generate a simple historical-quantile baseline using only challenge data."""

import csv
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
QUANTILES = (0.25, 0.50, 0.75, 0.95)


def quantiles(values):
    values = sorted(values)
    result = []
    for q in QUANTILES:
        position = (len(values) - 1) * q
        lower = int(position)
        upper = min(lower + 1, len(values) - 1)
        result.append(values[lower] + (values[upper] - values[lower]) * (position - lower))
    return result


def main():
    by_pair = defaultdict(list)
    by_product = defaultdict(list)
    all_values = []
    with (ROOT / "data/sales_history.csv").open(newline="") as source:
        for row in csv.DictReader(source):
            value = max(0.0, float(row["sales_count"]))
            by_pair[row["store_id"], row["product_id"]].append(value)
            by_product[row["product_id"]].append(value)
            all_values.append(value)
    pair_forecasts = {key: quantiles(values) for key, values in by_pair.items()}
    product_forecasts = {key: quantiles(values) for key, values in by_product.items()}
    fallback = quantiles(all_values)
    output = ROOT / "predictions_baseline.csv"
    count = 0
    with (ROOT / "data/test.csv").open(newline="") as source, output.open("w", newline="") as target:
        writer = csv.writer(target)
        writer.writerow(["id", "sales_count_p25", "sales_count_p50", "sales_count_p75", "sales_count_p95"])
        for row in csv.DictReader(source):
            forecast = pair_forecasts.get(
                (row["store_id"], row["product_id"]),
                product_forecasts.get(row["product_id"], fallback),
            )
            writer.writerow([row["id"], *(f"{value:.6f}" for value in forecast)])
            count += 1
    print(f"Wrote {count} predictions to {output}")


if __name__ == "__main__":
    main()
