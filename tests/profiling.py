"""Benchmark ForecastData creation with FER data."""

import time

from forecast_evaluation import ForecastData


def bench(label, fn, n=3):
    """Run fn n times and print min/mean/max."""
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    print(f"{label:>40s}:  min={min(times):.3f}s  mean={sum(times) / n:.3f}s  max={max(times):.3f}s")


# --- Load raw data once (not timed) ---
fd_raw = ForecastData(load_fer=True)
outturns = fd_raw._raw_outturns.copy()
forecasts = fd_raw._raw_forecasts.copy()
print(f"Outturns: {len(outturns)} rows, Forecasts: {len(forecasts)} rows\n")

# --- Full load_fer ---
bench("ForecastData(load_fer=True)", lambda: ForecastData(load_fer=True))

# --- add_outturns only ---
bench("add_outturns", lambda: ForecastData(outturns_data=outturns))


# --- add_forecasts only ---
fd_base = ForecastData(outturns_data=outturns)
bench("add_forecasts", lambda: fd_base.copy().add_forecasts(forecasts, data_check=False))

# --- add_outturns + add_forecasts ---
def add_fc():
    fd = ForecastData(outturns_data=outturns)
    fd.add_forecasts(forecasts, data_check=False)


bench("add_outturns + add_forecasts", add_fc)
