import subprocess
import re
import random
from pathlib import Path
import multiprocessing as mp
from tqdm import tqdm
import os

BOT_SCRIPT = "/mnt/c/Users/aipla/Downloads/alpaca_neural_bot_v10.00.00.py"

# ==================== GRID SEARCH SPACE (smart & realistic) ====================
PARAM_GRID = {
    'CONFIDENCE_THRESHOLD':      [0.45, 0.48, 0.50, 0.52],
    'PREDICTION_THRESHOLD_BUY':  [0.55, 0.58, 0.60, 0.62],
    'PREDICTION_THRESHOLD_SELL': [0.38, 0.40, 0.42, 0.45],
    'RSI_BUY_THRESHOLD':         [32, 35, 38, 40, 42],
    'RSI_SELL_THRESHOLD':        [65, 68, 70, 72, 75],
    'ADX_TREND_THRESHOLD':       [15, 18, 20, 22, 25],
    'MAX_VOLATILITY':            [25.0, 28.0, 32.0, 35.0, 38.0],
    'PREDICTION_TEMPERATURE':    [0.45, 0.50, 0.55, 0.60],
}

NUM_TRIALS = 170           # ← tuned for exactly ~10 hours at your current speed
MAX_WORKERS = 2
TIMEOUT_SECONDS = 1800     # 30 minutes (very safe)

def run_backtest(config_overrides: dict) -> dict:
    env = os.environ.copy()
    env["OPTIMIZER_MODE"] = "true"

    cmd = ["python", BOT_SCRIPT, "--backtest"]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT_SECONDS, env=env)

    output = result.stdout + result.stderr

    final_match = re.search(r"Final Best Portfolio Value\s*:\s*\$?([\d,]+\.?\d*)", output)
    final_value = float(final_match.group(1).replace(',', '')) if final_match else 100000.0

    sharpe_match = re.search(r"Sharpe Ratio.*?([\d.]+)", output)
    sharpe = float(sharpe_match.group(1)) if sharpe_match else 0.0

    trades_match = re.search(r"Trades.*?:?\s*(\d+)", output)
    trades = int(trades_match.group(1)) if trades_match else 0

    winrate_match = re.search(r"Win Rate.*?:?\s*([\d.]+)", output)
    winrate = float(winrate_match.group(1)) if winrate_match else 0.0

    return {
        "final_value": final_value,
        "sharpe": sharpe,
        "trades": trades,
        "winrate": winrate,
        "config": config_overrides.copy()
    }

def main():
    print("🚀 Starting 10-Hour Threshold Optimizer...")
    print(f"Testing {NUM_TRIALS} combinations (~10 hours with 2 workers)\n")

    all_combos = [{k: random.choice(v) for k, v in PARAM_GRID.items()} for _ in range(NUM_TRIALS)]

    results = []
    with mp.Pool(processes=MAX_WORKERS) as pool:
        for res in tqdm(pool.imap_unordered(run_backtest, all_combos), total=NUM_TRIALS):
            results.append(res)

    results.sort(key=lambda x: x["final_value"], reverse=True)

    print("\n" + "="*80)
    print("🏆 TOP 10 CONFIGURATIONS (10-hour run complete)")
    print("="*80)

    for i, r in enumerate(results[:10], 1):
        print(f"\n{i:2d}. Final Value: ${r['final_value']:,.2f} | "
              f"Sharpe: {r['sharpe']:.3f} | Trades: {r['trades']} | WinRate: {r['winrate']:.1f}%")
        print("   Config:")
        for k, v in r['config'].items():
            print(f"      {k}: {v}")

    best = results[0]
    best_path = Path("best_thresholds.py")
    with open(best_path, "w") as f:
        f.write("# BEST THRESHOLDS FROM 10-HOUR OPTIMIZER RUN\n")
        f.write("BEST_CONFIG = {\n")
        for k, v in best['config'].items():
            f.write(f"    '{k}': {v},\n")
        f.write("}\n")

    print(f"\n✅ Best config saved to: {best_path}")
    print(f"   Final Portfolio Value: ${best['final_value']:,.2f}")

if __name__ == "__main__":
    main()