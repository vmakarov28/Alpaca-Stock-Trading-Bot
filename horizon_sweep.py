import subprocess
import re
import pandas as pd
from colorama import Fore, Style

horizons = [14, 21, 28, 42]
attempts_per_horizon = 2   # Change to 5 later if you want more reliable stats

results = []

print(f"{Fore.CYAN}=== HORIZON SWEEP STARTING ==={Style.RESET_ALL}")
print(f"Testing horizons: {horizons} | {attempts_per_horizon} attempts each\n")

for h in horizons:
    print(f"{Fore.YELLOW}{'='*25} TESTING HORIZON = {h} bars {'='*25}{Style.RESET_ALL}\n")
    
    for attempt in range(1, attempts_per_horizon + 1):
        print(f"\n{Fore.MAGENTA}--- Attempt {attempt}/{attempts_per_horizon} for horizon {h} ---{Style.RESET_ALL}\n")
        
        cmd = [
            "python", "/mnt/c/Users/aipla/Downloads/alpaca_neural_bot_v9.9.98.py",
            "--backtest", "--force-train",
            "--horizon", str(h)
        ]
        
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT, 
            text=True, 
            bufsize=1, 
            universal_newlines=True
        )
        
        output = []
        for line in process.stdout:
            print(line, end='', flush=True)   # ← This prints everything live
            output.append(line)
        
        process.wait()
        
        stdout_text = ''.join(output)
        
        # Parse metrics
        total_return = re.search(r"Total Return \(%\).*?(\-?\d+\.\d+)", stdout_text)
        trades = re.search(r"Trades.*?(\d+)", stdout_text)
        accuracy = re.search(r"Accuracy \(%\).*?(\d+\.\d+)", stdout_text)
        
        tr = float(total_return.group(1)) if total_return else 0.0
        trd = int(trades.group(1)) if trades else 0
        acc = float(accuracy.group(1)) if accuracy else 0.0
        
        results.append({
            "Horizon": h,
            "Attempt": attempt,
            "Total Return (%)": tr,
            "Trades": trd,
            "Accuracy (%)": acc
        })

# Final table
df = pd.DataFrame(results)
print(f"\n{Fore.GREEN}=== FINAL HORIZON COMPARISON ==={Style.RESET_ALL}")
print(df.to_string(index=False))

best = df.loc[df['Total Return (%)'].idxmax()]
print(f"\n{Fore.CYAN}BEST HORIZON: {best['Horizon']} bars ({best['Total Return (%)']:.2f}% return, {best['Trades']} trades){Style.RESET_ALL}")