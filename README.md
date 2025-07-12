# Alpaca Stock Trading Bot

Alpaca Neural Bot v6.7 is an advanced AI-powered stock trading bot that leverages neural networks to predict market trends and execute trades via the Alpaca API. Built with PyTorch for GPU acceleration (optimized for NVIDIA RTX 5080), it incorporates technical indicators (via TA-Lib), sentiment analysis (using Transformers), and risk management features like trailing stops, drawdown limits, and volatility filters. The bot supports both backtesting on historical data (from 2015 to the current date) and live paper trading.

## Key capabilities

Neural Network Prediction: Uses a Conv1D + LSTM model to generate buy/sell signals based on historical and real-time data.
Sentiment Analysis: Integrates DistilBERT for news sentiment scoring.
Backtesting: Simulates trades with transaction costs, ATR-based stops, and performance metrics (Sharpe ratio, max drawdown).
Live Trading: Executes market orders during open hours, with email notifications for trades and summaries.
Multi-Symbol Support: Trades multiple stocks (e.g., SPY, MSFT, AAPL) in parallel using multiprocessing.
GPU Acceleration: Fully utilizes CUDA for faster training on RTX 5080.
Free Tier Compatible: Designed for Alpaca's free API tier, with retry logic for rate limits.

***This bot is for educational and testing purposes only. Use paper trading to avoid financial risk.***

## Features

Data Fetching: Retrieves 15-minute bars from Alpaca API, cached for efficiency.
Indicators: Calculates RSI, MACD, ATR, ADX, Volatility, BBands, Stoch, and more using TA-Lib.
Model Training: Trains per-symbol models with early stopping, data augmentation (noise), and Xavier initialization.
Risk Management: Enforces max drawdown, position sizing based on ATR/risk percentage, trailing stops.
Notifications: Sends email updates for trades, failures, and summaries via Gmail SMTP.
Backtest Mode: Computes total return, Sharpe ratio, max drawdown, win rate per symbol.
Live Mode: Runs every 15 minutes during market hours, with countdown timer.
Force Train: Option to retrain models instead of using cache.
Error Handling: Retries API calls, validates data, logs trades.

## Prerequisites

Drivers: Install the latest nvidia drivers from nvdia app along with Cuda 12.8 and cuDNN.
Hardware: NVIDIA GPU (Confirmed to work propperly on RTX 5080 but other 50 series and 40 series card are likley to work). At least 16GB VRAM for efficient training.
Operating System: Windows Subsystem for Linux (WSL2) on Windows 10/11, or native Ubuntu 22.04+.
Alpaca Account: Free account with paper trading enabled. Get API keys from Alpaca Dashboard.
Gmail Account: For email notifications (enable "Less secure app access" or use app password).
Internet: Stable connection for API calls (no VPN recommended to avoid rate limits).


# Installation Steps

Follow these steps in order to set up the environment. All commands are for WSL/Ubuntu terminal.

### Step 1: Set Up WSL (Windows Users Only)

If you're on Windows, enable WSL2 for Linux-based setup. Open PowerShell as Administrator and install Windows Subsystem for Linux:
'''
wsl --install
wsl --install -d Ubuntu
wsl --set-default-version 2
sudo apt update && sudo apt upgrade -y
'''

Restart your PC if prompted.

### Troubleshooting:
If WSL installation fails, enable "Virtual Machine Platform" in Windows Features (Control Panel > Programs > Turn Windows features on/off).
Check WSL version: wsl --list --verbose. Ensure Ubuntu is on version 2.

## Step 2: Install pyenv for Python Management

Pyenv allows isolated Python environments.

### Install dependencies for Pyenv:
Use this command:

    sudo apt install -y build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev python3-openssl git


Install pyenv:

    curl https://pyenv.run | bash

Add to shell profile (add to ~/.bashrc or ~/.zshrc). Open the file with the nano text editor with this command:

    nano ~/.bashrc
Then add the new paths to the file.
> export PATH="$HOME/.pyenv/bin:$PATH"
> 
> eval "$(pyenv init --path)"
> 
> eval "$(pyenv virtualenv-init -)"
> 
Press Cntrl + O to save and Cntrl + X to exit the file.
Now back in the command line refresh the file with:

    source ~/.bashrc
Verify Pyenv was setup correcty with:

    pyenv --version

### Troubleshooting:
If pyenv not found, restart terminal or run exec $SHELL.
Error with dependencies: Re-run sudo apt install command.

### Step 3: Set Up Python Environment

Use Python 3.10.12 (compatible with dependencies).
Install Python 3.10:

    pyenv install 3.10.12
Create virtual environment:

    pyenv virtualenv 3.10.12 pytorch_env
Activate the virtual envoriment (You will also need this command everytime you want to run the program after exiting:

    pyenv activate pytorch_env
Upgrade pip:

    pip install --upgrade pip

Troubleshooting:

    Installation takes time; be patient.
    If build fails, ensure all dependencies from Step 1 are installed.


Step 4: Install CUDA Toolkit in WSL

Open your WSL terminal (e.g., Ubuntu): Run wsl in Command Prompt or search for "Ubuntu" in the Start menu.
Update the package list:

    sudo apt update && sudo apt upgrade -y.'
    
Install the CUDA toolkit for WSL-Ubuntu:
Add the NVIDIA CUDA repository: Follow the instructions at https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0.
Select "WSL-Ubuntu" > "2.0" > "x86_64" and copy the provided commands:

        wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
        sudo dpkg -i cuda-keyring_1.1-1_all.deb
        sudo apt-get update
        sudo apt-get install cuda-toolkit-12-6  # Replace 12-6 with the latest version, e.g., 12-5 if needed
Verify CUDA installation: Run `nvcc --version` in WSL. It should display the CUDA version.

Step 3: Install cuDNN in WSL

    Download cuDNN from the NVIDIA Developer website: https://developer.nvidia.com/rdp/cudnn-download.
        Sign up for an NVIDIA Developer account if you don't have one (free).
        Select cuDNN for CUDA 12.x (matching your toolkit version) and "Linux" (x86_64).
        Download the tar file (e.g., cudnn-linux-x86_64-9.x.x.xx_cuda12-archive.tar.xz).
    In WSL, extract and install cuDNN:
        Copy the downloaded file to WSL (e.g., via /mnt/c/Users/YourUsername/Downloads/).
        Run:
        text

    tar -xvf cudnn-linux-x86_64-9.x.x.xx_cuda12-archive.tar.xz
    sudo cp cudnn-*-archive/include/cudnn*.h /usr/local/cuda/include
    sudo cp -P cudnn-*-archive/lib/libcudnn* /usr/local/cuda/lib64
    sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*

Update your environment: Add to ~/.bashrc (run nano ~/.bashrc):
text

    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
        Save and run source ~/.bashrc.

Step 4: Verify Installation and Run the Program

    In WSL, install PyTorch with CUDA support: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 (use cu121 for CUDA 12.1; adjust if your version differs).
    Test CUDA: Run python -c "import torch; print(torch.cuda.is_available())". It should return True.
    Run your program: python your_script.py --backtest (ensure your PyTorch environment is activated if using one).

Troubleshooting

    If CUDA is not detected: Check nvidia-smi in WSL for GPU info.
    Errors with versions: Ensure CUDA toolkit, cuDNN, and PyTorch match (e.g., all for CUDA 12.x).
    For detailed guides: Refer to NVIDIA's CUDA on WSL user guide (linked above) or PyTorch installation docs at https://pytorch.org/get-started/locally/.



    

Step 5: Install PyTorch with confirm CUDA Support

For RTX 50 & 40 series GPU acceleration we will PyTorch for training the AI model:

    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

Verify PyTorch, CUDA, and GPU detection:

    python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
Expected: Shows PyTorch version (e.g., 2.4.1+cu128), True, and "NVIDIA GeForce RTX _ _ _ _".

Troubleshooting:

    False for CUDA: Ensure NVIDIA drivers are installed (Step 8).
    Wrong device: Verify nvidia-smi shows RTX 5080.
    Installation error: Check PyTorch index URL for CUDA version compatibility.

Step 5: Install TA-Lib

TA-Lib requires building from source for technical indicators.

    Install dependencies:
    text

sudo apt install -y build-essential wget
Download and build TA-Lib:
text
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib
./configure --prefix=/usr
make
sudo make install
cd ..
rm -rf ta-lib ta-lib-0.4.0-src.tar.gz
Install Python wrapper:
text

    pip install TA-Lib==0.4.32

Troubleshooting:

    Build error: Ensure build-essential is installed. If configure fails, check for missing libraries (e.g., sudo apt install libncurses5-dev).
    Import error: Verify with python -c "import talib; print(talib.__version__)".

Step 6: Install Other Dependencies

Install the remaining packages with exact versions for compatibility.

    Install dependencies:
    text

pip install alpaca-py==0.28.0 transformers==4.45.2 pandas==2.2.3 numpy==1.26.4 scikit-learn==1.5.2 tenacity==9.0.0 tqdm==4.66.5 colorama==0.4.6 protobuf==5.28.3
Verify imports:
text

    python -c "import torch, numpy, pandas, alpaca, transformers, sklearn, talib, tenacity, tqdm, colorama; print('All dependencies imported successfully')"

Troubleshooting:

    Conflicts: Use --no-deps for problematic packages (e.g., pip install alpaca-py==0.28.0 --no-deps).
    Protobuf error: Ensure protobuf==5.28.3 (for Transformers compatibility).
    Import error: Reinstall the package (e.g., pip install --force-reinstall transformers==4.45.2).


Step 8: Configure API Keys and Email

    Replace placeholders in CONFIG (lines 147-148, 153-156) with your values:
        ALPACA_API_KEY and ALPACA_SECRET_KEY from Alpaca.
        EMAIL_SENDER, EMAIL_PASSWORD, EMAIL_RECEIVER for Gmail.
    Save the script.

Running the Script

    Activate environment:
    text

pyenv activate pytorch_env
Run backtest with force-train:
text
python /mnt/c/Users/aipla/Downloads/alpaca_neural_bot_v6.7.py --backtest --force-train
Run live trading:
text

    python /mnt/c/Users/aipla/Downloads/alpaca_neural_bot_v6.7.py

Troubleshooting

    Rate Limit Error ("too many requests"): Add time.sleep(1) after bars = client.get_stock_bars(request).df in fetch_data (line 287). Increase to 2 seconds if persists.
    Missing Headers (longintrepr.h): Ensure apt-get install -y python3.12-dev ran successfully. Verify with find /usr/include -name longintrepr.h.
    Wheel Not Supported: Use Python 3.12-compatible wheels. For aiohttp, try pip install aiohttp==3.9.5 if 3.8.1 fails.
    CUDA Not Available: Check nvidia-smi. Reinstall drivers/CUDA. Verify with python -c "import torch; print(torch.cuda.is_available())".
    Dependency Conflicts: Use --no-deps for alpaca-py (e.g., pip install alpaca-py==0.28.0 --no-deps). Uninstall conflicting packages (e.g., pip uninstall aiohttp).
    TA-Lib Import Error: Rebuild TA-Lib and reinstall wrapper. Verify with python -c "import talib; print(talib.__version__)".
    API Key Error: Ensure keys are valid. Test with python -c "from alpaca.trading.client import TradingClient; client = TradingClient('YOUR_KEY', 'YOUR_SECRET', paper=True); print(client.get_account())".
    Model Training Hangs: Reduce batch size in CONFIG (line 75) to 16 if GPU memory is low. Monitor with nvidia-smi.
    Backtest IndexError: Clear cache (rm -rf ./cache) and re-run to fetch data up to today.
    Email Failure: Ensure Gmail app password is used (not regular password). Test SMTP with a simple script.

If issues persist, check trades.log for details or open an issue on the GitHub repo.
License

GNU Lesser General Public License v2.1. See LICENSE for details.
Author

Vladimir Makarov

GitHub: vmakarov28/Alpaca-Stock-Trading-Bot
