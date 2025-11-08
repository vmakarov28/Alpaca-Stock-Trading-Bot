# Alpaca Stock Trading Bot

Alpaca Neural Bot v10.0.0 is an advanced AI-powered stock trading bot that uses a CNN-LSTM neural network built with PyTorch to predict market trends and execute trades via the Alpaca API, optimized for GPU acceleration on NVIDIA RTX 5080. It incorporates technical indicators from TA-Lib (such as RSI, MACD, ATR, and ADX), sentiment analysis via Hugging Face Transformers, and robust risk management features including ATR-based stops, trailing stops, drawdown limits, volatility filters, and minimum holding periods. The bot supports comprehensive backtesting with metrics like Sharpe ratio, max drawdown, Monte Carlo simulations, and buy-and-hold benchmarks on historical data from 2015 to the current date (November 08, 2025), alongside live paper trading with email notifications and multiprocessing for efficient training across multiple symbols.

## Key capabilities
- **Neural Network Prediction:** Employs a CNN-LSTM model (Conv1D layers followed by LSTM) to predict future price directions over LOOK_AHEAD_BARS (7 bars) based on 30-timestep sequences of 23 features from historical and real-time data.

- Model Training: Trains the CNN-LSTM model per symbol using Adam optimizer (learning rate 0.001), BCEWithLogitsLoss, early stopping, and ReduceLROnPlateau scheduler (patience 5, factor 0.5) over customizable epochs with customizable batch size; supports data augmentation via noise addition; in backtest mode, enables an automated retraining cycle (up to X attempts) that retrains until performance criteria are met, then selects and copies the best attempt's models/scalers.

- **Backtesting:** Simulates trades across multiple symbols between a specifyable data, and today. Includes transaction costs, ATR-based stops/profits, min holding periods, performance metrics (Sharpe ratio, max drawdown, win rates, accuracies), Monte Carlo simulations (50,000 runs), buy-and-hold benchmarks. 

- **Live Trading:** Executes market orders during open market hours with email notifications for individual trades and daily summaries.

- **Multi-Symbol Support:** Handles trading for multiple stocks (e.g., SPY, MSFT, AAPL, AMZN, NVDA, META, GOOGL) with parallel multiprocessing (4 workers) for model training and independent backtesting per symbol.

- **GPU Acceleration:** Leverages PyTorch with CUDA for accelerated training and inference on RTX 5080, including memory management via torch.cuda.empty_cache() after parallel sessions.

- **Sentiment Analysis:** Has a framework to utilizes DistilBERT (distilbert-base-uncased-finetuned-sst-2-english) via Hugging Face Transformers for news sentiment scoring, currently defualts to netural

- **Free Tier Compatible:** Optimized for Alpaca's free API tier with retry logic (3 attempts, 1-second delay), data caching (24-hour expiry), and rate-limit handling via tenacity.

***This bot is for educational and testing purposes only. Use paper trading to avoid financial risk.***


# Installation Steps
Follow these steps in order to set up the environment. *Note: All commands after steo 2 are for WSL/Ubuntu terminal.*

## Prerequisites

- Drivers: Install the latest nvidia drivers from nvdia app along with Cuda 12.8 and cuDNN.
- Hardware: NVIDIA GPU with at least 16GB VRAM for efficient training. (Confirmed to work propperly on RTX 5080 but other 50 series and 40 series card are likley to work).
- Operating System: Windows Subsystem for Linux (WSL2) on Windows 10/11, or native Ubuntu 22.04+.
- Alpaca Account: Free account with paper trading enabled. Get API keys from Alpaca Dashboard.
- Gmail Account: For email notifications (enable "Less secure app access" or use app password).
- Internet: Stable connection for API calls (no VPN recommended to avoid rate limits).



### Step 1: Set Up WSL (Windows Users Only)

If you're on Windows, enable WSL2 for Linux-based setup. Open PowerShell as Administrator and install Windows Subsystem for Linux:
```
wsl --install
wsl --install -d Ubuntu
wsl --set-default-version 2
sudo apt update && sudo apt upgrade -y
```

Restart your PC after successful installation.

### Troubleshooting:
- If WSL installation fails, enable "Virtual Machine Platform" in Windows Features (Control Panel > Programs > Turn Windows features on/off).
- Check WSL version: wsl --list --verbose. Ensure Ubuntu is on version 2.

## Step 2: Install pyenv for Python Management
*********\*\***From now on the rest of the steps will be completed in wsl*\*\***********

Pyenv allows isolated Python environments.

### Install dependencies for Pyenv:
Use this command:

    sudo apt install -y build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev python3-openssl git


Install pyenv:

    curl https://pyenv.run | bash

Add to shell profile (add to ~/.bashrc or ~/.zshrc). Open the file with the nano text editor with this command:

    nano ~/.bashrc
    
Then add the new paths to the end of the file.
> export PATH="$HOME/.pyenv/bin:$PATH"
> 
> eval "$(pyenv init --path)"
> 
> eval "$(pyenv virtualenv-init -)"
> 
Press `Cntrl + O` then `Enter` to save. Then use `Cntrl + X` to exit the file.
Now back in the command line refresh the file with:

    source ~/.bashrc
Verify Pyenv was setup correcty with:

    pyenv --version
Expected output: `pyenv 2.6.7` (Latest as of 8/24/25)
### Troubleshooting:
- If "Unable to locate package llvm" use:
  
        sudo add-apt-repository universe
        sudo apt update
  then re-attempt the command:

      sudo apt install -y make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev python3-openssl git
- If pyenv not found, restart terminal or run exec $SHELL.
- Error with dependencies: Re-run sudo apt install command.

## Step 3: Set Up Python Environment

Use Python 3.10.12 (compatible with dependencies).
Install Python 3.10:

    pyenv install 3.10.12
Create virtual environment:

    pyenv virtualenv 3.10.12 pytorch_env
Activate the virtual envoriment (You will also need this command everytime you want to run the program after exiting:

    pyenv activate pytorch_env
Upgrade pip:

    pip install --upgrade pip

Troubleshooting
- If you get ``pyenv activate' requires Pyenv and Pyenv-Virtualenv to be loaded into your shell.
Check your shell configuration and Pyenv and Pyenv-Virtualenv installation instructions.`` When using the command `pyenv activate pytorch_env` Run each line individually:

        echo 'eval "$(pyenv init -)"' >> ~/.bashrc
        eval "$(pyenv init -)"
        source ~/.bashrc
  then reattempt `pyenv activate pytorch_env`
- If build fails, ensure all dependencies from Step 1 are installed.
- Syntax Errors: Make sure all the paths are written properly with no mistakes.

## Step 4: Install CUDA Toolkit in WSL

Open your WSL terminal (e.g., Ubuntu): Run wsl in Command Prompt or search for "Ubuntu" in the Start menu.
Update the package list:

    sudo apt update && sudo apt upgrade -y
    
Install the CUDA toolkit for WSL-Ubuntu:
Add the NVIDIA CUDA repository: Follow the instructions at https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0.
> Select "WSL-Ubuntu" > "2.0" > "x86_64"

Then copy and run the provided commands:

        wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
        sudo dpkg -i cuda-keyring_1.1-1_all.deb
        sudo apt-get update
        sudo apt-get install cuda-toolkit-12-6  # Replace 12-6 with the latest version, e.g., 12-5 if needed
Add new paths to bash by running these commmands one by one:

        echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
        echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
        echo 'export PATH=/usr/lib/wsl/lib:$PATH' >> ~/.bashrc
        source ~/.bashrc

Verify CUDA installation: Run `nvcc --version` in WSL. It should display the CUDA version.

## Step 5: Install cuDNN in WSL

Download cuDNN from the NVIDIA Developer website: https://developer.nvidia.com/rdp/cudnn-download.
- You made need to sign up for an NVIDIA Developer account if you don't have one (free).
- Select cuDNN for CUDA 12.x (matching your toolkit version) and "Linux" (x86_64).
- Select Linux > x86_64 > Ubuntu > 24.04 (Yours may varry) > deb (local) > FULL
- Download the deb file (e.g., cudnn-local-repo-ubuntu2404-9.12.0_1.0-1_amd64.deb).
    In WSL, extract and install cuDNN:
        Copy the downloaded file to WSL (e.g., via /mnt/c/Users/YourUsername/Downloads/).
  
Run each command sequentially (**in pyenv**:
```
cd /mnt/c/Users/YOUR USER/Downloads
sudo dpkg -i cudnn-local-repo-ubuntu2404-9.12.0_1.0-1_amd64.deb
sudo cp /var/cudnn-local-repo-ubuntu2404-9.12.0/cudnn-local-*-keyring.gpg /usr/share/keyrings/
sudo apt update
sudo apt install -y libcudnn9-cuda-12 libcudnn9-dev-cuda-12 libcudnn9-static-cuda-12
echo 'export LD_LIBRARY_PATH=/usr/lib/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

### Troubleshooting
- Run:

      wget https://developer.download.nvidia.com/compute/cudnn/9.12.0/local_installers/cudnn-local-repo-ubuntu2404-9.12.0_1.0-1_amd64.deb
      sudo dpkg -i cudnn-local-repo-ubuntu2404-9.12.0_1.0-1_amd64.deb
      sudo cp /var/cudnn-local-repo-ubuntu2404-9.12.0/cudnn-*-keyring.gpg /usr/share/keyrings/
      sudo apt-get update
      sudo apt-get -y install cudnn
- If CUDA is not detected: Check nvidia-smi in WSL for GPU info.
- Errors with versions: Ensure CUDA toolkit, cuDNN, and PyTorch match (e.g., all for CUDA 12.x).
- For detailed guides: Refer to NVIDIA's CUDA on WSL user guide (linked above) or PyTorch installation docs at https://pytorch.org/get-started/locally/.



    

## Step 6: Install PyTorch with confirm CUDA Support

For RTX 50 & 40 series GPU acceleration we will PyTorch for training the AI model (use cu128 for CUDA 12.8; adjust if your version differs):

    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 

Verify PyTorch, CUDA, and GPU detection:

    python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
Expected: Shows PyTorch version (e.g., 2.4.1+cu128), True, and "NVIDIA GeForce RTX _ _ _ _".

### Troubleshooting


- False for CUDA: Ensure NVIDIA drivers are installed.
- Wrong device: Verify nvidia-smi shows RTX 5080.
- Installation error: Check PyTorch index URL for CUDA version compatibility.

## Step 7: Install TA-Lib

TA-Lib requires building from source for technical indicators.

Install dependencies:

    sudo apt install -y build-essential wget

Download and build TA-Lib:
Remove any previous build artifacts (optional, to clean)
    
    rm -rf ~/ta-lib ta-lib-0.4.0-src.tar.gz
Re-download and build with /usr/local prefix

    wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
    tar -xzf ta-lib-0.4.0-src.tar.gz
    cd ta-lib
    ./configure --prefix=/usr/local
    make
    sudo make install
    cd ~
    rm -rf ta-lib ta-lib-0.4.0-src.tar.gz
    
Set environment variables for pip to find TA-Lib

    export TA_INCLUDE_PATH=/usr/local/include
    export TA_LIBRARY_PATH=/usr/local/lib
    
Install the Python wrapper
    pip install TA-Lib==0.4.32

Install Python wrapper:

    pip install TA-Lib==0.4.32

Troubleshooting

- Build error: Ensure build-essential is installed. If configure fails, check for missing libraries (e.g., sudo apt install libncurses5-dev). Another option it to try and download a prebuilt wheel from the Ta-Lib repo.
- Import error: Verify with python -c "import talib; print(talib.__version__)".

## Step 8: Install Other Dependencies

Install the remaining packages with exact versions for compatibility.
Install dependencies:

    pip install alpaca-py==0.28.0 transformers==4.45.2 pandas==2.2.3 numpy==1.26.4 scikit-learn==1.5.2 tenacity==9.0.0 tqdm==4.66.5 colorama==0.4.6 protobuf==5.28.3

Verify imports:

    python -c "import torch, numpy, pandas, alpaca, transformers, sklearn, talib, tenacity, tqdm, colorama; print('All dependencies imported successfully')"

Troubleshooting:

    Conflicts: Use --no-deps for problematic packages (e.g., pip install alpaca-py==0.28.0 --no-deps).
    Protobuf error: Ensure protobuf==5.28.3 (for Transformers compatibility).
    Import error: Reinstall the package (e.g., pip install --force-reinstall transformers==4.45.2).


## Step 9: Configure API Keys and Email

- Replace placeholders in CONFIG (lines 147-148, 153-156) with your values:
- ALPACA_API_KEY and ALPACA_SECRET_KEY from Alpaca.
- EMAIL_SENDER, EMAIL_PASSWORD, EMAIL_RECEIVER for Gmail.
- Save the script.

## Step 10: Running the Script

Activate virtual environment:

    pyenv activate pytorch_env

Run backtest with force-train:

    python /mnt/c/Users/aipla/Downloads/alpaca_neural_bot_v6.7.py --backtest --force-train

Run live trading:

    python /mnt/c/Users/aipla/Downloads/alpaca_neural_bot_v6.7.py

Troubleshooting

- Rate Limit Error ("too many requests"): Add time.sleep(1) after bars = client.get_stock_bars(request).df in fetch_data (line 287). Increase to 2 seconds if persists.
- Missing Headers (longintrepr.h): Ensure apt-get install -y python3.12-dev ran successfully. Verify with find /usr/include -name longintrepr.h.
- Wheel Not Supported: Use Python 3.10-compatible wheels. For aiohttp, try `pip install aiohttp==3.9.5 if 3.8.1 fails`.
- CUDA Not Available: Check nvidia-smi. Reinstall drivers/CUDA. Verify with python -c "import torch; print(torch.cuda.is_available())".
- Dependency Conflicts: Use --no-deps for alpaca-py (e.g., `pip install alpaca-py==0.28.0 --no-deps`). Uninstall conflicting packages (e.g., `pip uninstall aiohttp`).
- TA-Lib Import Error: Rebuild TA-Lib and reinstall wrapper. Verify with `python -c "import talib; print(talib.__version__)"`.
- API Key Error: Ensure keys are valid. Test with `python -c "from alpaca.trading.client import TradingClient; client = TradingClient('YOUR_KEY', 'YOUR_SECRET', paper=True); print(client.get_account())"`.
- Model Training Hangs: Reduce batch size in CONFIG (line 75) to 16 if GPU memory is low. Monitor with nvidia-smi.
- Backtest IndexError: Clear cache with `rm -rf ./cache` and re-run to fetch data up to today.
- Email Failure: Ensure Gmail app password is used (not regular password). Test SMTP with a simple script.
Ensure all dependency files are installed:
    `pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128`
    `pip install --upgrade alpaca-py transformers pandas numpy scikit-learn ta-lib tenacity tqdm colorama protobuf==5.28.3`

If issues persist, check trades.log for details or open an issue on the GitHub repo.


License
GNU Lesser General Public License v2.1. See LICENSE for details.

Author: Vladimir Makarov

Most reacent change: 10/28/25

GitHub: vmakarov28/Alpaca-Stock-Trading-Bot
