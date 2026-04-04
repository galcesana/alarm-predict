# Deployment Plan: Israel Missile Alarm Predictor

Because the Home Front Command (Pikud HaOref) APIs block foreign IP addresses to prevent DDoS and scraping, **you cannot deploy this repository to standard free cloud hosts** like Render, Heroku, or PythonAnywhere. The server executing the container *must* be physically located in Israel.

Here is the exact layout to deploy this freely.

## Prerequisites

1. **A Cloud Server with an Israeli IP:**
   - **AWS EC2:** Create a free tier account and launch an Ubuntu `t3.micro` instance in the **Tel Aviv (il-central-1)** region. (Free for 12 months).
   - **Oracle Cloud:** Create an account and select **Jerusalem (il-jerusalem-1)** as your home region. Launch an "Always Free" ARM Ampere compute instance.

2. **Docker:** The target server must have Docker and Docker Compose installed.

## Step 1: Clone Repository to Server

SSH into your freshly provisioned server and clone the repository:

```bash
ssh -i /path/to/key.pem ubuntu@<SERVER_IP>

# Ensure git is installed
sudo apt update && sudo apt install -y git

# Clone the repository
git clone <YOUR_GITHUB_REPO_URL> alarm-predict
cd alarm-predict
```

## Step 2: Configure Secrets
The repository is designed to keep secrets safely ignored. Create your `.env` file on the server.

```bash
nano .env
```
Paste your Bot token:
```env
TELEGRAM_BOT_TOKEN=1234567:YOUR_BOT_TOKEN_HERE
```
Save and exit (`Ctrl+O`, `Enter`, `Ctrl+X`).

## Step 3: Start the Subsystem via Docker

Use the provided docker-compose configuration. This configuration mounts volumes for your `models` and `data` directory so that subscriber lists and trained ML models survive server reboots.

```bash
sudo apt install -y docker.io docker-compose
sudo docker-compose up -d --build
```

## Step 4: Train the ML Model
Because the pre-trained weights are ignored by git (to keep the repo clean), the newly spun-up server has an "untrained" model. 

Enter the docker container and force a training run on the latest historical data:

```bash
# This fetches the historical CSV and runs the Bayesian+XGBoost pipeline
sudo docker exec -it alarm-predict-bot python main.py --train
```

## Step 5: Subscribe Users

Open Telegram and send `/start` to your bot.
The container will detect your command, add you to `data/subscribers.json` (which is safely mounted to the host server), and begin pushing real-time predictions directly to your device.
