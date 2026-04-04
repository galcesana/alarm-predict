# Alarm Predictor — Tel Aviv 🚀

A real-time machine learning system designed to predict exactly which of Tel Aviv's 4 alert zones will receive a siren when a general Gush Dan missile warning is issued.

It leverages a hybrid **Bayesian + XGBoost model** trained on historical Home Front Command (Pikud HaOref) data and polls the official API in real-time, completely bypassing Akamai WAF blocks.

## Key Features
- **Real-Time API Polling**: Directly monitors the Home Front Command endpoints using `pikudhaoref.py` to bypass 403 Forbidden blocks.
- **Dynamic Risk Machine Learning**: Uses an XGBoost classifier evaluating the geographical spread, size, and category of the warning.
- **Telegram Broadcasting**: Instantly pushing cleanly aligned percentage tables to any user who messages `/start` to the bot.
- **Autonomous Training**: A completely autonomous pipeline that fetches raw CSV data from external archives, groups cities into temporal "attack events", and recalibrates model weights automatically.

## Quick Start
```bash
# Clone and install
pip install -r requirements.txt

# Run the live monitor in the background
python main.py

# To force a full retraining pipeline:
python main.py --train
```

## Cloud Deployment
See `docs/deployment_plan.md` for specific instructions on how to rent an Israeli physical proxy server to bypass geolocks, map secrets silently, and run the subsystem indefinitely via Docker Compose.
