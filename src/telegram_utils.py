import os
import time
import requests
import json
import threading
from pathlib import Path
from dotenv import load_dotenv

import logging
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_URL = f"https://api.telegram.org/bot{TOKEN}"

DATA_DIR = Path(__file__).parent.parent / "data"
SUBSCRIBERS_FILE = DATA_DIR / "subscribers.json"

def _load_subscribers() -> set:
    if not SUBSCRIBERS_FILE.exists():
        return set()
    try:
        with open(SUBSCRIBERS_FILE, "r") as f:
            return set(json.load(f))
    except json.JSONDecodeError:
        return set()

def _save_subscribers(subs: set):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(SUBSCRIBERS_FILE, "w") as f:
        json.dump(list(subs), f)

def send_alert_message(text: str):
    """
    Broadcast the formatted prediction to all subscribed Telegram users.
    """
    if not TOKEN:
        logger.debug("Telegram token not found in .env, skipping.")
        return False
        
    subs = _load_subscribers()
    if not subs:
        logger.debug("No Telegram subscribers found.")
        return False
        
    formatted_text = f"<pre><code>{text}</code></pre>"
    success_count = 0
    
    for chat_id in subs:
        payload = {
            "chat_id": chat_id,
            "text": formatted_text,
            "parse_mode": "HTML"
        }
        try:
            response = requests.post(f"{TELEGRAM_URL}/sendMessage", json=payload, timeout=5)
            response.raise_for_status()
            success_count += 1
        except Exception as e:
            logger.error(f"Failed to send to {chat_id}: {e}")
            
    logger.info(f"Broadcasted Telegram alert to {success_count}/{len(subs)} users.")
    return success_count > 0

def _send_welcome(chat_id: int, first_name: str):
    text = f"✅ Shalom {first_name}!\n\nYou are now subscribed. I will send you Tel Aviv alarm probability predictions in real-time as soon as a warning is issued.\n\nSend /stop at any time to unsubscribe."
    payload = {"chat_id": chat_id, "text": text}
    try:
        requests.post(f"{TELEGRAM_URL}/sendMessage", json=payload, timeout=5)
    except Exception as e:
        logger.error(f"Welcome msg failed: {e}")

def _send_goodbye(chat_id: int):
    payload = {"chat_id": chat_id, "text": "❌ Unsubscribed. You will no longer receive alerts."}
    try:
        requests.post(f"{TELEGRAM_URL}/sendMessage", json=payload, timeout=5)
    except Exception:
        pass

def _bot_polling_loop():
    logger.info("Started Telegram bot polling for /start commands...")
    last_update_id = 0
    
    while True:
        try:
            # timeout=30 means long-polling for instant responses
            url = f"{TELEGRAM_URL}/getUpdates?timeout=30&offset={last_update_id + 1}"
            response = requests.get(url, timeout=35)
            
            if response.status_code == 200:
                data = response.json()
                if data.get("ok") and data.get("result"):
                    for update in data["result"]:
                        last_update_id = update["update_id"]
                        
                        if "message" in update and "text" in update["message"]:
                            msg = update["message"]
                            chat_id = msg["chat"]["id"]
                            text = msg["text"].strip()
                            first_name = msg["chat"].get("first_name", "User")
                            
                            subs = _load_subscribers()
                            
                            if text.startswith("/start"):
                                if chat_id not in subs:
                                    subs.add(chat_id)
                                    _save_subscribers(subs)
                                    logger.info(f"New Telegram subscriber: {first_name} ({chat_id})")
                                    _send_welcome(chat_id, first_name)
                                    
                            elif text.startswith("/stop"):
                                if chat_id in subs:
                                    subs.remove(chat_id)
                                    _save_subscribers(subs)
                                    logger.info(f"User unsubscribed: {first_name} ({chat_id})")
                                    _send_goodbye(chat_id)
        except Exception as e:
            # Keep trying even on network errors
            time.sleep(2)

def start_bot_polling():
    """Start the Telegram polling loop in a background thread."""
    if not TOKEN:
        logger.warning("No TELEGRAM_BOT_TOKEN found. Telegram features disabled.")
        return
        
    thread = threading.Thread(target=_bot_polling_loop, daemon=True)
    thread.start()
