import requests
import json
import time

TOKEN = "8374179702:AAERrCdyBrJAV11QKCOYMZYqO-6_CDx7Vr8"

def main():
    print("=" * 50)
    print("  Telegram Setup")
    print("=" * 50)
    print("\n1. Please go to Telegram and send a message (e.g. 'hello') to:")
    print("   @GoodMorningTelAvivBot\n")
    print("2. I am waiting for your message to capture your chat ID...")
    
    url = f"https://api.telegram.org/bot{TOKEN}/getUpdates"
    
    chat_id = None
    
    # Poll for 2 minutes
    for _ in range(24):
        try:
            response = requests.get(url, timeout=5)
            data = response.json()
            
            if data["ok"] and data["result"]:
                # Get the last message
                last_message = data["result"][-1]
                if "message" in last_message:
                    chat_id = last_message["message"]["chat"]["id"]
                    first_name = last_message["message"]["chat"].get("first_name", "User")
                    print(f"\n✅ Message received from {first_name}!")
                    break
        except Exception as e:
            pass
            
        time.sleep(5)
        
    if chat_id:
        print(f"✅ Your telegram chat ID is: {chat_id}")
        
        # Write to .env
        with open(".env", "w") as f:
            f.write(f"TELEGRAM_BOT_TOKEN={TOKEN}\n")
            f.write(f"TELEGRAM_CHAT_ID={chat_id}\n")
            
        print("\n✅ Saved credentials to .env file!")
        print("\nTest message is being sent now...")
        
        # Test it
        from src.telegram_utils import send_alert_message
        success = send_alert_message("✅ *Alarm Predictor connected!*\n\nYou will receive Tel Aviv missile alert probabilities here.")
        if success:
            print("✅ Check your Telegram! Your bot is ready.")
    else:
        print("\n❌ Timed out waiting for a message. Please run this script again and make sure you message the bot.")

if __name__ == "__main__":
    main()
