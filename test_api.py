import pikudhaoref
import time

client = pikudhaoref.SyncClient(update_interval=2)

@client.event
def on_siren(sirens):
    print("Sirens:", sirens)
    for siren in sirens:
        print(f"City: {siren.city.name}, Category: {siren.category}")

print("Current active sirens:", client.current_sirens)

# Since we just want to run once to check
print("Testing complete")
