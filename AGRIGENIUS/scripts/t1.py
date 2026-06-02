import pandas as pd

# Load the files to see what columns we have
growers = pd.read_csv("data/growers.csv")
campaigns = pd.read_csv("data/whatsapp_campaign.csv")

print("--- GROWERS COLUMNS ---")
print(growers.columns.tolist())

print("\n--- WHATSAPP CAMPAIGN COLUMNS ---")
print(campaigns.columns.tolist())