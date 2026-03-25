import os
import email
from email.utils import parsedate_to_datetime
import pandas as pd
import sys

# Ensure custom modules can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

enron_path = 'benchmarkers/data/input/raw/enron/'
target_folders = ['sent', 'sent_items', '_sent_mail']
output_txt = 'data/input/raw/edgelist/enron.txt'

# A list to hold all our valid edges before putting them in a dataframe
edges_list = []

print("Extracting emails... this might take a minute.")

for root, dirs, files in os.walk(enron_path):
    current_folder_name = os.path.basename(root).lower()
    
    if current_folder_name in target_folders:
        for file_name in files:
            file_path = os.path.join(root, file_name)
            
            try:
                with open(file_path, 'r', encoding='latin-1') as f:
                    msg = email.message_from_file(f)
                    
                    raw_date = msg.get('Date')
                    msg_from = msg.get('From')
                    raw_to = msg.get('To')
                    
                    if raw_date and msg_from and raw_to:
                        
                        msg_from = msg_from.strip().lower()
                        if '@enron.com' not in msg_from:
                            continue 
                            
                        try:
                            dt = parsedate_to_datetime(raw_date)
                            formatted_date = dt.strftime('%Y-%m-%d')
                        except Exception:
                            continue 
                        
                        clean_to_string = raw_to.replace('\n', '').replace('\r', '').replace('\t', '')
                        recipients = [r.strip().lower() for r in clean_to_string.split(',')]
                        
                        for recipient in recipients:
                            if recipient and '@enron.com' in recipient and recipient != msg_from:
                                edges_list.append({
                                    'from': msg_from,
                                    'to': recipient,
                                    'date': formatted_date
                                })
                                
            except Exception:
                pass

print("Extraction complete. Building DataFrame...")

df = pd.DataFrame(edges_list)

# =====================================================================
# Restrict to the "Core" Network (~150 nodes)
# =====================================================================
print("Restricting network to core mailbox owners...")
core_employees = set(df['from'].unique())
df = df[df['to'].isin(core_employees)].copy()

# =====================================================================
# Aggregate Daily Weights (The 'value' column)
# =====================================================================
print("Aggregating daily email counts to calculate 'value' weights...")
# This groups by sender, receiver, and date, then counts the number of emails 
# to create the 'value' column
df = df.groupby(['from', 'to', 'date']).size().reset_index(name='value')

df.sort_values(by='date', inplace=True)
df.reset_index(drop=True, inplace=True)

print("Mapping email addresses to integer IDs (0 to N-1)...")
unique_emails = pd.concat([df['from'], df['to']]).unique()
email_to_id = {email: i for i, email in enumerate(unique_emails)}

df['from'] = df['from'].map(email_to_id)
df['to'] = df['to'].map(email_to_id)

# Save to CSV format but keep the .txt extension, including the header
os.makedirs(os.path.dirname(output_txt), exist_ok=True)
df[['from', 'to', 'date', 'value']].to_csv(output_txt, sep=',', header=True, index=False)

print(f"Success! {len(df)} sorted, mapped, and weighted edges saved to {output_txt}")
print(f"Total Unique Nodes Processed: {len(unique_emails)}")