import pandas as pd
import os
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='lastfm', help='config files')
args, _ = parser.parse_known_args()

dataset = args.dataset

df = pd.read_csv(f"./processed_data/{dataset}.csv")
df = df[['ts','item','session_id','user']]
df = df.rename(columns = {'ts':'timestamp:float', 'item':'item_id:token', 'session_id':'session_id:token', 'user':'user_id:token'})

path = f"./processed_data/{dataset}"
if not os.path.exists(path):
    os.makedirs(path)

# Step 1: Filter out sessions with less than 3 interactions
df= df.groupby('session_id:token').filter(lambda x: len(x) >= 3) # 3850909 -> 3787863

# Step 2: Filter out users with less than 5 sessions from the previously filtered DataFrame
df = df.groupby('user_id:token').filter(lambda x: x['session_id:token'].nunique() >= 5) 

# statistics
num_items = df['item_id:token'].nunique()
num_sessions = df['session_id:token'].nunique()
avg_session_length = df.groupby('session_id:token').size().mean()
num_users = df['user_id:token'].nunique()
sessions_per_user = df.groupby('user_id:token')['session_id:token'].nunique().mean()

print(f"dataset: {dataset}")
print(f"no. of user: {num_users}")
print(f"no. of item: {num_items}")
print(f"no. of session: {num_sessions}")
print(f"avg. session length: {avg_session_length}")
print(f"avg. no. of sessions per user: {sessions_per_user}")
# import pdb; pdb.set_trace()
df.to_csv(path + f"/{dataset}.inter", sep='\t', index=False)
