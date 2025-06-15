import pandas as pd
df = pd.read_csv('hansard40000.csv')
df['party']  = df['party'].replace('Labour (Co-op)', 'Labour')
df = df[df.party != 'Speaker']
common_party = df['party'].value_counts().nlargest(4)
df = df[df['party'].isin(common_party.index)]
print(common_party)
print(df)
