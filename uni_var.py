import pandas as pd


for data in ['ETTh1', 'ETTm1', 'ILI']:

    print(data)
    df = pd.read_csv(f'datasets/csv/{data}.csv')

    x = df['date']
    y = df['OT']

    df = pd.DataFrame({'date':x, 'OT':y})
    df.to_csv(f'datasets/csv/{data}.csv', index=False)

print('wth')
df = pd.read_csv('datasets/csv/WTH.csv')

x = df['date']
y = list(df['DryBulbFarenheit'])

df = pd.DataFrame({'date':x, 'OT':y})
df.to_csv('datasets/csv/WTH.csv', index=False)

print('traffic')
df = pd.read_csv('datasets/csv/Traffic.csv')

x = df['date']
y = df['0']

df = pd.DataFrame({'date':x, 'OT':y})
df.to_csv('datasets/csv/Traffic.csv', index=False)

df = pd.read_csv('datasets/csv/ECL.csv')

x = df['date']
y = df['MT_000']

df = pd.DataFrame({'date':x, 'OT':y})
df.to_csv('datasets/csv/ECL.csv', index=False)