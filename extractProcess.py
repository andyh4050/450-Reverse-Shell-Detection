import json
import os
import pandas as pd


src_file= '2022-01-20-21-50-53.out'

with open(os.path.join('ReverseShellETWData1',src_file)) as fin:
    data = [json.loads(line) for line in fin]

df = pd.DataFrame.from_records(data)
processes = list(df.groupby(['processID']))

for item in processes:
    item[1].to_json(os.path.join('ReverseShellETWData1',src_file[:-4],str(item[0])+'.json'), orient='records', lines=True)
