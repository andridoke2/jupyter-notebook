import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb

df = pd.read_csv('train.csv')
cleanData = df.copy()

def changeToNol(val):
  return 0 if pd.isna(val) else val
for field in df.columns:
  cleanData[field] = df[field].apply(changeToNol)
print(cleanData)

# define new value
defining_value = cleanData.copy()
dict_value = []
def transform_data(val):
  # print(val)
  if(val==0):
    return 0
  if(dict_value.count(val) > 0):
    _index = dict_value.index(val)+1
  else :
    _index = len(dict_value)+1
    dict_value.append(val)
  return _index

for field in ['Alley']:
  dict_value = []
  defining_value[field] = cleanData[field].apply(transform_data)

print(defining_value['Alley'])

sub = df[['LotFrontage','LotArea','SalePrice','MSSubClass','MasVnrArea']]
print(sub)
sub = sub.dropna()
print(sub)

sb.pairplot(sub[['SalePrice','LotFrontage','LotArea','MSSubClass','MasVnrArea']],diag_kind = 'kde')