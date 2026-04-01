import pandas as pd
import os
import re

# Function to convert b.out files into readable form
from convert_bifdata import convert_bifdata 
#convert_bifdata函数将AUTO跑一个分岔参数的模型转化为python的字典
#{'type':bif_type, 'value':bif_val, 'bif_param':bif_param(如a1), 'branch_vals':xVals}

time = list()
bif_type_list = list()
bif_value_list = list()

# 指定文件夹路径
folder_path = 'output_auto'

# 列出文件夹中的所有文件
files = os.listdir(folder_path)

# 使用正则表达式提取文件名中的数字
numbers = []
for filename in files:
    match = re.search(r'b.out_(\d+)', filename)
    if match:
        numbers.append(int(match.group(1)))

numbers.sort()

df = pd.read_csv('../../results/trans/trans.csv')

bif_pars = df['b'].values

initial_theta = df['initial_theta'].values
delta_theta = df['delta_theta'].values
end_theta = initial_theta + 499*delta_theta

for ii in numbers:

    out = convert_bifdata(f'output_auto/b.out_{ii}')
    bif_type = out['type']
    bif_value = (out['value'] - end_theta[ii])*2.9412 / (end_theta[ii] - initial_theta[ii]) + bif_pars[ii]

    bif_type_list.append(bif_type)
    bif_value_list.append(bif_value)
    time.append(ii+500)

dic_bif = {'Time':time, 'bif_type':bif_type_list, 'bif_value':bif_value_list}
csv_out = pd.DataFrame(dic_bif)
csv_out.to_csv('../../results/trans/auto_output.csv',header = True)

