import numpy as np
import pandas as pd
import os
import random
from gplearn.genetic import SymbolicRegressor
from sympy import symbols, diff, sympify
import re

# Get the absolute path of the current file
current_file_path = os.path.abspath(__file__)

# Get the directory containing the current file
current_dir = os.path.dirname(current_file_path)

# Change the working directory to the directory of the current file
os.chdir(current_dir)

if not os.path.exists('../results/fold'):
    os.makedirs('../results/fold')

def parse_expression(expr):
    """
    将字符串形式的S表达式如 `mul(sub(a, b), div(c, d))` 解析成树形结构
    """
    # 去除空格
    expr = expr.replace(" ", "")

    # 递归解析
    def helper(expr):
        # 如果是数字或字母（变量），直接返回
        if re.match(r'^[a-zA-Z0-9]+$', expr):
            return expr
        
        # 处理运算符和括号，找到第一个运算符
        for op in ["mul", "div", "add", "sub"]:
            if expr.startswith(op):
                # 找到匹配的运算符
                pattern = re.compile(rf"{op}\((.*)\)")
                match = pattern.match(expr)
                if match:
                    # 匹配到后，获取括号中的内容
                    inside = match.group(1)
                    # 分割内部表达式
                    split_expr = split_expressions(inside)
                    # 递归解析每部分
                    return (op, *[helper(sub_expr) for sub_expr in split_expr])
        return expr

    # 分割括号内的多个表达式
    def split_expressions(inside):
        balance = 0
        start = 0
        split_exprs = []
        for i, char in enumerate(inside):
            if char == ',' and balance == 0:
                split_exprs.append(inside[start:i])
                start = i + 1
            elif char == '(':
                balance += 1
            elif char == ')':
                balance -= 1
        split_exprs.append(inside[start:])
        return split_exprs

    return helper(expr)

def s_expr_to_math(expr):
    """将S-表达式转换为标准的数学表达式"""
    # 如果表达式是原子值（数字或变量），直接返回
    if isinstance(expr, (int, float, str)):
        return str(expr)

    # 如果表达式是一个列表，且包含操作符
    if isinstance(expr, tuple) and len(expr) >= 2:
        op = expr[0]  # 操作符
        operands = expr[1:]  # 操作数

        # 根据操作符选择对应的数学符号
        if op == "add":
            op_str = "+"
        elif op == "sub":
            op_str = "-"
        elif op == "mul":
            op_str = "*"
        elif op == "div":
            op_str = "/"
        elif op == "sin":
            op_str = "sin"
        elif op == "cos":
            op_str = "cos"
        elif op == "exp":
            op_str = "exp"
        else:
            op_str = op  # 如果是其他操作符，保留原始符号

        # 将操作数递归地转换为数学表达式
        operand_strs = [s_expr_to_math(operand) for operand in operands]
        
        # 对于加法、减法、乘法和除法，添加括号
        return f"({operand_strs[0]} {op_str} {operand_strs[1]})"

    return str(expr)

def diff_f(expression_f, x_t, b_t):

    X0 = symbols('X0')
    X1 = symbols('X1')

    f = sympify(expression_f)

    df_dx = diff(f, X0)

    print(df_dx)

    df_dx_value = df_dx.subs({X0: x_t, X1: b_t})

    return df_dx_value

length = 1000

train_length = 500 # Length of the rolling window
test_length = length - train_length

# input dimension of x
d = 1
# size of linear part of feature vector (x1,...,xn,b)
dlin = d + 1
# total size of feature vector: linear + nonlinear
dtot = dlin

df_tseries = pd.read_csv('../data/fold/fold_train.csv')
keep_col_tseries = ['traj_x','b','Time','rrate']
new_f_tseries = df_tseries[keep_col_tseries]
values_tseries = new_f_tseries.values

x_tseries = np.array(values_tseries[:,0])
b_tseries = np.array(values_tseries[:,1])
tseries = np.array(values_tseries[:,2])
rrate_tseries = np.array(values_tseries[:,3])

rand_seed = 2
np.random.seed(rand_seed)
random.seed(rand_seed)

predict_traj_x = list()
estimate_rrate = list()

# create an array to hold the linear part of the feature vector
x = np.zeros((dlin,length))

# fill in the linear part of the feature vector for all times
for j in range(length):
    x[0:d,j]=x_tseries[j]

i = 0 # count

for winstart in range(test_length):

    x[-1,winstart:winstart + train_length] = np.arange(1, 1 + train_length * 0.01, 0.01)
    x_train = x[:,winstart:winstart + train_length].T

    y_train = (x[:d,winstart+1 : winstart + train_length+1] - x[:d,winstart : winstart + train_length]).T
    y_train = y_train.reshape(-1)

    model = SymbolicRegressor(population_size=5000, generations=20, tournament_size=20,
                          stopping_criteria=0.01, init_depth=(2, 6), init_method='half and half', const_range=(1, 10),
                          function_set=['add','mul','sin'],
                          metric='mean absolute error', random_state=rand_seed)

    model.fit(x_train, y_train)

    x_input = x_train[-1,:].reshape(1, -1)
    
    delta_x = model.predict(x_input)

    next_x = x_input[0][0] + delta_x

    predict_traj_x.append(next_x)

    print(next_x)

    print(model._program)

    expression = s_expr_to_math(parse_expression(str(model._program)))

    print(expression)

    x_t = x_input[0][0]
    b_t = x_input[0][1]

    rrate = diff_f(expression, x_t, b_t)
    estimate_rrate.append(rrate)

    print(rrate)

    i += 1

    print(i)

dic_pred = {'Time':tseries[train_length - 1 : -1],'b_param':xb_tseries[train_length - 1 : -1][:,1],'next_x':xb_tseries[train_length:][:,0],'pred_x':predict_traj_x,
            'rrate':rrate_tseries[train_length - 1 : -1],'esti_rrate':estimate_rrate}

csv_out = pd.DataFrame(dic_pred)
csv_out.to_csv('../results/fold/FNN_lin_fold.csv',header = True)
