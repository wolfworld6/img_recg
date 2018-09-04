import numpy as np
from pyecharts import Line

file = "E:\GitRepository\img_recg\Davion\\result\LOF(5)(70norm).txt"
x = []
acc_y = []
recall_y = []
spe_y = []
with open(file, 'r') as f:
    lines = f.readlines()

    for line in lines:
        l = line.split(' ')
        x.append(str(round(float(l[1]), 2)))
        acc_y.append(round(float(l[2]), 4))
        recall_y.append(round(float(l[3]), 4))
        spe_y.append(round(float(l[4]), 4))

line = Line("异常分割点t-性能折线图")
line.add("正确率", x, acc_y, is_smooth=True)\
    .add("召回率", x, recall_y, is_smooth=True)\
    .add("特效度", x, spe_y, is_smooth=True)
print(x)
print(acc_y)
line.render("./performance.html")
