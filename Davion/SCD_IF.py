# -*- coding:utf-8 -*-

from sklearn.ensemble import IsolationForest
# 自定义模块
from performance import *
from load import *

'''
   @author:bqFirst
   date:2018-8p

'''


if __name__ == '__main__':

    rng = np.random.RandomState(0)  # 42
    pklf = "./feature.pkl"
    X_train, X_test, y_train, y_test, outliers_fraction, unused = load(pklf)

    # fit the model
    clf = IsolationForest(random_state=rng)
    clf.fit(X_train)
    scores_pred = clf.decision_function(X_train)

    X_predict = X_test
    y_predict = y_test
    times = np.arange(-1.5, -0.5, 0.1)
    # times = [-2.8]
    for t in times:
        valid = clf.predict(X_predict, t)
        # 计算预测正确率
        TP = 0  # 样本为正，预测结果为正
        TN = 0  # 样本为负，预测结果为正
        FP = 0  # 样本为负，预测结果为负,误报
        FN = 0  # 样本为正，预测结果为负,漏报
        P = 0  # 正类样本数
        N = 0  # 负类样本数

        for i in range(len(valid)):
            if valid[i] == 1:  # 预测为正类
                if valid[i] == y_predict[i]:  # 实际为正类
                    P = P + 1
                    TP = TP + 1
                else:  # 实际为负类
                    FP = FP + 1
                    N = N + 1
            else:  # 预测为负类
                if valid[i] == y_predict[i]:  # 实际为负类
                    TN = TN + 1
                    N = N + 1
                else:  # 实际为正类
                    P = P + 1
                    FN = FN + 1

        per_eva = PerformanceEvaluation(TP, TN, FP, FN, P, N)

        print("t: %f" % t,
              "正确率为：%f " % (per_eva.get_accuracy()),
              "召回率、正常图像识别率：%d/%d = %f" % (TP, P, per_eva.get_recall()),
              " 特效度、异常图像识别率：%d/%d = %f" % (TN, N, per_eva.get_specificity())
              )
