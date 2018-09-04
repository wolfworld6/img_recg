class PerformanceEvaluation:
    """
    算法性能评估
    :param TP: True Positive(真正)：将正类预测为正类数
    :param TN: True Negative(真负)：将负类预测为负类数
    :param FP: False Positive(假正)：将负类预测为正类数，误报
    :param FN: False Negative(假负)：将正类预测为负类数，漏报
    :param P: 预测为正类样本数
    :param N: 预测为负类样本数
    :return:
    """

    def __init__(self, tp, tn, fp, fn, p, n):
        self.__TP = tp
        self.__TN = tn
        self.__FP = fp
        self.__FN = fn
        self.__P = p
        self.__N = n

    def get(self):
        return self.__TP, \
               self.__TN, \
               self.__FP, \
               self.__FN, \
               self.__P, \
               self.__N

    def set(self, tp, tn, fp, fn, p, n):
        self.__TP = tp
        self.__TN = tn
        self.__FP = fp
        self.__FN = fn
        self.__P = p
        self.__N = n

    def get_accuracy(self):
        # 准确率
        acc = (self.__TP + self.__TN) / (self.__TP + self.__TN + self.__FP + self.__FN)
        return acc

    def get_error(self):
        # 错误率
        err = (self.__FP + self.__FN) / (self.__TP + self.__TN + self.__FP + self.__FN)
        return err

    def get_recall(self):
        # 灵敏度或召回率，所有正例中被分对的比率，衡量了分类器对正例的识别能力
        rec = self.__TP / self.__P
        return rec

    def get_specificity(self):
        # 特效度，所有负例中被分对的比例，衡量了分类器对负例的识别能力
        spec = self.__TN / self.__N
        return spec

    def get_precision(self):
        # 表示被分为正例的实例中实际为正例的比例
        p = self.__TP / (self.__TP + self.__FP)
        return p
