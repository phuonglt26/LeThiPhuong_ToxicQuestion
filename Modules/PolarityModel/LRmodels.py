import pickle

import numpy as np
from sklearn.linear_model import LogisticRegression

from Modules.models import Model


# one-hot
from models import Output


class LRModel(Model):
    def __init__(self):
        self.vocab = []
        with open('Data/vocab/vocabulary.txt', encoding='utf-8') as f:
            self.vocab = [v.strip() for v in f]
        self.models = LogisticRegression()
        # RandomForestClassifier
        # LogisticRegression
        # MultinomialNB
        # KNeighborsClassifier
        # DecisionTreeClassifier
        # SVC

    def _represent(self, inputs):
        features = []
        for ip in inputs:
            _features = [1 if v in ip.question_text else 0 for v in
                         self.vocab]
            features.append(_features)
        # print(features)

        return np.array(features).astype(np.float)

    def train(self, inputs, outputs):
        """

        :param list of models.Input inputs:
        :param list of models.AspectOutput outputs:
        :return:
        """
        X = self._represent(inputs)
        ys = [output.scores for output in outputs]
        self.models.fit(X, ys)

    def save(self, path):
        # save the model to disk
        pickle.dump(self.models, open(path, 'wb'))

    def load(self, path):
        # load the model from disk
        model = pickle.load(open(path, 'rb'))
        self.models = model

    def predict(self, inputs):
        """
        :param inputs:
        :return:
        :rtype: list of models.AspectOutput
        """
        X = self._represent(inputs)
        outputs = []
        predicts = self.models.predict(X)
        for p in predicts:
            scores = p
            outputs.append(Output(scores))
        return outputs
    def evaluate_pos(self, y_test, y_predicts):
        tp = 0
        fp = 0
        fn = 0
        for g, p in zip(y_test, y_predicts):
            if g.scores == p.scores == 1:
                tp += 1
            elif g.scores == 1:
                fn += 1
            elif p.scores == 1:
                fp += 1
        if tp == 0 and fp == 0:
            print("khong bat duoc")
            p = 0
        else:
            p = tp / (tp + fp)
        r = tp / (tp + fn)
        if r == 0 and p == 0:
            f1 = 0
        else:
            f1 = 2 * p * r / (p + r)
        return tp, fp, fn, p, r, f1

    # def evaluate_neg(self, y_test, y_predicts):
    #     tp = 0
    #     fp = 0
    #     fn = 0
    #     for g, p in zip(y_test, y_predicts):
    #         if g.scores == p.scores == -1:
    #             tp += 1
    #         elif g.scores == -1:
    #             fn += 1
    #         elif p.scores == -1:
    #             fp += 1
    #     if tp == 0 and fp == 0:
    #         print("khong bat duoc")
    #         p = 0
    #     else:
    #         p = tp / (tp + fp)
    #     # if tp == 0 and fn == 0:
    #     #     r = 0
    #     # else:
    #     r = tp / (tp + fn)
    #     if r == 0 and p == 0:
    #         f1 = 0
    #     else:
    #         f1 = 2 * p * r / (p + r)
    #     return tp, fp, fn, p, r, f1
