class Input:
    def __init__(self, question_text, qid):
        self.question_text = question_text
        self.qid = qid


class Output:
    def __init__(self, scores):
        self.scores = scores
