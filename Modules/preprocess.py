import pandas as pd

from models import Input, Output


def load_data(path):
    inputs = []
    outputs = []
    df = pd.read_csv(path)
    # inputs = df.question_text.to_numpy()
    # print(inputs)
    for _, r in df.iterrows():
        t = r['question_text'].strip().lower()
        id = r['target']
        inputs.append(Input(t, id))
        score = r['target']
        outputs.append(Output(score))
    return inputs, outputs

def vocab(inputs):
    dicts = {}

    for i in inputs:
        print(i)
        for w in i.question_text.split():
            dicts[w] = 1

    with open('Data/vocab/vocabulary.txt', mode='w', encoding='utf-8') as f:
        f.write("\n".join(list(dicts.keys())))


def preprocessing(inputs):
    pass
