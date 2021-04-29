from Modules.PolarityModel.LRmodels import LRModel
from Modules.preprocess import load_data, preprocessing, vocab
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    inputs, outputs = load_data('Data/temp.csv')
    X_train, X_test, Y_train, Y_test = train_test_split(inputs, outputs, test_size=0.2, random_state=42)
    # inputs = preprocessing(inputs)
    model = LRModel()
    model.train(inputs, outputs)
    predicts = model.predict(X_test)

    tp, fp, fn, p, r, f1 = model.evaluate_pos(Y_test, predicts)
    print(f1)


