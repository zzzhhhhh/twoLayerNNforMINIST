from model import twoLayerNN
import pandas as pd
import numpy as np
import os

def loadTestset(test_path):
    test = pd.read_csv(test_path, header=None)
    test = test.values
    Xtest, ytest = test[:, 1:] /255.0 * 0.99 + 0.01, test[:, 0]
    return Xtest, ytest

def confusion_matrix(ytest, ypred):
    cm = np.zeros((10, 10), int)
    for i in range(len(ytest)):
        cm[ypred[i], ytest[i]] += 1
    return cm    

def precision(label, confusion_matrix):
    row = confusion_matrix[label, :]
    return np.round(confusion_matrix[label, label] / row.sum(), 5)

def recall(label, confusion_matrix):
    col = confusion_matrix[:, label]
    return np.round(confusion_matrix[label, label] / col.sum(), 5)

if __name__ == "__main__":
    # 读取数据
    testPath = "./dataset/mnist_test.csv"
    Xtest, ytest = loadTestset(testPath)
    print("Test set:", Xtest.shape)

    # 初始化模型并加载参数
    input_size = 28 * 28
    hidden_size = 256
    num_classes = 10
    filename = 'model.npz'
    predictModel = twoLayerNN(input_size, hidden_size, num_classes)
    predictModel.loadParams(os.path.join(os.curdir, filename))

    # 预测
    ypred = predictModel.predict(Xtest)
    pd.DataFrame(ypred).to_csv('test_pred.txt', header=None, index=False)
    print("Prediction Done! Results in 'test_pred.txt'!")

    # 评估
    print("================ Evaluation ================")
    test_acc = predictModel.evaluation(Xtest, ytest)
    print("Test accuracy:", test_acc)
    print("Confusion Matrix:")
    cfm = confusion_matrix(ytest, ypred)
    print(pd.DataFrame(cfm))
    print("Precision:")
    precisions = [precision(i, cfm) for i in range(10)]
    print(precisions)
    print("Recall:")
    recalls = [recall(i, cfm) for i in range(10)]
    print(recalls)
