import numpy as np
class twoLayerNN:
    def __init__(self, input_size, hidden_size, output_size, std = 1e-3):
        """
        随机初始化参数
        """
        self.W1 = std * np.random.rand(input_size,hidden_size)
        self.b1 = np.zeros(hidden_size)
        self.W2 = std * np.random.rand(hidden_size,output_size)
        self.b2 = np.zeros(output_size)
    
    def loadParams(self, path):
        """
        加载模型参数
        """
        try:
            paras = np.load(path)
            self.W1 = np.asarray(paras['W1'])
            self.W2 = np.asarray(paras['W2'])
            self.b1 = np.asarray(paras['b1'])
            self.b2 = np.asarray(paras['b2'])
        except:
            print("No models found. Please Train the network first.")
            exit()

    def computeLossandGrad(self, X, y, loss_only=False, reg=1e-3):
        """
        计算梯度和损失
        """
        N = X.shape[0]
        z1 = X.dot(self.W1) + self.b1
        a1 = np.maximum(z1, 0)
        z2 = z1.dot(self.W2) + self.b2
        expres = np.exp(z2)
        probs = expres / np.sum(expres, axis=1, keepdims=True)
        loss = -np.log(probs[range(N), y]).sum()/ N
        # 正则
        loss = loss + reg * (np.sum(self.W1 * self.W1) + np.sum(self.W2 * self.W2))
        # 如果只计算损失，则直接返回损失值
        if loss_only:
            return loss
            
        # Backward pass: compute gradients
        grads = {}
        dy = probs
        dy[range(N), y] -= 1
        dy /= N
        grads['W2'] = a1.T.dot(dy)+ reg * self.W2
        grads['b2'] = dy.sum(axis=0)
        da1 = dy.dot(self.W2.T)
        dz1 = np.where(z1 >= 0, da1, 0)
        grads['W1'] = X.T.dot(dz1) + reg * self.W1
        grads['b1'] = dz1.sum(axis=0)
        return loss, grads

    def train(self, X, y, Xval, yval, lr, learning_rate_decay, reg, epochs, batch_size):
        """
        训练模型
        """
        N = X.shape[0]
        iterations = int(np.floor(N / batch_size))
        train_loss_history = []
        val_loss_history = []
        train_acc_history = []
        val_acc_history = []

        for epoch in range(epochs):
            for iteration in range(iterations):
                # 随机采样
                batch_idxes = np.random.choice(N, batch_size)
                Xbatch = X[batch_idxes, :]
                ybatch = y[batch_idxes]
                # Compute loss and gradients using the current minibatch
                loss, grads = self.computeLossandGrad(Xbatch, y=ybatch, loss_only=False, reg=reg)
                # update the parameters
                self.W1 -= lr * grads['W1']
                self.b1 -= lr * grads['b1']
                self.W2 -= lr * grads['W2']
                self.b2 -= lr * grads['b2']
            # Check loss
            trainloss = self.computeLossandGrad(X, y=y, loss_only=True, reg=reg)
            valloss = self.computeLossandGrad(Xval, y=yval, loss_only=True, reg=reg)
            train_loss_history.append(trainloss)
            val_loss_history.append(valloss)
            # Check accuracy
            train_acc = self.evaluation(X, y)
            val_acc = self.evaluation(Xval, yval)
            train_acc_history.append(train_acc)
            val_acc_history.append(val_acc)
            print('Epoch %d / %d: train_loss %f \t val_loss %f \t train_acc %f \t val_acc %f, ' % (
                epoch, epochs, np.round(trainloss, 4), np.round(valloss, 4), 
                train_acc, val_acc))
            # Decay learning rate
            lr *= learning_rate_decay

        return {
            'train_loss_history': train_loss_history,
            'val_loss_history': val_loss_history,
            'train_acc_history': train_acc_history,
            'val_acc_history': val_acc_history,
            }

    def predict(self, X):
        """
        模型预测
        """
        scores = np.maximum(X.dot(self.W1) + self.b1, 0).dot(self.W2) + self.b2
        ypred = np.argmax(scores, axis=1)
        return ypred

    def evaluation(self, X, y):
        """
        模型评估：准确率
        """
        return (self.predict(X) == y).mean()