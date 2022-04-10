import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from model import twoLayerNN

def getMinistData(train_path, val_num):
    """
    对数据作归一化，并划分训练机、验证集和测试集
    """
    train = pd.read_csv(train_path, header=None)
    train = train.values
    np.random.shuffle(train)
    Xtrain, ytrain = train[val_num:, 1:] /255.0 * 0.99 + 0.01, train[val_num:, 0]
    Xval, yval = train[:val_num, 1:] /255.0 * 0.99 + 0.01, train[:val_num, 0]
    return Xtrain, ytrain, Xval, yval

def save(model, filename):
    np.savez_compressed(
        file = os.path.join(os.curdir, filename),
        W1 = model.W1,
        W2 = model.W2,
        b1 = model.b1,
        b2 = model.b2
    )

if __name__ == "__main__":
    # 得到训练集和验证集
    trainPath = "./dataset/mnist_train.csv"
    Xtrain, ytrain, Xval, yval = getMinistData(trainPath, val_num=10000)
    print("Train set:", Xtrain.shape)
    print("Val set:", Xval.shape)
    input_size = 28 * 28

    # the parameters
    hidden_size = 256
    num_classes = 10
    net = twoLayerNN(input_size, hidden_size, num_classes)

    # Train the network
    print("Start Training...")
    stats = net.train(Xtrain, ytrain, Xval, yval, lr=0.01, learning_rate_decay=0.95, reg=1e-5, epochs=30, batch_size=64)

    # save the model
    filename='model.npz'   
    save(net, filename)
    print("Done training! Model is saved as 'model.npz'!")

    print("Visualizing...")
    # 绘制损失曲线
    plt.plot(stats['train_loss_history'])
    plt.plot(stats['val_loss_history'])
    plt.title('Loss history')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(["Train","Val"])
    plt.savefig('loss.png')
    print("Loss curve is drawn in 'loss.png'!")

    # 绘制准确率曲线
    plt.plot(stats['train_acc_history'], label='train')
    plt.plot(stats['val_acc_history'], label='val')
    plt.title('Accuracy history')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(["Train","Val"])
    plt.savefig('accuracy.png')
    print("Accuracy curve is drawn in 'accuracy.png'!")
    
    # 可视化隐层参数
    W1 = net.W1
    W1 = W1.reshape(28, 28, -1).transpose(2, 0, 1)
    N, L = W1.shape[0], 28
    grid_size = int(np.ceil(np.sqrt(N)))
    padding = 3
    grid_height = L * grid_size + padding * (grid_size - 1)
    grid_width = L * grid_size + padding * (grid_size - 1)
    grid = np.zeros((grid_height, grid_width))
    next_idx = 0
    y0, y1 = 0, L
    for y in range(grid_size):
        x0, x1 = 0, L
        for x in range(grid_size):
            if next_idx < N:
                img = W1[next_idx]
                low, high = np.min(img), np.max(img)
                grid[y0:y1, x0:x1] = 255.0 * (img - low) / (high - low)
            next_idx += 1
            x0 += L + padding
            x1 += L + padding
        y0 += L + padding
        y1 += L + padding
    plt.figure(figsize=(40, 40), dpi=60)
    plt.margins(0,0)
    plt.imshow(grid, cmap = mpl.cm.binary)
    plt.axis("off")
    plt.savefig('netParas.png', bbox_inches='tight')
    print("The parameters is visualized in 'netParas.png'!")