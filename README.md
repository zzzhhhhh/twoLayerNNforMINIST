# twoLayerNNforMINIST

* model_analysis.ipynb：包含模型训练，loss 和 accuracy 曲线绘制，网络参数绘制；模型调参；模型测试集预测及评估三个部分的过程及结果。
* train.py：模型训练文件
* test.py：模型预测文件
* model.py：两层的神经网络分类模型
* figures/：loss 和 accuracy 曲线图、网络参数图、调参对比图等。
1. 下载数据集：链接: https://pan.baidu.com/s/1TMkaAqPZN_XOEa6RH_U17w  密码: d69b，与 train.py 放在同目录下。
2. `python train.py` 训练模型，得到的参数存储在 model.npz 。
3. `python test.py` 加载模型参数，对测试集进行预测，预测结果存储在 test_pred.txt 。
