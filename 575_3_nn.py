# 用神经网络深度学习作手写数字识别完整代码
# 包含数据预处理、模型构建、训练及可视化

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import matplotlib.pyplot as plt

# 忽略TensorFlow CPU指令集警告
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 数据预处理函数
def load_subset():
    # 加载完整MNIST数据集
    (x_train_full, y_train_full), (x_test_full, y_test_full) = tf.keras.datasets.mnist.load_data()
    
    # 子集筛选函数
    def filter_samples(x, y, samples_per_class):
        indices = []
        for label in range(10):  # 遍历0-9所有数字
            # 获取当前标签的所有索引并取前samples_per_class个
            idx = np.where(y == label)[0][:samples_per_class]
            indices.extend(idx)
        # 打乱顺序避免类别顺序影响
        np.random.shuffle(indices)
        return x[indices], y[indices]
    
    # 创建训练子集 (6000样本)
    x_train, y_train = filter_samples(x_train_full, y_train_full, 600)
    # 创建测试子集 (1000样本)
    x_test, y_test = filter_samples(x_test_full, y_test_full, 100)
    
    # 数据预处理
    # 归一化并添加通道维度
    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    # 标签转为one-hot编码
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)
    
    return x_train, y_train, x_test, y_test

# 模型构建函数
def build_model(kernel_size=(3,3), filters1=6, filters2=16):
    model = Sequential([
        Conv2D(filters1, kernel_size, activation='relu', input_shape=(28,28,1)),
        MaxPooling2D((2,2)),
        Conv2D(filters2, kernel_size, activation='relu'),
        MaxPooling2D((2,2)),
        Flatten(),
        Dense(120, activation='relu'),
        Dense(84, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# 训练与可视化函数
def train_and_visualize(models_config):
    # 加载数据
    x_train, y_train, x_test, y_test = load_subset()
    
    histories = []
    for config in models_config:
        print(f"\nTraining Model: {config['name']}")
        model = build_model(**config['params'])
        history = model.fit(x_train, y_train,
                          epochs=12,
                          validation_data=(x_test, y_test),
                          verbose=1)
        histories.append({
            'name': config['name'],
            'history': history,
            'test_loss': model.evaluate(x_test, y_test, verbose=0)[0],
            'test_acc': model.evaluate(x_test, y_test, verbose=0)[1]
        })
    
    # 可视化结果
    plt.figure(figsize=(12,5))
    
    # 准确率曲线
    plt.subplot(1,2,1)
    for h in histories:
        plt.plot(h['history'].history['val_accuracy'], label=h['name'])
    plt.title('Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # 损失曲线
    plt.subplot(1,2,2)
    for h in histories:
        plt.plot(h['history'].history['val_loss'], label=h['name'])
    plt.title('Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # 打印测试结果
    print("\nFinal Test Results:")
    for h in histories:
        print(f"{h['name']} - Loss: {h['test_loss']:.4f}, Acc: {h['test_acc']:.4f}")

# 项目配置
models_config = [
    {
        'name': 'Baseline (3x3)',
        'params': {'kernel_size': (3,3), 'filters1':6, 'filters2':16}
    },
    {
        'name': '5x5 Kernel',
        'params': {'kernel_size': (5,5), 'filters1':6, 'filters2':16}
    },
    {
        'name': '5x5 + More Filters',
        'params': {'kernel_size': (5,5), 'filters1':12, 'filters2':32}
    }
]

# 执行完整流程
if __name__ == "__main__":
    train_and_visualize(models_config)