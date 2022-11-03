import tensorflow as tf
import matplotlib.pyplot as plt
import os
import cv2
import random
import numpy as np
from sklearn.model_selection import train_test_split

DATADIR = "/mnt/c/Users/imoga/ac/media2/data"
CATEGORIES = ["1_1", "1_2", "1_3"]
IMG_SIZE = 300
training_data = []

def create_training_data():
    for class_num, category in enumerate(CATEGORIES):
        path = os.path.join(DATADIR, category)
        for image_name in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, image_name),)  # 画像読み込み
                img_resize_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # 画像のリサイズ
                training_data.append([img_resize_array, class_num])  # 画像データ、ラベル情報を追加
            except Exception as e:
                pass

def select_label(Y_train):
    if Y_train == 0:
        return '1_1'
    elif Y_train == 1:
        return '1_2'
    else:
        return '1_3'

create_training_data()

random.shuffle(training_data)  # データをシャッフル

X_train = []  # 画像データ
Y_train = []  # ラベル情報

# データセット作成
for feature, label in training_data:
    X_train.append(feature)
    Y_train.append(label)

# numpy配列に変換
X_train = np.array(X_train)
Y_train = np.array(Y_train)

# データセットの確認
#for i in range(0, 4):
    #print("学習データのラベル：", Y_train[i])
    #plt.subplot(2, 2, i+1)
    #plt.axis('off')
    #plt.title(label = select_label(Y_train[i]))
    #img_array = cv2.cvtColor(X_train[i], cv2.COLOR_BGR2RGB)
    #plt.imshow(img_array)
#plt.savefig("test.jpg")

# trainとtestに分類
x_train, x_test, y_train, y_test = train_test_split(X_train, Y_train, test_size=0.1, random_state=0, stratify=Y_train)

# 正規化
x_train, x_test = x_train / 255.0, x_test /255.0

# ニューラルネットワークを構築
model = tf.keras.models.Sequential([
tf.keras.layers.Flatten(input_shape=(300, 300, 3)),
tf.keras.layers.Dense(128, activation='relu'),
tf.keras.layers.Dense(3, activation='softmax'),
])

##推定
#predictions = model(x_train[:1]).numpy()
#print(tf.nn.softmax(predictions).numpy())

# modelcompile
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer='adam',loss=loss_fn,metrics=['accuracy'])

# training
model.fit(x_train, y_train,epochs=25)

# 評価
model.evaluate(x_test, y_test, verbose=2)

#testを用いた評価
probability_model = tf.keras.Sequential([model,tf.keras.layers.Softmax()])
print(probability_model(x_test[:8]))
#test画像を表示4個
for i in range(0, 8):
    print("学習データのラベル：", y_test[i])
    plt.subplot(4, 4, i+1)
    plt.axis('off')
    plt.title(label = select_label(y_test[i]))
    plt.imshow(x_test[i])
plt.savefig("test.jpg")