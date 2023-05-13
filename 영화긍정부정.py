import json
import numpy as np
import tensorflow as tf

from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence

import logging, os

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"



np_load_old = np.load
np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)

# 데이터를 불러오고 전처리하는 함수입니다.

n_of_training_ex = 5000
n_of_testing_ex = 1000

PATH = "D:/pythonProject/EliceExample/"


def imdb_data_load():
    X_train = np.load(PATH + "X_train.npy")[:n_of_training_ex]
    y_train = np.load(PATH + "y_train.npy")[:n_of_training_ex]
    X_test = np.load(PATH + "X_test.npy")[:n_of_testing_ex]
    y_test = np.load(PATH + "y_test.npy")[:n_of_testing_ex]

    # 단어 사전 불러오기
    with open(PATH + "imdb_word_index.json") as f:
        word_index = json.load(f)
    # 인덱스 -> 단어 방식으로 딕셔너리 설정
    inverted_word_index = dict((i, word) for (word, i) in word_index.items())
    # 인덱스를 바탕으로 문장으로 변환
    decoded_sequence = " ".join(inverted_word_index[i] for i in X_train[0])

    print("첫 번째 X_train 데이터 샘플 문장: \n", decoded_sequence)
    print("\n첫 번째 X_train 데이터 샘플 토큰 인덱스 sequence: \n", X_train[0])
    print("첫 번째 X_train 데이터 샘플 토큰 시퀀스 길이: ", len(X_train[0]))
    print("첫 번째 y_train 데이터: ", y_train[0])

    return X_train, y_train, X_test, y_test



# 학습용 및 평가용 데이터를 불러오고 샘플 문장을 출력합니다.
X_train, y_train, X_test, y_test = imdb_data_load()

"""
1. 인덱스로 변환된 X_train, X_test 시퀀스에 패딩을 수행하고 각각 X_train, X_test에 저장합니다.
   시퀀스 최대 길이는 300으로 설정합니다.
"""
X_train = sequence.pad_sequences(X_train, maxlen=300, padding='post')
X_test = sequence.pad_sequences(X_test, maxlen=300, padding='post')

print("\n패딩을 추가한 첫 번째 X_train 데이터 샘플 토큰 인덱스 sequence: \n",X_train[0])

embedding_vector_length = 32
max_review_length = 300


"""
1. 모델을 구현합니다.
   임베딩 레이어 다음으로 `SimpleRNN`을 사용하여 RNN 레이어를 쌓고 노드의 개수는 5개로 설정합니다. 
   Dense 레이어는 0, 1 분류이기에 노드를 1개로 하고 activation을 'sigmoid'로 설정되어 있습니다.
"""
model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(1000, embedding_vector_length, input_length = max_review_length),
    tf.keras.layers.SimpleRNN(5),
    tf.keras.layers.Dense(1, activation='sigmoid')
    ])

# 모델을 확인합니다.
print(model.summary())

# 학습 방법을 설정합니다.
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

# 학습을 수행합니다.
model_history = model.fit(X_train, y_train, epochs = 3, verbose = 2)

"""
1. 평가용 데이터를 활용하여 모델을 평가합니다.
   loss와 accuracy를 계산하고 loss, test_acc에 저장합니다.
"""
loss, test_acc = model.evaluate(X_test, y_test, verbose = 0)

"""
2. 평가용 데이터에 대한 예측 결과를 predictions에 저장합니다.
"""
predictions = model.predict(X_test)

# 모델 평가 및 예측 결과를 출력합니다.
print('\nTest Loss : {:.4f} | Test Accuracy : {}'.format(loss, test_acc))
print('예측한 Test Data 클래스 : ',1 if predictions[0]>=0.5 else 0)
