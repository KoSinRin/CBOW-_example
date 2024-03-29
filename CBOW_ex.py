# импортирую библиотеки
from keras.models import Sequential
from keras.layers import Embedding, Lambda, Dense
import keras.backend as K
from keras.callbacks import EarlyStopping
import numpy as np

# устанавливаю параметры
vocab_size = 10000
embedding_dim = 100
window_size = 2
batch_size = 64
num_epochs = 20

# генерация training data
sentences = [['I', 'love', 'working', 'with', 'data'], ['Machine', 'learning', 'is', 'fun']]
word_dict = {}
for sentence in sentences:
    for word in sentence:
        if word not in word_dict:
            word_dict[word] = len(word_dict)
            
# создание CBOW training data
cbow_pairs = []
for sentence in sentences:
    for i, word in enumerate(sentence):
        context_words = sentence[max(0, i - window_size):i] + sentence[i + 1:min(i + window_size + 1, len(sentence))]
        context_ids = [word_dict[w] for w in context_words]
        cbow_pairs.append((context_ids, word_dict[word]))

# создаю входной и выходной массивы
x_train = np.zeros((len(cbow_pairs), 2 * window_size), dtype=np.int32)
y_train = np.zeros((len(cbow_pairs), vocab_size), dtype=np.int32)
for i, (context, target) in enumerate(cbow_pairs):
    x_train[i, :] = context
    y_train[i, target] = 1

# создание CBOW модели
cbow_model = Sequential()
cbow_model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=2 * window_size))
cbow_model.add(Lambda(lambda x: K.mean(x, axis=1), output_shape=(embedding_dim,)))
cbow_model.add(Dense(vocab_size, activation='softmax'))

# компилирую модель
cbow_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')

# ранняя остановка
early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min')

# обучаю 
cbow_model.fit(x_train, y_train, batch_size=batch_size, epochs=num_epochs, callbacks=[early_stopping], validation_split=0.1)

# оцениваю
cbow_model.evaluate(x_train, y_train, batch_size=batch_size)

# данный код является лишь примером, его можно улучшить множеством способов.
# upd1: sparse_categorical_crossentropy вместо categorical_crossentropy
# upd2: callback-функция EarlyStopping
