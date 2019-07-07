import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from keras.callbacks import EarlyStopping

######################################################################
# optimal svm model is;
#   SVC(C=10, cache_size=200, class_weight=None, coef0=0.0,
#   decision_function_shape='ovr', degree=3, gamma=0.1, kernel='rbf',
#   max_iter=-1, probability=False, random_state=None, shrinking=True,
#   tol=0.001, verbose=False)
######################################################################
# this is found through using the GridSearchCrossValidation Technique
######################################################################
# from sklearn.model_selection import GridSearchCV
# parameters = {
#    "kernel": ["rbf"],
#    "C": [1,10, 100,1000],
#    "gamma": [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
#    }
#grid = GridSearchCV(SVC(), parameters, cv=5, verbose=2)
# clf.best_estimator_
######################################################################

def get_svm_predictions(X_train, X_test, y_train):

    clf = SVC(C=10, cache_size=200, class_weight=None, coef0=0.0,
              decision_function_shape='ovr', degree=3, gamma=0.1, kernel='rbf',
              max_iter=-1, probability=False, random_state=None, shrinking=True,
              tol=0.001, verbose=False)
    clf.fit(X_train, y_train)
    svm_predictions = clf.predict(X_test)
    np.save('predictions/svm_predictions', svm_predictions)
    print('svm predictions saved...')

def get_log_predictions(X_train, X_test, y_train):

    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    logistic_predictions = clf.predict(X_test)
    np.save('predictions/log_predictions', logistic_predictions)
    print('log_reg predictions saved...')

def get_rf_predictions(X_train, X_test, y_train):

    clf = RandomForestClassifier(n_estimators=1000)
    clf.fit(X_train, y_train)
    rf_predictions = clf.predict(X_test)
    np.save('predictions/rf_predictions', rf_predictions)
    print('rf predictions saved...')

def get_lstm_predictions(X_train, X_test, y_train, y_test, use_glove=False):

    split = len(X_train)
    # Rough Avg number of words in each article. How long every sequence will be.
    MAX_SEQUENCE_LENGTH = 1000
    # This is fixed.
    EMBEDDING_DIM = 100

    X = X_train + X_test
    y = y_train + y_test
    tokenizer = Tokenizer(lower=True)
    tokenizer.fit_on_texts(X)
    word_index = tokenizer.word_index
    vocab_size = len(word_index) + 1
    print('Found %s unique tokens.' % len(word_index))
    X = tokenizer.texts_to_sequences(X)
    X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
    print('Shape of data tensor:', X.shape)

    # one hot encoding the labels
    y_one_hot = []
    for i in y:
        temp = np.zeros(23)
        temp[i - 1] = 1
        y_one_hot.append(temp)

    y_one_hot = np.array(y_one_hot)

    # splitting the train test
    X_train = X[:split]
    X_test = X[split:]
    y_train = y_one_hot[:split]
    y_test = y_one_hot[split:]

    # GloVe

    if use_glove:
        embeddings_index = dict()
        f = open('glove.6B.100d.txt')
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()
        print('Loaded %s word vectors.' % len(embeddings_index))
        # create a weight matrix for words in training docs
        embedding_matrix = np.zeros((vocab_size, EMBEDDING_DIM))
        for word, i in word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

    # construction of the lstm
    model = Sequential()
    if use_glove:
        model.add(Embedding(vocab_size, EMBEDDING_DIM, weights=[embedding_matrix], input_length=X.shape[1]))
    else:
        model.add(Embedding(vocab_size, EMBEDDING_DIM, input_length=X.shape[1]))
    model.add(SpatialDropout1D(0.2))
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(23, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())

    # running the model
    epochs = 20
    batch_size = 64
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test),
                        callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])

    # plotting
    plt.title('Loss')
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show();

    plt.title('Accuracy')
    plt.plot(history.history['acc'], label='train')
    plt.plot(history.history['val_acc'], label='test')
    plt.legend()
    plt.show();

    # getting predictions
    predictions = model.predict(X_test)
    predictions = np.array([np.argmax(i) + 1 for i in predictions])

    if use_glove:
        np.save('predictions/lstm_glove_predictions', predictions)
    else:
        np.save('predictions/lstm_predictions', predictions)