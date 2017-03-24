import json as json

import numpy as np


class CharSCNN:
    def __init__(self):

        '''
            hyper-parameters declaration and initialization
        '''

        self.dwrd = 30
        self.kwrd = 5
        self.dchar = 5
        self.kchar = 3
        self.clu0 = 50
        self.clu1 = 300
        self.hlu = 300
        self.learning_rate = 0.05
        self.no_of_tags = 3

        self.padding_char = '^'
        self.padding_word = '^*'

    def __init__(self, model_path):

        '''
            hyper-parameters declaration and initialization
        '''

        self.dwrd = 30
        self.kwrd = 5
        self.dchar = 5
        self.kchar = 3
        self.clu0 = 50
        self.clu1 = 300
        self.hlu = 300
        self.learning_rate = 0.05
        self.no_of_tags = 3

        self.padding_char = '^'
        self.padding_word = '^*'

        '''
            Model paramenters declaration and initialization
        '''
        self.model_path = model_path
        self.load_model(model_path)

    def load_model(self, model_path):
        '''
            load the model from folder
        '''
        self.word_vocab = 100
        self.Wwrd = np.loadtxt(model_path+'Wwrd.txt').reshape((self.dwrd, self.word_vocab))
        self.char_vocab = 50
        self.Wchar = np.loadtxt(model_path+'Wchar.txt').reshape((self.dchar, self.char_vocab))

        self.w0 = np.loadtxt(model_path+'w0.txt').reshape((self.clu0, self.dchar * self.kchar))
        self.b0 = np.loadtxt(model_path+'b0.txt').reshape((self.clu0, 1))

        self.w1 = np.loadtxt(model_path+'w1.txt').reshape((self.clu1, (self.dwrd + self.clu0) * self.kwrd))
        self.b1 = np.loadtxt(model_path+'b1.txt').reshape((self.clu1, 1))

        self.w2 = np.loadtxt(model_path+'w2.txt').reshape((self.hlu, self.clu1))
        self.b2 = np.loadtxt(model_path+'b2.txt').reshape((self.hlu, 1))

        self.w3 = np.loadtxt(model_path+'w3.txt').reshape((self.no_of_tags, self.hlu))
        self.b3 = np.loadtxt(model_path+'b3.txt').reshape((self.no_of_tags, 1))

        self.word_vocab = json.load(file(model_path+'word_vocab.txt'))
        self.char_vocab = json.load(file(model_path+'char_vocab.txt'))

    def save_model(self, model_path):
        np.savetxt(model_path+'Wwrd.txt', self.Wwrd)
        np.savetxt(model_path+'Wchar.txt', self.Wchar)
        np.savetxt(model_path+'w0.txt', self.w0)
        np.savetxt(model_path+'b0.txt', self.b0)
        np.savetxt(model_path+'w1.txt', self.w1)
        np.savetxt(model_path+'b1.txt', self.b1)
        np.savetxt(model_path+'w2.txt', self.w2)
        np.savetxt(model_path+'b2.txt', self.b2)
        np.savetxt(model_path+'w3.txt', self.w3)
        np.savetxt(model_path+'b3.txt', self.b3)

    def rand_model(self):
        self.word_vocab = 100
        self.Wwrd = np.random.random(size=self.dwrd * self.word_vocab).reshape((self.dwrd, self.word_vocab))
        self.char_vocab = 50
        self.Wchar = np.random.random(size=self.dchar * self.char_vocab).reshape((self.dchar, self.char_vocab))

        self.w0 = np.random.random(size=self.clu0 * self.dchar * self.kchar).reshape(
                (self.clu0, self.dchar * self.kchar))
        self.b0 = np.random.random(size=self.clu0).reshape((self.clu0, 1))

        self.w1 = np.random.random(size=(self.dwrd + self.clu0) * self.kwrd * self.clu1).reshape(
                (self.clu1, (self.dwrd + self.clu0) * self.kwrd))
        self.b1 = np.random.random(size=self.clu1).reshape((self.clu1, 1))

        self.w2 = np.random.random(size=self.hlu * self.clu1).reshape((self.hlu, self.clu1))
        self.b2 = np.random.random(size=self.hlu).reshape((self.hlu, 1))

        self.w3 = np.random.random(size=self.hlu * self.no_of_tags).reshape((self.no_of_tags, self.hlu))
        self.b3 = np.random.random(size=self.no_of_tags).reshape((self.no_of_tags, 1))
        self.save_model(self.model_path)

    '''
        Implementation for section 2.1 of the paper
        Converting each word to a feature vector
    '''

    def word_vector(self, word):
        rwrd = np.array(self.get_word_embedding(word))
        rwch = np.array(self.get_char_embedding(word))
        return np.append(rwrd, rwch)

    '''
        Implementation for section 2.1.1 of the paper
    '''

    def get_word_embedding(self, word):
        w_index = self.get_index_of_word(word)
        rwrd = self.Wwrd[:, w_index]
        return rwrd

    '''
        Implementation for section 2.1.2 of the paper
    '''

    def get_char_embedding(self, word):
        padding_char = self.padding_char
        padding = (self.kchar - 1) / 2
        convolving_word = (padding_char * padding) + str(word) + (padding_char * padding)
        output = np.ndarray((self.clu0, len(word)))
        for i in range(padding, padding + len(word)):
            zi = np.ndarray(shape=(self.dchar, self.kchar))
            for j in range(i - padding, i + padding + 1):
                temp_index = self.get_index_of_char(convolving_word[j])
                zi[:, j - i + padding] = self.Wchar[:, temp_index]
            zi = zi.T.flatten()
            zi = np.reshape(zi, (len(zi), 1))
            ''' Convolution Layer 1 : Convolution at the character level '''
            output[:, (i - padding):(i - padding + 1)] = np.array(np.dot(self.w0, zi) + self.b0)
        convolving_word = 0

        ''' Convolution Layer 1 : Max Pooling '''
        output = output.max(1)
        return output

    def get_index_of_word(self, word):
        return self.word_vocab[word]

    def get_index_of_char(self, char):
        return self.char_vocab[char]

    def get_sent_representation(self, sent_array):

        padding_word = self.padding_word
        padding = (self.kwrd - 1) / 2

        no_of_words = len(sent_array)
        convolving_sent = [padding_word for i in range(0, padding)] + sent_array + [padding_word for i in
                                                                                    range(0, padding)]
        word_vectors = np.ndarray((self.dwrd + self.clu0, no_of_words + 2 * padding))
        for i in range(0, len(convolving_sent)):
            word_vectors[:, i:i + 1] = np.array(self.word_vector(convolving_sent[i])).reshape(
                    (self.dwrd + self.clu0, 1))

        # output = np.ndarray((((self.dwrd + self.clu0)*self.kwrd), no_of_words))
        output = np.ndarray((self.clu1, no_of_words))
        for i in range(padding, padding + no_of_words):
            zi = np.ndarray(shape=((self.dwrd + self.clu0), self.kwrd))
            for j in range(i - padding, i + padding + 1):
                zi[:, j - i + padding] = word_vectors[:, j]
            zi = zi.T.flatten()
            zi = np.reshape(zi, (len(zi), 1))
            ''' Convolution Layer 2 : Convolution at the Word level '''
            output[:, (i - padding):(i - padding + 1)] = np.array(np.dot(self.w1, zi) + self.b1)
        convolving_sent = 0

        ''' Convolution Layer 2 : Max Pooling '''
        output = output.max(1)
        return output

    def NN_Classifier(self, rxsent):

        rxsent = np.reshape(rxsent, (len(rxsent), 1))

        # Layer 1 of NN Classifier
        score_layer1 = np.array(np.dot(self.w2, rxsent) + self.b2)
        score_layer1 = np.tanh(score_layer1)

        # Layer 2 of NN Classifier
        score = np.array(np.dot(self.w3, score_layer1) + self.b3)

        return self.softmax(score)

    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    def get_tag_for(self, sent_array):
        rxsent = self.get_sent_representation(sent_array)
        return self.NN_Classifier(rxsent)

    def train_model(self, x_data, y_data):
        self.sgd_train(x_data, y_data)
        # self.save_model(self.model_path)
        return

    def sgd_train(self, x_data, y_data):
        epsilon = 0.00001
        max_iter = 10000
        training_set_size = len(x_data)
        prev_update_value = pres_update_value = 10000000
        iter_count = 0
        while True:
            iter_count += 1
            update_value = 0
            for i in range(0, training_set_size):
                score_vect = self.get_tag_for(x_data[i])
                error = np.log(1 - score_vect[y_data][i][0])
                update_value = self.learning_rate * error
                self.w0 += update_value
                self.b0 += update_value
                self.w1 += update_value
                self.b1 += update_value
                self.w2 += update_value
                self.b2 += update_value
                self.w3 += update_value
                self.b3 += update_value

                self.Wwrd += update_value
                self.Wchar += update_value
            prev_update_value = pres_update_value
            pres_update_value = update_value

            # if (abs(prev_update_value - pres_update_value)/prev_update_value) < epsilon:
            # if (abs(prev_update_value - pres_update_value)) < epsilon:
            if iter_count >= max_iter:
                # print iter_count
                break

        # print self.get_tag_for(x_data[0])
        return
