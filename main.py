import numpy as np
import json as json

from CharSCNN import CharSCNN


def main():
    sent_1 = ['I', 'am', 'a', 'good', 'boy']
    sent_2 = ['good', 'boy']

    x_data = [sent_1]
    y_data = [0, 0]
    scnn = CharSCNN('./model/')

    print 'Before Trainging'
    print 'Sent 1 - I am a good boy'
    print scnn.get_tag_for(sent_1)
    print 'Sent 2 - good boy'
    print scnn.get_tag_for(sent_2)
    scnn.train_model(x_data, y_data)
    scnn.save_model('./model/new/')
    scnn = CharSCNN('./model/new/')
    print 'After Trainging'
    print 'Sent 1 - I am a good boy'
    print scnn.get_tag_for(sent_1)
    print 'Sent 2 - good boy'
    print scnn.get_tag_for(sent_2)


def init():
    dwrd = 30
    kwrd = 5
    dchar = 5
    kchar = 3
    clu0 = 50
    clu1 = 300
    hlu = 300
    learning_rate = 0.01
    no_of_tags = 3

    padding_char = '^'
    padding_word = '^*'

    '''
        Model paramenters declaration and initialization
    '''
    word_vocab = 100
    Wwrd = np.random.random(size=dwrd * word_vocab).reshape((dwrd, word_vocab))
    np.savetxt('./model/Wwrd.txt', Wwrd)
    char_vocab = 50
    Wchar = np.random.random(size=dchar * char_vocab).reshape((dchar, char_vocab))
    np.savetxt('./model/Wchar.txt', Wchar)
    w0 = np.random.random(size=clu0 * dchar * kchar).reshape(
            (clu0, dchar * kchar))
    b0 = np.random.random(size=clu0).reshape((clu0, 1))
    np.savetxt('./model/w0.txt', w0)
    np.savetxt('./model/b0.txt', b0)

    w1 = np.random.random(size=(dwrd + clu0) * kwrd * clu1).reshape(
            (clu1, (dwrd + clu0) * kwrd))
    b1 = np.random.random(size=clu1).reshape((clu1, 1))
    np.savetxt('./model/w1.txt', w1)
    np.savetxt('./model/b1.txt', b1)

    w2 = np.random.random(size=hlu * clu1).reshape((hlu, clu1))
    b2 = np.random.random(size=hlu).reshape((hlu, 1))
    np.savetxt('./model/w2.txt', w2)
    np.savetxt('./model/b2.txt', b2)

    w3 = np.random.random(size=hlu * no_of_tags).reshape((no_of_tags, hlu))
    b3 = np.random.random(size=no_of_tags).reshape((no_of_tags, 1))
    np.savetxt('./model/w3.txt', w3)
    np.savetxt('./model/b3.txt', b3)


if __name__ == '__main__':
    '''
        Vocab Creation
    '''
    # word_vocab = {'^*': 0, 'I': 1, 'am': 2, 'a': 3, 'good': 4, 'boy': 5}
    # json.dump(word_vocab, file('./model/word_vocab.txt', 'w'))
    #
    # char_vocab = {'^': 0,'*': 1, 'I': 2, 'a': 3, 'm': 4, 'g': 5, 'o': 6, 'd': 7, 'b': 8, 'y': 9}
    # json.dump(char_vocab, file('./model/char_vocab.txt', 'w'))

    '''
        Random Initialization
    '''
    # init()
    # sent_array = ['I', 'am', 'a', 'good', 'boy']
    #
    # scnn = CharSCNN('./model/')
    # print scnn.get_tag_for(sent_array)

    '''
        Training and performance test
    '''
    main()
