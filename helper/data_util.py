import random
import os
class textData(object):
    """docstring for textData.
    data: T *
    where
        T - text file(s)
    """
    def __init__(self, allchar):
        super(textData, self).__init__()
        self.data = []
        self.categories = 0
        self.allchar = allchar
        self.nchar = len(allchar)
    def load_data(self, path='resource'):
        data_list = os.listdir(path)
        for item in data_list:
            try:
                self.data.append(''.join(_ for _ in open(os.path.join(path,item))))
            except:
                print("cant load data {}".format(item))
        self.categories = len(self.data)
    def _randomText():
        return random.randint(0,self.categories-1)
    def _textToNum(text):
        return [self.nchar.find(x) for x in self.allchar]
    def randomInandOut(l):
        text = self._randomText()
        line = self.data[text]
        lenght = len(self.data[text])
        st = random.randint(0, lenght - l - 1)
        input_line = self._textToNum(line[st:st+l])
        output_line = input_line[1:]
        output_line.append(self._textToNum(line[st+l]))
        return input_line, output_line
