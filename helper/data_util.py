import random
import os
import numpy as np
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
    def _randomText(self):
        return random.randint(0,self.categories-1)
    def _textNormalize(self,text):
        replace = {'“':'"','”':'"','’':'\'','‘':'\''}
        out = text[:]
        for key in replace:
            out = out.replace(key,replace[key])
        return out
    def load_data(self, path='resource'):
        self.data = []
        data_list = os.listdir(path)
        for item in data_list:
            try:
                self.data.append(''.join(self._textNormalize(_) for _ in open(os.path.join(path,item))))
            except IOError:
                print("cant load data {}".format(item))
        self.categories = len(self.data)
    def textToNum(self,text):
        return [self.allchar.find(x) for x in text]
    def numToText(self,num_list):
        return ''.join(self.allchar[x] for x in num_list)
    def randomInandOut(self, l_in,l_out, offset=None, batch_num=20):
        if offset is None:
            offset = l_in - l_out + 1
        text = self._randomText()
        line = self.data[text]
        lenght = len(self.data[text])
        st = random.randint(0, lenght - l_in - 1 - offset)
        input_line = self.textToNum(line[st:st+l_in])
        output_line = self.textToNum(line[st+offset:st+offset+l_out])
        return input_line, output_line
    def makeInandOutTensor(self,l_in, l_out, batch_size):
        in_tensor = np.zeros((l_in, batch_size, self.nchar))
        out_tensor = np.zeros((l_out, batch_size, self.nchar))
        for i in range(batch_size):
            inp, outp = self.randomInandOut(l_in,l_out,batch_num=batch_size)
            in_tensor[np.arange(0,l_in),i,inp] = 1
            out_tensor[np.arange(0,l_out),i,outp] = 1
        return in_tensor, out_tensor

    def tensorToText(self,tensor):
        '''
        Input: (T,1,nchar)
        '''
        T = tensor.shape[0]
        inp = tensor.reshape((T,self.nchar))
        onehot = np.argmax(inp,axis=1)
        print(onehot)
        return self.numToText(onehot)
    def penalty(self):
        collect = np.zeros(self.nchar)
        for text in self.data:
            for char in text:
                collect[self.allchar.find(char)] += 1
        return collect/np.sum(collect) * self.nchar
    def textTotensor(self, text):
        numlist = self.textToNum(text)
        n = len(text)
        out = np.zeros((n,1,self.nchar))
        out[np.arange(0,n),0,numlist] = 1
        return out
