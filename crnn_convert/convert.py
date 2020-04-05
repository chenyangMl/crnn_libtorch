"""

# Step1: Converting to Torch Script via Tracing
import torch
import torchvision

# An instance of your model.
model = torchvision.models.resnet18()

# An example input you would normally provide to your model's forward() method.
example = torch.rand(1, 3, 224, 224)

# Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
traced_script_module = torch.jit.trace(model, example)


# test code
# output = traced_script_module(torch.ones(1, 3, 224, 224))
# print(output[0, :5])


# ================================================

# Step2: Serializing your script module to a file
traced_script_module.save("./traced_resnet_model.pt")

# ================================================
# Step3: Loding your script module in c++

"""
import torch
import torch.nn as nn
from collections import OrderedDict
import os

from model import keys


class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output


class CRNN(nn.Module):

    def __init__(self, imgH, nc, nclass, nh, n_rnn=2):
        super(CRNN, self).__init__()
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'

        ks = [3, 3, 3, 3, 3, 3, 2]
        ps = [1, 1, 1, 1, 1, 1, 0]
        ss = [1, 1, 1, 1, 1, 1, 1]
        nm = [64, 128, 256, 256, 512, 512, 512]

        cnn = nn.Sequential()

        # def convRelu(i, batchNormalization=False):
        #     nIn = nc if i == 0 else nm[i - 1]
        #     nOut = nm[i]
        #     cnn.add_module('conv{0}'.format(i),
        #                    nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
        #
        #     if batchNormalization:
        #         cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
        #     if leakyRelu:
        #         cnn.add_module('relu{0}'.format(i),
        #                        nn.LeakyReLU(0.2, inplace=True))
        #     else:
        #         cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

        "*********************************************************"
        # convRelu(0)
        nIn = nc
        nOut = nm[0]
        cnn.add_module('conv{0}'.format(0),
                       nn.Conv2d(nIn, nOut, ks[0], ss[0], ps[0]))
        cnn.add_module('relu{0}'.format(0), nn.ReLU(True))
        "*********************************************************"

        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # 64x16x64

        "*********************************************************"
        # convRelu(1)
        nIn = nm[0]
        nOut = nm[1]
        cnn.add_module('conv{0}'.format(1),
                       nn.Conv2d(nIn, nOut, ks[1], ss[1], ps[1]))
        cnn.add_module('relu{0}'.format(1), nn.ReLU(True))
        "*********************************************************"

        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 128x8x32

        "*********************************************************"
        # convRelu(2, True)
        nIn = nm[1]
        nOut = nm[2]
        cnn.add_module('conv{0}'.format(2),
                       nn.Conv2d(nIn, nOut, ks[2], ss[2], ps[2]))
        cnn.add_module('batchnorm{0}'.format(2), nn.BatchNorm2d(nOut))
        cnn.add_module('relu{0}'.format(2), nn.ReLU(True))
        "*********************************************************"

        "*********************************************************"
        # convRelu(3)
        nIn = nm[2]
        nOut = nm[3]
        cnn.add_module('conv{0}'.format(3),
                       nn.Conv2d(nIn, nOut, ks[3], ss[3], ps[3]))
        cnn.add_module('relu{0}'.format(3), nn.ReLU(True))
        "*********************************************************"

        cnn.add_module('pooling{0}'.format(2),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 256x4x16

        "*********************************************************"
        # convRelu(4, True)
        nIn = nm[3]
        nOut = nm[4]
        cnn.add_module('conv{0}'.format(4),
                       nn.Conv2d(nIn, nOut, ks[4], ss[4], ps[4]))
        cnn.add_module('batchnorm{0}'.format(4), nn.BatchNorm2d(nOut))
        cnn.add_module('relu{0}'.format(4), nn.ReLU(True))
        "*********************************************************"

        "*********************************************************"
        # convRelu(5)
        nIn = nm[4]
        nOut = nm[5]
        cnn.add_module('conv{0}'.format(5),
                       nn.Conv2d(nIn, nOut, ks[5], ss[5], ps[5]))
        cnn.add_module('relu{0}'.format(5), nn.ReLU(True))
        "*********************************************************"

        cnn.add_module('pooling{0}'.format(3),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 512x2x16

        "*********************************************************"
        # convRelu(6, True)  # 512x1x16
        nIn = nm[5]
        nOut = nm[6]
        cnn.add_module('conv{0}'.format(6),
                       nn.Conv2d(nIn, nOut, ks[6], ss[6], ps[6]))
        cnn.add_module('batchnorm{0}'.format(6), nn.BatchNorm2d(nOut))
        cnn.add_module('relu{0}'.format(6), nn.ReLU(True))
        "*********************************************************"

        self.cnn = cnn
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, nh, nh),
            BidirectionalLSTM(nh, nh, nclass))

    def forward(self, input):
        # conv features
        conv = self.cnn(input)
        b, c, h, w = conv.size()
        # assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c]

        # rnn features
        output = self.rnn(conv)

        return output


if __name__ == "__main__":
    Base_dir = os.path.abspath(os.path.join(
        os.path.dirname(__file__), os.pardir))
    model_path = os.path.join(Base_dir, "crnn_convert/model/crnn.pth")

    # load model
    if torch.cuda.is_available():
        # LSTMFLAG=True crnn 否则 dense ocr
        model = CRNN(32, 1, len(keys.alphabetEnglish) + 1, 256, 1).cuda()
    else:
        model = CRNN(32, 1, len(keys.alphabetEnglish) + 1, 256, 1).cpu()

    state_dict = torch.load(
        model_path, map_location=lambda storage, loc: storage)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace('module.', '')  # remove `module.`
        new_state_dict[name] = v
    # # # load params
    model.load_state_dict(new_state_dict)

    # convert pth-model to pt-model
    example = torch.rand(1, 1, 32, 512)
    traced_script_module = torch.jit.trace(model, example)
    traced_script_module.save("src/crnn.pt")
