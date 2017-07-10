import chainer.functions as F
import chainer.links as L
import numpy as np
import chainer


class Discriminator(chainer.Chain):
    def __init__(self, bottom_width=2, ch=512, wscale=0.02):
        w = chainer.initializers.Normal(wscale)
        super(Discriminator, self).__init__()
        with self.init_scope():
            self.c0 = L.Convolution2D(3, ch // 16, 3, 1, 1, initialW=w)
            self.c1 = L.Convolution2D(ch // 16, ch // 8, 4, 2, 1, initialW=w)
            self.c2 = L.Convolution2D(ch // 8, ch // 4, 4, 2, 1, initialW=w)
            self.c3 = L.Convolution2D(ch // 4, ch // 2, 4, 2, 1, initialW=w)
            self.c4 = L.Convolution2D(ch // 2, ch // 1, 4, 2, 1, initialW=w)
            self.l5 = L.Linear(bottom_width * bottom_width * ch, 1, initialW=w)
            self.bn1 = L.BatchNormalization(ch // 8)
            self.bn2 = L.BatchNormalization(ch // 4)
            self.bn3 = L.BatchNormalization(ch // 2)
            self.bn4 = L.BatchNormalization(ch // 1, use_gamma=False)

    def __call__(self, x):
        h = x
        h = F.leaky_relu(self.c0(h))
        h = F.leaky_relu(self.bn1(self.c1(h)))
        h = F.leaky_relu(self.bn2(self.c2(h)))
        h = F.leaky_relu(self.bn3(self.c3(h)))
        feature = self.bn4(self.c4(h))
        h = F.leaky_relu(feature)
        return feature, self.l5(h)


class Denoiser(chainer.Chain):
    def __init__(self):
        super(Denoiser, self).__init__()
        with self.init_scope():
            self.l0 = L.Linear(2048, 2048)
            self.l1 = L.Linear(2048, 2048)
            self.l2 = L.Linear(2048, 2048)
            self.l3 = L.Linear(2048, 2048)
            self.l4 = L.Linear(2048, 2048)
            self.l5 = L.Linear(2048, 2048)
            self.l6 = L.Linear(2048, 2048)
            self.l7 = L.Linear(2048, 2048)
            self.l8 = L.Linear(2048, 2048)
            self.l9 = L.Linear(2048, 2048)
            self.bn0 = L.BatchNormalization(2048)
            self.bn1 = L.BatchNormalization(2048)
            self.bn2 = L.BatchNormalization(2048)
            self.bn3 = L.BatchNormalization(2048)
            self.bn4 = L.BatchNormalization(2048)
            self.bn5 = L.BatchNormalization(2048)
            self.bn6 = L.BatchNormalization(2048)
            self.bn7 = L.BatchNormalization(2048)
            self.bn8 = L.BatchNormalization(2048)

    def __call__(self, x):
        h = F.reshape(x, (len(x), 2048))
        h = F.leaky_relu(self.bn0(self.l0(h)))
        h = F.leaky_relu(self.bn1(self.l1(h)))
        h = F.leaky_relu(self.bn2(self.l2(h)))
        h = F.leaky_relu(self.bn3(self.l3(h)))
        h = F.leaky_relu(self.bn4(self.l4(h)))
        h = F.leaky_relu(self.bn5(self.l5(h)))
        h = F.leaky_relu(self.bn6(self.l6(h)))
        h = F.leaky_relu(self.bn7(self.l7(h)))
        h = F.leaky_relu(self.bn8(self.l8(h)))
        return F.reshape(self.l9(h), (len(x), 512, 2, 2))
