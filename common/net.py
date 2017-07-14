import sys
import os

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda
import numpy as np
from sn.sn_linear import SNLinear
from sn.sn_convolution_2d import SNConvolution2D


def add_noise(h, sigma=0.2):
    xp = cuda.get_array_module(h.data)
    if not chainer.config.train:
        return h
    else:
        return h + sigma * xp.random.randn(*h.data.shape)


# differentiable backward functions

def backward_linear(x_in, x, l):
    y = F.matmul(x, l.W)
    return y


def backward_convolution(x_in, x, l):
    y = F.deconvolution_2d(x, l.W, None, l.stride, l.pad, (x_in.data.shape[2], x_in.data.shape[3]))
    return y


def backward_deconvolution(x_in, x, l):
    y = F.convolution_2d(x, l.W, None, l.stride, l.pad)
    return y


def backward_relu(x_in, x):
    y = (x_in.data > 0) * x
    return y


def backward_leaky_relu(x_in, x, a):
    y = (x_in.data > 0) * x + a * (x_in.data < 0) * x
    return y


def backward_sigmoid(x_in, g):
    y = F.sigmoid(x_in)
    return g * y * (1 - y)


# common generators

class DCGANGenerator(chainer.Chain):
    def __init__(self, n_hidden=128, bottom_width=4, ch=512, wscale=0.02,
                 z_distribution="uniform", hidden_activation=F.relu, output_activation=F.tanh, use_bn=True):
        super(DCGANGenerator, self).__init__()
        self.n_hidden = n_hidden
        self.ch = ch
        self.bottom_width = bottom_width
        self.z_distribution = z_distribution
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.use_bn = use_bn

        with self.init_scope():
            w = chainer.initializers.Normal(wscale)
            self.l0 = L.Linear(self.n_hidden, bottom_width * bottom_width * ch,
                               initialW=w)
            self.dc1 = L.Deconvolution2D(ch, ch // 2, 4, 2, 1, initialW=w)
            self.dc2 = L.Deconvolution2D(ch // 2, ch // 4, 4, 2, 1, initialW=w)
            self.dc3 = L.Deconvolution2D(ch // 4, ch // 8, 4, 2, 1, initialW=w)
            self.dc4 = L.Deconvolution2D(ch // 8, 3, 3, 1, 1, initialW=w)
            if self.use_bn:
                self.bn0 = L.BatchNormalization(bottom_width * bottom_width * ch)
                self.bn1 = L.BatchNormalization(ch // 2)
                self.bn2 = L.BatchNormalization(ch // 4)
                self.bn3 = L.BatchNormalization(ch // 8)

    def make_hidden(self, batchsize):
        if self.z_distribution == "normal":
            return np.random.randn(batchsize, self.n_hidden, 1, 1) \
                .astype(np.float32)
        elif self.z_distribution == "uniform":
            return np.random.uniform(-1, 1, (batchsize, self.n_hidden, 1, 1)) \
                .astype(np.float32)
        else:
            raise Exception("unknown z distribution: %s" % self.z_distribution)

    def __call__(self, z):
        if not self.use_bn:
            h = F.reshape(self.hidden_activation(self.l0(z)),
                          (len(z), self.ch, self.bottom_width, self.bottom_width))
            h = self.hidden_activation(self.dc1(h))
            h = self.hidden_activation(self.dc2(h))
            h = self.hidden_activation(self.dc3(h))
            x = self.output_activation(self.dc4(h))
        else:
            h = F.reshape(self.hidden_activation(self.bn0(self.l0(z))),
                          (len(z), self.ch, self.bottom_width, self.bottom_width))
            h = self.hidden_activation(self.bn1(self.dc1(h)))
            h = self.hidden_activation(self.bn2(self.dc2(h)))
            h = self.hidden_activation(self.bn3(self.dc3(h)))
            x = self.output_activation(self.dc4(h))
        return x


class UpResBlock(chainer.Chain):
    """
        pre activation residual block
    """

    def __init__(self, ch, wscale=0.02):
        super(UpResBlock, self).__init__()
        with self.init_scope():
            w = chainer.initializers.Normal(wscale)
            self.c0 = L.Convolution2D(ch, ch, 3, 1, 1, initialW=w)
            self.c1 = L.Convolution2D(ch, ch, 3, 1, 1, initialW=w)
            self.cs = L.Convolution2D(ch, ch, 3, 1, 1, initialW=w)
            self.bn0 = L.BatchNormalization(ch)
            self.bn1 = L.BatchNormalization(ch)

    def __call__(self, x):
        h = self.c0(F.unpooling_2d(F.relu(self.bn0(x)), 2, 2, 0, cover_all=False))
        h = self.c1(F.relu(self.bn1(h)))
        hs = self.cs(F.unpooling_2d(x, 2, 2, 0, cover_all=False))
        return h + hs


class ResnetGenerator(chainer.Chain):
    def __init__(self, n_hidden=128, bottom_width=4, z_distribution="normal", wscale=0.02):
        self.n_hidden = n_hidden
        self.bottom_width = bottom_width
        self.z_distribution = z_distribution
        super(ResnetGenerator, self).__init__()
        with self.init_scope():
            w = chainer.initializers.Normal(wscale)
            self.l0 = L.Linear(n_hidden, n_hidden * bottom_width * bottom_width)
            self.r0 = UpResBlock(n_hidden)
            self.r1 = UpResBlock(n_hidden)
            self.r2 = UpResBlock(n_hidden)
            self.bn2 = L.BatchNormalization(n_hidden)
            self.c3 = L.Convolution2D(n_hidden, 3, 3, 1, 1, initialW=w)

    def make_hidden(self, batchsize):
        if self.z_distribution == "normal":
            return np.random.randn(batchsize, self.n_hidden, 1, 1) \
                .astype(np.float32)
        elif self.z_distribution == "uniform":
            return np.random.uniform(-1, 1, (batchsize, self.n_hidden, 1, 1)) \
                .astype(np.float32)
        else:
            raise Exception("unknown z distribution: %s" % self.z_distribution)

    def __call__(self, x):
        h = F.reshape(F.relu(self.l0(x)), (x.data.shape[0], self.n_hidden, self.bottom_width, self.bottom_width))
        h = self.r0(h)
        h = self.r1(h)
        h = self.r2(h)
        h = self.bn2(F.relu(h))
        h = F.tanh(self.c3(h))
        return h


# common discriminators

class DCGANDiscriminator(chainer.Chain):
    def __init__(self, bottom_width=4, ch=512, wscale=0.02, output_dim=1):
        w = chainer.initializers.Normal(wscale)
        super(DCGANDiscriminator, self).__init__()
        with self.init_scope():
            self.c0_0 = L.Convolution2D(3, ch // 8, 3, 1, 1, initialW=w)
            self.c0_1 = L.Convolution2D(ch // 8, ch // 4, 4, 2, 1, initialW=w)
            self.c1_0 = L.Convolution2D(ch // 4, ch // 4, 3, 1, 1, initialW=w)
            self.c1_1 = L.Convolution2D(ch // 4, ch // 2, 4, 2, 1, initialW=w)
            self.c2_0 = L.Convolution2D(ch // 2, ch // 2, 3, 1, 1, initialW=w)
            self.c2_1 = L.Convolution2D(ch // 2, ch // 1, 4, 2, 1, initialW=w)
            self.c3_0 = L.Convolution2D(ch // 1, ch // 1, 3, 1, 1, initialW=w)
            self.l4 = L.Linear(bottom_width * bottom_width * ch, output_dim, initialW=w)
            self.bn0_1 = L.BatchNormalization(ch // 4, use_gamma=False)
            self.bn1_0 = L.BatchNormalization(ch // 4, use_gamma=False)
            self.bn1_1 = L.BatchNormalization(ch // 2, use_gamma=False)
            self.bn2_0 = L.BatchNormalization(ch // 2, use_gamma=False)
            self.bn2_1 = L.BatchNormalization(ch // 1, use_gamma=False)
            self.bn3_0 = L.BatchNormalization(ch // 1, use_gamma=False)

    def __call__(self, x):
        h = F.leaky_relu(self.c0_0(x))
        h = F.leaky_relu(self.bn0_1(self.c0_1(h)))
        h = F.leaky_relu(self.bn1_0(self.c1_0(h)))
        h = F.leaky_relu(self.bn1_1(self.c1_1(h)))
        h = F.leaky_relu(self.bn2_0(self.c2_0(h)))
        h = F.leaky_relu(self.bn2_1(self.c2_1(h)))
        h = F.leaky_relu(self.bn3_0(self.c3_0(h)))
        return self.l4(h)


class SNDCGANDiscriminator(chainer.Chain):
    def __init__(self, bottom_width=4, ch=512, wscale=0.02, output_dim=1):
        w = chainer.initializers.Normal(wscale)
        super(SNDCGANDiscriminator, self).__init__()
        with self.init_scope():
            self.c0_0 = SNConvolution2D(3, ch // 8, 3, 1, 1, initialW=w)
            self.c0_1 = SNConvolution2D(ch // 8, ch // 4, 4, 2, 1, initialW=w)
            self.c1_0 = SNConvolution2D(ch // 4, ch // 4, 3, 1, 1, initialW=w)
            self.c1_1 = SNConvolution2D(ch // 4, ch // 2, 4, 2, 1, initialW=w)
            self.c2_0 = SNConvolution2D(ch // 2, ch // 2, 3, 1, 1, initialW=w)
            self.c2_1 = SNConvolution2D(ch // 2, ch // 1, 4, 2, 1, initialW=w)
            self.c3_0 = SNConvolution2D(ch // 1, ch // 1, 3, 1, 1, initialW=w)
            self.l4 = SNLinear(bottom_width * bottom_width * ch, output_dim, initialW=w)

    def __call__(self, x):
        h = F.leaky_relu(self.c0_0(x))
        h = F.leaky_relu(self.c0_1(h))
        h = F.leaky_relu(self.c1_0(h))
        h = F.leaky_relu(self.c1_1(h))
        h = F.leaky_relu(self.c2_0(h))
        h = F.leaky_relu(self.c2_1(h))
        h = F.leaky_relu(self.c3_0(h))
        return self.l4(h)



class WGANDiscriminator(chainer.Chain):
    def __init__(self, bottom_width=4, ch=512, wscale=0.02, output_dim=1):
        w = chainer.initializers.Normal(wscale)
        super(WGANDiscriminator, self).__init__()
        with self.init_scope():
            self.c0 = L.Convolution2D(3, ch // 8, 3, 1, 1, initialW=w)
            self.c1 = L.Convolution2D(ch // 8, ch // 4, 4, 2, 1, initialW=w)
            self.c1_0 = L.Convolution2D(ch // 4, ch // 4, 3, 1, 1, initialW=w)
            self.c2 = L.Convolution2D(ch // 4, ch // 2, 4, 2, 1, initialW=w)
            self.c2_0 = L.Convolution2D(ch // 2, ch // 2, 3, 1, 1, initialW=w)
            self.c3 = L.Convolution2D(ch // 2, ch // 1, 4, 2, 1, initialW=w)
            self.c3_0 = L.Convolution2D(ch // 1, ch // 1, 3, 1, 1, initialW=w)
            self.l4 = L.Linear(bottom_width * bottom_width * ch, output_dim, initialW=w)

    def __call__(self, x):
        self.x = x
        self.h0 = F.leaky_relu(self.c0(self.x))
        self.h1 = F.leaky_relu(self.c1(self.h0))
        self.h2 = F.leaky_relu(self.c1_0(self.h1))
        self.h3 = F.leaky_relu(self.c2(self.h2))
        self.h4 = F.leaky_relu(self.c2_0(self.h3))
        self.h5 = F.leaky_relu(self.c3(self.h4))
        self.h6 = F.leaky_relu(self.c3_0(self.h5))
        return self.l4(self.h6)

    def differentiable_backward(self, x):
        g = backward_linear(self.h6, x, self.l4)
        g = F.reshape(g, (x.shape[0], 512, 4, 4))
        g = backward_leaky_relu(self.h6, g, 0.2)
        g = backward_convolution(self.h5, g, self.c3_0)
        g = backward_leaky_relu(self.h5, g, 0.2)
        g = backward_convolution(self.h4, g, self.c3)
        g = backward_leaky_relu(self.h4, g, 0.2)
        g = backward_convolution(self.h3, g, self.c2_0)
        g = backward_leaky_relu(self.h3, g, 0.2)
        g = backward_convolution(self.h2, g, self.c2)
        g = backward_leaky_relu(self.h2, g, 0.2)
        g = backward_convolution(self.h1, g, self.c1_0)
        g = backward_leaky_relu(self.h1, g, 0.2)
        g = backward_convolution(self.h0, g, self.c1)
        g = backward_leaky_relu(self.h0, g, 0.2)
        g = backward_convolution(self.x, g, self.c0)
        return g


class DownResBlock1(chainer.Chain):
    """
        pre activation residual block
    """

    def __init__(self, ch):
        w = chainer.initializers.Normal(0.02)
        super(DownResBlock1, self).__init__()
        with self.init_scope():
            self.c0 = L.Convolution2D(3, ch, 3, 1, 1, initialW=w)
            self.c1 = L.Convolution2D(ch, ch, 4, 2, 1, initialW=w)
            self.cs = L.Convolution2D(3, ch, 4, 2, 1, initialW=w)

    def __call__(self, x):
        self.h0 = x
        self.h1 = self.c0((self.h0))
        self.h2 = self.c1(F.relu(self.h1))
        self.h3 = self.cs(self.h0)
        self.h4 = self.h2 + self.h3
        return self.h4

    def differentiable_backward(self, g):
        gs = backward_convolution(self.h0, g, self.cs)
        g = backward_convolution(self.h1, g, self.c1)
        g = backward_leaky_relu(self.h1, g, 0.0)
        g = backward_convolution(self.h0, g, self.c0)
        # g = backward_leaky_relu(self.h0, g, 0.0)
        g = g + gs
        return g


class DownResBlock2(chainer.Chain):
    """
        pre activation residual block
    """

    def __init__(self, ch):
        w = chainer.initializers.Normal(0.02)
        super(DownResBlock2, self).__init__()
        with self.init_scope():
            self.c0 = L.Convolution2D(ch, ch, 3, 1, 1, initialW=w)
            self.c1 = L.Convolution2D(ch, ch, 4, 2, 1, initialW=w)
            self.cs = L.Convolution2D(ch, ch, 4, 2, 1, initialW=w)

    def __call__(self, x):
        self.h0 = x
        self.h1 = self.c0(F.relu(self.h0))
        self.h2 = self.c1(F.relu(self.h1))
        self.h3 = self.cs(self.h0)
        self.h4 = self.h2 + self.h3
        return self.h4

    def differentiable_backward(self, g):
        gs = backward_convolution(self.h0, g, self.cs)
        g = backward_convolution(self.h1, g, self.c1)
        g = backward_leaky_relu(self.h1, g, 0.0)
        g = backward_convolution(self.h0, g, self.c0)
        g = backward_leaky_relu(self.h0, g, 0.0)
        g = g + gs
        return g


class DownResBlock3(chainer.Chain):
    """
        pre activation residual block
    """

    def __init__(self, ch):
        w = chainer.initializers.Normal(0.02)
        super(DownResBlock3, self).__init__()
        with self.init_scope():
            self.c0 = L.Convolution2D(ch, ch, 3, 1, 1, initialW=w)
            self.c1 = L.Convolution2D(ch, ch, 3, 1, 1, initialW=w)

    def __call__(self, x):
        self.h0 = x
        self.h1 = self.c0(F.relu(self.h0))
        self.h2 = self.c1(F.relu(self.h1))
        self.h4 = self.h2 + self.h0
        return self.h4

    def differentiable_backward(self, g):
        gs = g
        g = backward_convolution(self.h1, g, self.c1)
        g = backward_leaky_relu(self.h1, g, 0.0)
        g = backward_convolution(self.h0, g, self.c0)
        g = backward_leaky_relu(self.h0, g, 0.0)
        g = g + gs
        return g


class ResnetDiscriminator(chainer.Chain):
    def __init__(self, bottom_width=8, ch=128, wscale=0.02, output_dim=1):
        w = chainer.initializers.Normal(wscale)
        super(ResnetDiscriminator, self).__init__()
        self.bottom_width = bottom_width
        self.ch = ch
        with self.init_scope():
            self.r0 = DownResBlock1(128)
            self.r1 = DownResBlock2(128)
            self.r2 = DownResBlock3(128)
            self.r3 = DownResBlock3(128)
            self.l4 = L.Linear(bottom_width * bottom_width * ch, output_dim, initialW=w)

    def __call__(self, x):
        self.x = x
        self.h1 = self.r0(self.x)
        self.h2 = self.r1(self.h1)
        self.h3 = self.r2(self.h2)
        self.h4 = self.r3(self.h3)
        return self.l4(F.relu(self.h4))

    def differentiable_backward(self, x):
        g = backward_linear(self.h4, x, self.l4)
        g = F.reshape(g, (x.shape[0], self.ch, self.bottom_width, self.bottom_width))
        g = backward_leaky_relu(self.h4, g, 0.0)
        g = self.r3.differentiable_backward(g)
        g = self.r2.differentiable_backward(g)
        g = self.r1.differentiable_backward(g)
        g = self.r0.differentiable_backward(g)
        return g
