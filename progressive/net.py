import os
import sys
import math

import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.abspath(os.path.dirname(__file__)) + os.path.sep + os.path.pardir)


def feature_vector_normalization(x, eps=1e-8):
    # x: (B, C, H, W)
    alpha = 1.0 / F.sqrt(F.mean(x*x, axis=1, keepdims=True) + eps)
    return F.broadcast_to(alpha, x.data.shape) * x


class EqualizedConv2d(chainer.Chain):
    def __init__(self, in_ch, out_ch, ksize, stride, pad):
        w = chainer.initializers.Normal(1.0) # equalized learning rate
        self.inv_c = np.sqrt(2.0/(in_ch*ksize**2))
        super(EqualizedConv2d, self).__init__()
        with self.init_scope():
            self.c = L.Convolution2D(in_ch, out_ch, ksize, stride, pad, initialW=w)
    def __call__(self, x):
        return self.c(self.inv_c * x)

class EqualizedLinear(chainer.Chain):
    def __init__(self, in_ch, out_ch):
        w = chainer.initializers.Normal(1.0) # equalized learning rate
        self.inv_c = np.sqrt(2.0/in_ch)
        super(EqualizedLinear, self).__init__()
        with self.init_scope():
            self.c = L.Linear(in_ch, out_ch, initialW=w)
    def __call__(self, x):
        return self.c(self.inv_c * x)

class EqualizedDeconv2d(chainer.Chain):
    def __init__(self, in_ch, out_ch, ksize, stride, pad):
        w = chainer.initializers.Normal(1.0) # equalized learning rate
        self.inv_c = np.sqrt(2.0/(in_ch))
        super(EqualizedDeconv2d, self).__init__()
        with self.init_scope():
            self.c = L.Deconvolution2D(in_ch, out_ch, ksize, stride, pad, initialW=w)
    def __call__(self, x):
        return self.c(self.inv_c * x)

def minibatch_std(x):
    xp = chainer.cuda.get_array_module(x)
    m = F.mean(x, axis=0, keepdims=True)
    # double bp ga dekinai node **2 wo tsukawa nai
    v = F.mean((x - F.broadcast_to(m, x.shape))*(x - F.broadcast_to(m, x.shape)), axis=0, keepdims=True)
    std = F.mean(F.sqrt(v + 1e-8), keepdims=True)
    std = F.broadcast_to(std, (x.shape[0], 1, x.shape[2], x.shape[3]))
    return F.concat([x, std], axis=1)

class GeneratorBlock(chainer.Chain):
    def __init__(self, in_ch, out_ch):
        super(GeneratorBlock, self).__init__()
        with self.init_scope():
            self.c0 = EqualizedConv2d(in_ch, out_ch, 3, 1, 1)
            self.c1 = EqualizedConv2d(out_ch, out_ch, 3, 1, 1)
            #self.toRGB = F.Convolution2D(out_ch, 3, 1, 1, 0)
    def __call__(self, x):
        h = F.unpooling_2d(x, 2, 2, 0, outsize=(x.shape[2]*2, x.shape[3]*2))
        h = F.leaky_relu(feature_vector_normalization(self.c0(h)))
        h = F.leaky_relu(feature_vector_normalization(self.c1(h)))
        return h


class Generator(chainer.Chain):
    def __init__(self, n_hidden=512, ch=512, max_stage=6):
        super(Generator, self).__init__()
        self.n_hidden = n_hidden
        self.max_stage = max_stage
        with self.init_scope():
            #self.c0 = EqualizedDeconv2d(n_hidden, ch, 4, 1, 0)
            self.c0 = EqualizedConv2d(n_hidden, ch, 4, 1, 3)
            self.c1 = EqualizedConv2d(ch, ch, 3, 1, 1)
            self.out0 = EqualizedConv2d(ch, 3, 1, 1, 0)

            self.b1 = GeneratorBlock(ch, ch)
            self.out1 = EqualizedConv2d(ch, 3, 1, 1, 0)
            self.b2 = GeneratorBlock(ch, ch)
            self.out2 = EqualizedConv2d(ch, 3, 1, 1, 0)
            self.b3 = GeneratorBlock(ch, ch//2)
            self.out3 = EqualizedConv2d(ch//2, 3, 1, 1, 0)

    def make_hidden(self, batchsize):
        xp = self.xp
        z = xp.random.normal(size=(batchsize, self.n_hidden, 1, 1)) \
            .astype(np.float32)
        z /= xp.sqrt(xp.sum(z*z, axis=1, keepdims=True)/self.n_hidden + 1e-8)
        return z

    def __call__(self, z, stage):
        # stage0: c0->c1->out0
        # stage1: c0->c1-> (1-a)*(up->out0) + (a)*(b1->out1)
        # stage2: c0->c1->b1->out1
        # stage3: c0->c1->b1-> (1-a)*(up->out1) + (a)*(b2->out2)
        # stage4: c0->c1->b2->out2

        stage = min(stage, self.max_stage)
        alpha = stage - math.floor(stage)
        stage = math.floor(stage)

        h = F.reshape(z,(len(z), self.n_hidden, 1, 1))
        h = F.leaky_relu(feature_vector_normalization(self.c0(h)))
        #h = F.leaky_relu((self.c0(h)))
        h = F.leaky_relu(feature_vector_normalization(self.c1(h)))

        for i in range(1, int(stage//2+1)):
            h = getattr(self, "b%d"%i)(h)

        if int(stage)%2==0:
            out = getattr(self, "out%d"%(stage//2))
            x = out(h)
        else:
            out_prev = getattr(self, "out%d"%(stage//2))
            out_curr = getattr(self, "out%d"%(stage//2+1))
            b_curr = getattr(self, "b%d"%(stage//2+1))

            x_0 = out_prev(F.unpooling_2d(h, 2, 2, 0, outsize=(2*h.shape[2], 2*h.shape[3])))
            x_1 = out_curr(b_curr(h))
            x = (1.0-alpha)*x_0 + alpha*x_1

        if chainer.configuration.config.train:
            return x
        else:
            scale = int(32 // x.data.shape[2])
            return F.unpooling_2d(x, scale, scale, 0, outsize=(32,32))


class DiscriminatorBlock(chainer.Chain):
    # conv-conv-downsample
    def __init__(self, in_ch, out_ch, pooling_comp):
        super(DiscriminatorBlock, self).__init__()
        self.pooling_comp = pooling_comp
        with self.init_scope():
            self.c0 = EqualizedConv2d(in_ch, in_ch, 3, 1, 1)
            self.c1 = EqualizedConv2d(in_ch, out_ch, 3, 1, 1)
            #self.toRGB = F.Convolution2D(out_ch, 3, 1, 1, 0)
    def __call__(self, x):
        h = F.leaky_relu((self.c0(x)))
        h = F.leaky_relu((self.c1(h)))
        h = self.pooling_comp * F.average_pooling_2d(h, 2, 2, 0)
        return h


class Discriminator(chainer.Chain):
    def __init__(self, ch=512, max_stage=6, pooling_comp=1.0):
        super(Discriminator, self).__init__()
        self.max_stage = max_stage
        self.pooling_comp = pooling_comp # compensation of ave_pool is 0.5-Lipshitz
        with self.init_scope():
            self.in3 = EqualizedConv2d(3, ch//2, 1, 1, 0)
            self.b3 = DiscriminatorBlock(ch//2, ch, pooling_comp)
            self.in2 = EqualizedConv2d(3, ch, 1, 1, 0)
            self.b2 = DiscriminatorBlock(ch, ch, pooling_comp)
            self.in1 = EqualizedConv2d(3, ch, 1, 1, 0)
            self.b1 = DiscriminatorBlock(ch, ch, pooling_comp)
            self.in0 = EqualizedConv2d(3, ch, 1, 1, 0)

            self.out0 = EqualizedConv2d(ch+1, ch, 3, 1, 1)
            self.out1 = EqualizedConv2d(ch, ch, 4, 1, 0)
            self.out2 = EqualizedLinear(ch, 1)

    def __call__(self, x, stage):
        # stage0: in0->m_std->out0_0->out0_1->out0_2
        # stage1: (1-a)*(down->in0) + (a)*(in1->b1) ->m_std->out0->out1->out2
        # stage2: in1->b1->m_std->out0_0->out0_1->out0_2
        # stage3: (1-a)*(down->in1) + (a)*(in2->b2) ->b1->m_std->out0->out1->out2
        # stage4: in2->b2->b1->m_std->out0->out1->out2

        stage = min(stage, self.max_stage)
        alpha = stage - math.floor(stage)
        stage = math.floor(stage)
        #print(stage, alpha)
        #print(x.shape)

        if int(stage)%2==0:
            fromRGB = getattr(self, "in%d"%(stage//2))
            h = F.leaky_relu(fromRGB(x))
        else:
            fromRGB0 = getattr(self, "in%d"%(stage//2))
            fromRGB1 = getattr(self, "in%d"%(stage//2+1))
            b1 = getattr(self, "b%d"%(stage//2+1))


            h0 = F.leaky_relu(fromRGB0(self.pooling_comp * F.average_pooling_2d(x, 2, 2, 0)))
            h1 = b1(F.leaky_relu(fromRGB1(x)))
            h = (1-alpha)*h0 + alpha*h1

        #print(h.shape)
        for i in range(int(stage // 2), 0, -1):
            h = getattr(self, "b%d" % i)(h)
            #print(i, h.shape)

        h = minibatch_std(h)
        #print(h.shape)
        h = F.leaky_relu((self.out0(h)))
        #print(h.shape)
        h = F.leaky_relu((self.out1(h)))
        return self.out2(h)
