import chainer
import chainer.functions as F
from chainer import Variable

import numpy as np


class Updater(chainer.training.StandardUpdater):
    def __init__(self, *args, **kwargs):
        self.gen, self.dis = kwargs.pop('models')
        self.gamma = kwargs.pop('gamma')
        self.kt = 0.
        super(Updater, self).__init__(*args, **kwargs)

    def update_core(self):
        gen_optimizer = self.get_optimizer('opt_gen')
        dis_optimizer = self.get_optimizer('opt_dis')
        xp = self.gen.xp

        batch = self.get_iterator('main').next()
        batchsize = len(batch)
        x = []
        for i in range(batchsize):
            x.append(np.asarray(batch[i]).astype("f"))
        x_real = Variable(xp.asarray(x))
        y_real = self.dis(x_real)

        z = Variable(xp.asarray(self.gen.make_hidden(batchsize)))
        x_fake = self.gen(z)
        y_fake = self.dis(x_fake)

        loss_dis = y_real - self.kt * y_fake
        loss_gen = y_fake
        self.kt = self.kt + 0.001 * (self.gamma * y_real.data.get() - y_fake.data.get())
        self.kt = np.clip(self.kt, 0, 1)
        measure = y_real.data.get() + np.abs(self.gamma * y_real.data.get() - y_fake.data.get())

        self.gen.cleargrads()
        loss_gen.backward()
        gen_optimizer.update()
        x_fake.unchain_backward()

        self.dis.cleargrads()
        loss_dis.backward()
        dis_optimizer.update()

        chainer.reporter.report({'loss_gen': loss_gen})
        chainer.reporter.report({'loss_dis': loss_dis})
        chainer.reporter.report({'kt': self.kt})
        chainer.reporter.report({'measure': measure})
