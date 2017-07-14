import numpy as np

import chainer
import chainer.functions as F
from chainer import Variable


class Updater(chainer.training.StandardUpdater):
    def __init__(self, *args, **kwargs):
        self.gen, self.dis = kwargs.pop('models')
        self.n_dis = kwargs.pop('n_dis')
        super(Updater, self).__init__(*args, **kwargs)

    def update_core(self):
        gen_optimizer = self.get_optimizer('opt_gen')
        dis_optimizer = self.get_optimizer('opt_dis')
        xp = self.gen.xp

        for i in range(self.n_dis):
            batch = self.get_iterator('main').next()
            batchsize = len(batch)
            x = []
            for j in range(batchsize):
                x.append(np.asarray(batch[j]).astype("f"))

            if i == 0:
                z = Variable(xp.asarray(self.gen.make_hidden(batchsize)))
                x_fake = self.gen(z)
                y_fake = self.dis(x_fake)
                loss_gen = F.sum(F.softplus(-y_fake)) / batchsize
                self.gen.cleargrads()
                loss_gen.backward()
                gen_optimizer.update()
                chainer.reporter.report({'loss_gen': loss_gen})

            x_real = Variable(xp.asarray(x))
            y_real = self.dis(x_real)
            z = Variable(xp.asarray(self.gen.make_hidden(batchsize)))
            x_fake = self.gen(z)
            y_fake = self.dis(x_fake)
            x_fake.unchain_backward()

            loss_dis = F.sum(F.softplus(-y_real)) / batchsize
            loss_dis += F.sum(F.softplus(y_fake)) / batchsize

            self.dis.cleargrads()
            loss_dis.backward()
            dis_optimizer.update()

            chainer.reporter.report({'loss_dis': loss_dis})
