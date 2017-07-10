import numpy as np

import chainer
import chainer.functions as F
from chainer import Variable


class Updater(chainer.training.StandardUpdater):
    def __init__(self, *args, **kwargs):
        self.gen, self.dis = kwargs.pop('models')
        self.n_dis = kwargs.pop('n_dis')
        self.lam = kwargs.pop('lam')
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
            x_real = Variable(xp.asarray(x))
            y_real = self.dis(x_real)

            z = Variable(xp.asarray(self.gen.make_hidden(batchsize)))
            x_fake = self.gen(z)
            y_fake = self.dis(x_fake)

            if i == 0:
                loss_gen = F.sum(-y_fake) / batchsize
                self.gen.cleargrads()
                loss_gen.backward()
                gen_optimizer.update()
                chainer.reporter.report({'loss_gen': loss_gen})
            x_fake.unchain_backward()

            eps = xp.random.uniform(0, 1, size=batchsize).astype("f")[:, None, None, None]
            x_mid = eps * x_real + (1.0 - eps) * x_fake

            x_mid_v = Variable(x_mid.data)
            y_mid = self.dis(x_mid_v)
            dydx = self.dis.differentiable_backward(xp.ones_like(y_mid.data))
            dydx = F.sqrt(F.sum(dydx ** 2, axis=(1, 2, 3)))
            loss_gp = self.lam * F.mean_squared_error(dydx, xp.ones_like(dydx.data))

            loss_dis = F.sum(-y_real) / batchsize
            loss_dis += F.sum(y_fake) / batchsize

            self.dis.cleargrads()
            loss_dis.backward()
            loss_gp.backward()
            dis_optimizer.update()

            chainer.reporter.report({'loss_dis': loss_dis})
            chainer.reporter.report({'loss_gp': loss_gp})
            chainer.reporter.report({'g': F.mean(dydx)})
