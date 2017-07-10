import numpy as np

import chainer
import chainer.functions as F
from chainer import Variable


class Updater(chainer.training.StandardUpdater):

    def __init__(self, *args, **kwargs):
        self.gen, self.dis, self.den = kwargs.pop('models')
        super(Updater, self).__init__(*args, **kwargs)

    def update_core(self):
        gen_optimizer = self.get_optimizer('opt_gen')
        dis_optimizer = self.get_optimizer('opt_dis')
        den_optimizer = self.get_optimizer('opt_den')
        xp = self.gen.xp

        batch = self.get_iterator('main').next()
        batchsize = len(batch)
        x = []
        for i in range(batchsize):
            x.append(np.asarray(batch[i]).astype("f"))
        x_real = Variable(xp.asarray(x))

        f_real, y_real = self.dis(x_real)
        f_real_noise = f_real.data + 1.0 * xp.random.randn(*f_real.data.shape).astype("f")
        f_real_rec = self.den(f_real_noise)

        z = Variable(xp.asarray(self.gen.make_hidden(batchsize)))
        x_fake = self.gen(z)
        f_fake, y_fake = self.dis(x_fake)
        with chainer.using_config('train', False):
            f_fake_rec = self.den(f_fake)

        loss_dis = F.sum(F.softplus(-y_real)) / batchsize
        loss_dis += F.sum(F.softplus(y_fake)) / batchsize

        loss_gen = F.sum(F.softplus(-y_fake)) / batchsize
        loss_gen += 0.03 * F.mean_squared_error(f_fake, f_fake_rec.data)

        loss_den = F.mean_squared_error(f_real_rec, f_real.data)

        self.den.cleargrads()
        loss_den.backward()

        self.gen.cleargrads()
        loss_gen.backward()
        x_fake.unchain_backward()

        self.dis.cleargrads()
        loss_dis.backward()

        den_optimizer.update()
        gen_optimizer.update()
        dis_optimizer.update()

        chainer.reporter.report({'loss_gen': loss_gen})
        chainer.reporter.report({'loss_dis': loss_dis})
        chainer.reporter.report({'loss_den': loss_den})
