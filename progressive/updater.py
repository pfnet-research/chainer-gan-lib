import numpy as np
import math
import os, sys

import chainer
import chainer.functions as F
from chainer import Variable

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.abspath(os.path.dirname(__file__)) + os.path.sep + os.path.pardir)
from common.misc import soft_copy_param

class Updater(chainer.training.StandardUpdater):
    def __init__(self, *args, **kwargs):
        self.gen, self.dis, self.gs = kwargs.pop('models')
        self.n_dis = kwargs.pop('n_dis')
        self.lam = kwargs.pop('lam')
        self.gamma = kwargs.pop('gamma')
        self.smoothing = kwargs.pop('smoothing')
        self.stage_interval = kwargs.pop('stage_interval')
        self.initial_stage = kwargs.pop('initial_stage')
        self.counter = math.ceil(self.initial_stage * self.stage_interval)
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

            self.stage = self.counter / self.stage_interval

            if math.floor(self.stage)%2==0:
                reso = min(32, 4 * 2**(((math.floor(self.stage)+1)//2)))
                scale = max(1, 32//reso)
                if scale>1:
                    x_real = F.average_pooling_2d(x_real, scale, scale, 0)
            else:
                alpha = self.stage - math.floor(self.stage)
                reso_low = min(32, 4 * 2**(((math.floor(self.stage))//2)))
                reso_high = min(32, 4 * 2**(((math.floor(self.stage)+1)//2)))
                scale_low = max(1, 32//reso_low)
                scale_high = max(1, 32//reso_high)
                if scale_low>1:
                    x_real_low = F.unpooling_2d(
                        F.average_pooling_2d(x_real, scale_low, scale_low, 0),
                        2, 2, 0, outsize=(reso_high, reso_high))
                    x_real_high = F.average_pooling_2d(x_real, scale_high, scale_high, 0)
                    x_real = (1-alpha)*x_real_low + alpha*x_real_high


            y_real = self.dis(x_real, stage=self.stage)

            z = Variable(xp.asarray(self.gen.make_hidden(batchsize)))
            x_fake = self.gen(z, stage=self.stage)
            y_fake = self.dis(x_fake, stage=self.stage)

            x_fake.unchain_backward()

            eps = xp.random.uniform(0, 1, size=batchsize).astype("f")[:, None, None, None]
            x_mid = eps * x_real + (1.0 - eps) * x_fake

            x_mid_v = Variable(x_mid.data)
            y_mid = F.sum(self.dis(x_mid_v, stage=self.stage))

            dydx, = chainer.grad([y_mid], [x_mid_v], enable_double_backprop=True)
            dydx = F.sqrt(F.sum(dydx*dydx, axis=(1, 2, 3)))
            loss_gp = self.lam * F.mean_squared_error(dydx, self.gamma * xp.ones_like(dydx.data)) * (1.0/self.gamma**2)

            loss_dis = F.sum(-y_real) / batchsize
            loss_dis += F.sum(y_fake) / batchsize

            # prevent drift factor
            loss_dis += 0.001 * F.sum(y_real**2) / batchsize

            loss_dis_total = loss_dis + loss_gp
            self.dis.cleargrads()
            loss_dis_total.backward()
            dis_optimizer.update()
            loss_dis_total.unchain_backward()

            # train generator
            z = Variable(xp.asarray(self.gen.make_hidden(batchsize)))
            x_fake = self.gen(z, stage=self.stage)
            y_fake = self.dis(x_fake, stage=self.stage)
            loss_gen = F.sum(-y_fake) / batchsize
            self.gen.cleargrads()
            loss_gen.backward()
            gen_optimizer.update()

            # update smoothed generator
            soft_copy_param(self.gs, self.gen, 1.0-self.smoothing)

            chainer.reporter.report({'loss_dis': loss_dis})
            chainer.reporter.report({'loss_gen': loss_gen})
            chainer.reporter.report({'loss_gp': loss_gp})
            chainer.reporter.report({'g': F.mean(dydx)})
            chainer.reporter.report({'stage': self.stage})

            self.counter += batchsize
