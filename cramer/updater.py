import chainer
import chainer.functions as F
from chainer import Variable

import numpy as np


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
            h_real = self.dis(x_real)

            z = Variable(xp.asarray(self.gen.make_hidden(batchsize)))
            x_fake1 = self.gen(z)
            h_fake1 = self.dis(x_fake1)

            z2 = Variable(xp.asarray(self.gen.make_hidden(batchsize)))
            x_fake2 = self.gen(z2)
            h_fake2 = self.dis(x_fake2)

            def l2_distance(a, b):
                return F.sqrt(F.sum((a - b) ** 2, axis=1, keepdims=True))

            def backward_l2_distance(g, a, b):
                out = F.broadcast_to(l2_distance(a, b), a.data.shape)
                g = F.broadcast_to(g, a.data.shape)
                return g * (a - b) / out, g * (b - a) / out

            def energy_distance(r, f1, f2):
                ret = l2_distance(r, f1)
                ret += l2_distance(r, f2)
                ret -= l2_distance(f1, f2)
                return F.mean(ret)

            def critic(a, b):
                return l2_distance(a, b) - l2_distance(a, xp.zeros_like(a.data))

            def backward_critic(g, a, b):
                ga0, gb0 = backward_l2_distance(g, a, b)
                ga1, gb1 = backward_l2_distance(g, a, xp.zeros_like(a.data))
                return ga0 - ga1, gb0 - gb1

            critic_real = critic(h_real, h_fake2)
            critic_fake = critic(h_fake1, h_fake2)

            loss_surrogate = F.mean(critic_real - critic_fake)

            if i == 0:
                loss_gen = energy_distance(h_real, h_fake1, h_fake2)
                self.gen.cleargrads()
                loss_gen.backward()
                gen_optimizer.update()
                chainer.reporter.report({'loss_gen': loss_gen})
            x_fake1.unchain_backward()
            x_fake2.unchain_backward()

            eps = xp.random.uniform(0, 1, size=batchsize).astype("f")[:, None, None, None]
            x_mid = eps * x_real + (1.0 - eps) * x_fake1
            h_mid = Variable(self.dis(x_mid).data)
            critic_mid = critic(h_mid, h_fake2.data)

            # calc gradient penalty
            g = Variable(xp.ones_like(critic_mid.data))
            dydh, _ = backward_critic(g, h_mid, h_fake2.data)
            dydx = self.dis.differentiable_backward(dydh)
            dydx_norm = F.sqrt(F.sum(dydx ** 2, axis=(1, 2, 3)))
            loss_gp = self.lam * F.mean_squared_error(dydx_norm, xp.ones_like(dydx_norm.data))

            self.dis.cleargrads()
            (-loss_surrogate).backward()
            loss_gp.backward()
            dis_optimizer.update()

            chainer.reporter.report({'loss_dis': loss_surrogate})
            chainer.reporter.report({'loss_gp': loss_gp})
            chainer.reporter.report({'g': F.mean(dydx_norm)})
