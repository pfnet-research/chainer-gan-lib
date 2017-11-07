import argparse
import os
import sys

import chainer
from chainer import training
from chainer.training import extension
from chainer.training import extensions

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.abspath(os.path.dirname(__file__)) + os.path.sep + os.path.pardir)

from common.dataset import Cifar10Dataset
from evaluation import sample_generate, sample_generate_light, calc_inception, calc_FID
from common.record import record_setting

from net import Discriminator, Generator
from updater import Updater

from common.misc import copy_param

def main():
    parser = argparse.ArgumentParser(
        description='Train script')
    parser.add_argument('--batchsize', '-b', type=int, default=16)
    parser.add_argument('--max_iter', '-m', type=int, default=400000)
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--snapshot_interval', type=int, default=25000,
                        help='Interval of snapshot')
    parser.add_argument('--evaluation_interval', type=int, default=50000,
                        help='Interval of evaluation')
    parser.add_argument('--out_image_interval', type=int, default=12500,
                        help='Interval of evaluation')
    parser.add_argument('--stage_interval', type=int, default=400000,
                        help='Interval of stage progress')
    parser.add_argument('--display_interval', type=int, default=100,
                        help='Interval of displaying log to console')
    parser.add_argument('--n_dis', type=int, default=1,
                        help='number of discriminator update per generator update')
    parser.add_argument('--lam', type=float, default=10,
                        help='gradient penalty')
    parser.add_argument('--gamma', type=float, default=750,
                        help='gradient penalty')
    parser.add_argument('--model', type=str, default="dcgan",
                        help='Network architecture (dcgan or resnet)')
    parser.add_argument('--pooling_comp', type=float, default=1.0,
                        help='compensation')
    parser.add_argument('--pretrained_generator', type=str, default="")
    parser.add_argument('--pretrained_discriminator', type=str, default="")
    parser.add_argument('--initial_stage', type=float, default=0.0)
    parser.add_argument('--generator_smoothing', type=float, default=0.999)

    args = parser.parse_args()
    record_setting(args.out)

    report_keys = ["stage", "loss_dis", "loss_gp", "loss_gen", "g", "inception_mean", "inception_std", "FID"]
    max_iter = args.max_iter

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()

    generator = Generator()
    generator_smooth = Generator()
    discriminator = Discriminator(pooling_comp=args.pooling_comp)

    # select GPU
    if args.gpu >= 0:
        generator.to_gpu()
        generator_smooth.to_gpu()
        discriminator.to_gpu()
        print("use gpu {}".format(args.gpu))

    if args.pretrained_generator != "":
        chainer.serializers.load_npz(args.pretrained_generator, generator)
    if args.pretrained_discriminator != "":
        chainer.serializers.load_npz(args.pretrained_discriminator, discriminator)
    copy_param(generator_smooth, generator)

    # Setup an optimizer
    def make_optimizer(model, alpha=0.001, beta1=0.0, beta2=0.99):
        optimizer = chainer.optimizers.Adam(alpha=alpha, beta1=beta1, beta2=beta2)
        optimizer.setup(model)
        # optimizer.add_hook(chainer.optimizer.WeightDecay(0.00001), 'hook_dec')
        return optimizer

    opt_gen = make_optimizer(generator)
    opt_dis = make_optimizer(discriminator)

    train_dataset = Cifar10Dataset()
    train_iter = chainer.iterators.SerialIterator(train_dataset, args.batchsize)

    # Set up a trainer
    updater = Updater(
        models=(generator, discriminator, generator_smooth),
        iterator={
            'main': train_iter},
        optimizer={
            'opt_gen': opt_gen,
            'opt_dis': opt_dis},
        device=args.gpu,
        n_dis=args.n_dis,
        lam=args.lam,
        gamma=args.gamma,
        smoothing=args.generator_smoothing,
        initial_stage=args.initial_stage,
        stage_interval=args.stage_interval
    )
    trainer = training.Trainer(updater, (max_iter, 'iteration'), out=args.out)
    trainer.extend(extensions.snapshot_object(
        generator, 'generator_{.updater.iteration}.npz'), trigger=(args.snapshot_interval, 'iteration'))
    trainer.extend(extensions.snapshot_object(
        generator_smooth, 'generator_smooth_{.updater.iteration}.npz'), trigger=(args.snapshot_interval, 'iteration'))
    trainer.extend(extensions.snapshot_object(
        discriminator, 'discriminator_{.updater.iteration}.npz'), trigger=(args.snapshot_interval, 'iteration'))

    trainer.extend(extensions.LogReport(keys=report_keys,
                                        trigger=(args.display_interval, 'iteration')))
    trainer.extend(extensions.PrintReport(report_keys), trigger=(args.display_interval, 'iteration'))
    trainer.extend(sample_generate(generator_smooth, args.out), trigger=(args.out_image_interval, 'iteration'),
                   priority=extension.PRIORITY_WRITER)
    trainer.extend(sample_generate_light(generator_smooth, args.out), trigger=(args.evaluation_interval // 10, 'iteration'),
                   priority=extension.PRIORITY_WRITER)
    trainer.extend(calc_inception(generator_smooth), trigger=(args.evaluation_interval, 'iteration'),
                   priority=extension.PRIORITY_WRITER)
    trainer.extend(calc_FID(generator_smooth), trigger=(args.evaluation_interval, 'iteration'),
                   priority=extension.PRIORITY_WRITER)
    trainer.extend(extensions.ProgressBar(update_interval=10))

    # Run the training
    trainer.run()


if __name__ == '__main__':
    main()
