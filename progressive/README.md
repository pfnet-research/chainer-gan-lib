# Progressive Growing of GANs
An implementation of [_Karras et al., "Progressive Growing of GANs for Improved Quality, Stability, and Variation", 2017_](https://arxiv.org/abs/1710.10196) using Chainer.

## Dependencies

The code for progressive GAN depends on more recent version of `chainer` and `cupy`. 
Specifically, the code was tested to work with the following versions:

chainer == 4.0.0b2

cupy == 4.0.0b2

To install these versions as of 2018.01, you need to install from source. Links to installation instructions: [cupy](https://docs-cupy.chainer.org/en/stable/install.html#install-cupy-from-source), [chainer](https://docs.chainer.org/en/stable/install.html#install-chainer-from-source). For both packages install the most recent version from github. 

Results
-------
CIFAR10 (inception score: 8.5)
<p align="center">
  <img src="../images/progressive.png" height="320" width="320" alt="CIFAR10"/>
</p>

Illustration (512x512, cherry picked)
<p align="center">
  <img src="../images/interpolation1.gif" height="128" width="128" alt="int1"/> 
  <img src="../images/interpolation2.gif" height="128" width="128" alt="int2"/> 
  <img src="../images/yurinterpolation.gif" height="128" width="128" alt="int3"/> 
  <br>
  <img src="../images/pggan_s.png" height="768" width="768" alt="solo"/> 
  <br>
  <img src="../images/pggan_y.png" height="768" width="768" alt="yuri"/>
  <br>
</p>
