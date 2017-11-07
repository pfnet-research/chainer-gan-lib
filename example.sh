python train.py --gpu 0 --algorithm dcgan --out result_dcgan
python train.py --gpu 0 --algorithm minibatch_discrimination --out result_minibatch_discrimination  --adam_alpha 0.0001 --adam_beta1 0.0 --adam_beta2 0.9
python train.py --gpu 0 --algorithm dfm --out result_dfm --adam_alpha 0.0001 --adam_beta1 0.5 --adam_beta2 0.9
python train.py --gpu 0 --algorithm began --out result_began --gamma 0.5
python train.py --gpu 0 --algorithm cramer --out result_cramer --adam_alpha 0.0001 --adam_beta1 0.5 --adam_beta2 0.9 --output_dim 256 --n_dis 5 --lam 10
python train.py --gpu 0 --algorithm dragan --out result_dragan --adam_alpha 0.0001 --adam_beta1 0.0 --adam_beta2 0.9 --n_dis 2 --lam 10
python train.py --gpu 0 --algorithm wgan_gp --out result_wgan_gp --n_dis 5 --lam 10
python train.py --gpu 0 --algorithm stdgan --architecture sndcgan --out result_sndcgan --n_dis 1 --adam_beta1 0.5 --adam_beta2 0.999 
python progressive/train.py --gpu 0 --out result_progressive
