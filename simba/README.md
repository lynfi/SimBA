Modified by me.

This repository contains code for the ICML 2019 paper:

Chuan Guo, Jacob R. Gardner, Yurong You, Andrew G. Wilson, Kilian Q. Weinberger. Simple Black-box Adversarial Attacks.
https://arxiv.org/abs/1905.07121

To run SimBA (pixel attack):
```
python simba_cifar.py --gpu 0 --pixel --idx 0
```
For targeted attack, add flag --targeted.
For DCT attack, delete flag --pixel.

Need to put the checkpoint in `.\checkpoint` folder
