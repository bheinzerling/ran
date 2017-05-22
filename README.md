# Recurrent Additive Networks

This is a PyTorch implementation of Recurrent Additive Networks (RAN) by Kenton Lee, 
Omer Levy, and Luke Zettlemoyer:

http://www.kentonl.com/pub/llz.2017.pdf

The RAN model is implemented in `ran.py`.


Code for running Penn Tree Bank (PTB) experiments is taken from:

https://github.com/pytorch/examples/tree/master/word_language_model


To run PTB experiments, clone this repository: 

```
git clone https://github.com/bheinzerling/ran
```

and then do:

```
cd ran
python main.py --cuda --emsize 256 --nhid 1024 --dropout 0.5 --epochs 100 --nlayers 1 --batch-size 512
```

