# Recurrent Additive Networks

Note: This code is not up-to-date, please refer to the implementation by the original authors: https://github.com/kentonl/ran

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
python main.py --cuda --emsize 256 --nhid 1024 --dropout 0.5 --epochs 100 --nlayers 1 --batch-size 512 --model RAN
```

This should result in a test set perplexity which roughly agrees with the RAN (tanh) result reported in the paper:

```
End of training | test loss  4.78 | test ppl   119.40
```

Better results can be achieved with smaller batch sizes, e.g. with batch size 40:

```
End of training | test loss  4.45 | test ppl    85.24
```

batch size 20:

```
| End of training | test loss  4.42 | test ppl    83.42
```

batch size 10:

```
| End of training | test loss  4.41 | test ppl    82.62
```

batch size 5:

```
| End of training | test loss  4.49 | test ppl    89.21
```
