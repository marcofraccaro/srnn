Sequential Neural Models with Stochastic Layers
=======
This repository contains the implementation of the Stochastic Recurrent Neural Network (SRNN) model described in

*Sequential Neural Models with Stochastic Layers*
Marco Fraccaro, Søren Kaae Sønderby, Ulrich Paquet, Ole Winther
*NIPS 2016, arXiv preprint [arXiv:1605.07571](https://arxiv.org/abs/1605.07571)*

The implementation is built on the [Theano](<https://github.com/Theano/Theano>), [Lasagne](<http://github.com/Lasagne/Lasagne>) and [Parmesan](<https://github.com/casperkaae/parmesan>) libraries.

If you have questions on the code, feel free to create a Github issue or contact us: Marco Fraccaro (marfra@dtu.dk), 
Søren Kaae Sønderby (skaaesonderby@gmail.com).


Installation
------------
Please make sure you have installed the requirements before executing the python scripts.

```
  pip install numpy
  pip install matplotlib
  pip install https://github.com/Theano/Theano/archive/master.zip
  pip install https://github.com/Lasagne/Lasagne/archive/master.zip
  git clone https://github.com/casperkaae/parmesan.git
  cd parmesan
  python setup.py develop
```


Examples
-------------
The repository includes code to run the SRNN on polyphonic music and TIMIT data. 

* *MainSRNN_midi.py* runs the polyphonic music experiment on the *Muse* data set.
* *MainSRNN_timit.py* runs the TIMIT experiment. Unfortunately we cannot release the TIMIT data, that needs to be obtained
 from https://catalog.ldc.upenn.edu/ldc93s1. We have released however our preprocessing script, *timit_for_srnn.py*.

Further details on the experimental setup can be found in the code and in the supplementary material of the paper.
