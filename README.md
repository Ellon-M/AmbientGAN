# AmbientGAN


**Overview**


A Generative Model that parses piano notes from MIDI files and uses them as input to generate new note sequences.
Most of the input MIDI files are sourced from the [ADL Piano MIDI Dataset](https://github.com/lucasnfe/adl-piano-midi).

The adversarial network here follows the default architecture of simultaneously training a generative model `G`
that captures the data distribution, and a discriminative model `D` that estimates the probability that a sample came from the training data rather than `G`. 

Additionally, a dot-product **self-attention** layer is employed and applied to the last layers of both the generator and the discriminator.



**Req**: Python 3, Pytorch 1.10, CUDA 10.2.



**Training Loss**

Training time on GPU: 8-10 Minutes for 2000 Epochs.

![alt text](https://res.cloudinary.com/denphvygd/image/upload/v1646280805/ambience/loss_per_epoch2_xebbst.png)


**Results**

*MIDI tempo used for all outputs ranged between 250000 and 500000.*
*Time sig, key sig and any necessary meta messages or lyrics found in MIDI files are pre-defined, and are not part of the model output.*

Samples outputs are [here](https://ellon-m.github.io/AmbientGAN/).


**TODO**


- [ ] Implement S.A Layer.
- [ ] Fully Unsupervised Setting.





