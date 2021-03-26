# Neural Arithmetic with MC-LSTM

This repository contains the code for part of the experiments in the MC-LSTM paper.
More specifically, the experiments that benchmark MC-LSTM on arithmetic tasks.
The code for other experiments can be found in the [main repo](https://github.com/ml-jku/mc-lstm).

For these experiments we used the code base from 
[Madsen and Johansen (2021)](https://openreview.net/forum?id=H1gNOeHKPS).
The starting point for this code is tagged `madsen`.
To get an overview of the changes we made, you can run `git diff madsen`.

The pytorch module for MC-LSTM can be found in `stable_nalu/layer/mclstm.py`,
the fully-connected layer with mass conservation is in `stable_nalu/layer/mcfc.py`.

### Reproduction

Experiments should be reproducible by running the shell scripts `mclstm_*.sh`.
The bash scripts repeatedly train different networks using pytorch (python),
and then call R scripts to generate a table or figure to summarise the experiment.
The relevant information for every single run is also logged to tensorboard.

###### Requirements

You should be able to use the `setup.py` file to install the code on your system.
Alternatively, you can install the requirements in `setup.py` manually
and run the code by setting the `PYTHONPATH` to the top-level directory.

To run the scripts as they are, you would need a graphics card with at least 11GB VRAM.
Furthermore experiments were performed on 2 18-core CPUs and 384GB RAM,
but it should be no problem to run the scripts on any modern high-end PC.

### Paper

To cite this work, you can use the following bibtex entry:
 ```bib
@report{mclstm,
	author = {Hoedt, Pieter-Jan and Kratzert, Frederik and Klotz, Daniel and Halmich, Christina and Holzleitner, Markus and Nearing, Grey and Hochreiter, Sepp and Klambauer, G{\"u}nter},
	title = {MC-LSTM: Mass-Conserving LSTM},
	institution = {Institute for Machine Learning, Johannes Kepler University, Linz},
	type = {preprint},
	date = {2021},
	url = {http://arxiv.org/abs/2101.05186},
	eprinttype = {arxiv},
	eprint = {2101.05186},
}
```
