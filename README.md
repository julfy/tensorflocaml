General idea: 
  1. Develop and train model using Keras.
  2. Use model in OCaml.
It makes more sense for preprocessor to be where the model will be used, on OCaml side that is.

[tensorflow-ocaml](https://github.com/LaurentMazare/tensorflow-ocaml) bindings are used to load (in tensorflow .pb format) and run model in OCaml. 
Even though it is not well developed it prooved to be enough and can be used to simply run a ready model.

Pre-trained model (models/final_.pb) is provided for demonstration purpose. Trained on Imdb movie review [dataset](http://ai.stanford.edu/~amaas/data/sentiment/), even though very simple, provides accuracy of 86%.
Note that you will have to download and unpack the original dataset to '/data' if you want to play with preprocessor.

`dostuff.ml` contains data preprocessor and tensorflow graph usage demo in `predict : string -> unit` function
`main.py` should be run using python3.
  Really important functions are `convert (inp,out)` and `save_model_as_graph(model,dir,name)` which show a pretty simple way to export keras model as a `.pb` file.
  Note that input and output node names are required to correctly setup the model on OCaml side.

How to:
```bash
opam install tensorflow devkit re2
make build
./dostuff.native -predict inp_good.txt
```
