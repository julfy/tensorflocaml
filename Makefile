build: 
	ocamlbuild dostuff.native -use-ocamlfind -pkg tensorflow,devkit,re2 -tag thread
clean:
	rm -r _build dostuff.native
