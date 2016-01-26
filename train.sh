#!/bin/bash

./bin/mglda --option load_corpus --k_global 50 --k_local 10 --gamma 0.1 --alpha_global 0.1 --beta_global 0.1 --alpha_local 0.1 --beta_local 0.1 --alpha_mix_global 0.1 --alpha_mix_local 0.1 --slidding_window_width 3 --vocabulary_size 600000 --max_iterator 100 --corpus_path data/demo_corpus.dat --save_step 10 --save_prefix demo --model_args_file data/demo_args.res --model_vocabulary_file data/demo_vocab.dat 1>tmp.log 2>&1 &

