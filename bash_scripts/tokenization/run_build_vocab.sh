# stroke
python tokenization/scripts/build_vocab.py --tok_name stroke --tok_type ko --vocab_size 200 --input_corpus datasets/wiki/wikiextracted/clean-ko-wiki-20220923.txt --output_dir tokenization/resources --n_jobs 20

# cji
python tokenization/scripts/build_vocab.py --tok_name cji --tok_type ko --vocab_size 200 --input_corpus datasets/wiki/wikiextracted/clean-ko-wiki-20220923.txt --output_dir tokenization/resources --n_jobs 20

# bts
python tokenization/scripts/build_vocab.py --tok_name bts --tok_type ko --vocab_size 200 --input_corpus datasets/wiki/wikiextracted/clean-ko-wiki-20220923.txt --output_dir tokenization/resources --n_jobs 20

# jamo
python tokenization/scripts/build_vocab.py --tok_name jamo --tok_type ko --vocab_size 200 --input_corpus datasets/wiki/wikiextracted/clean-ko-wiki-20220923.txt --output_dir tokenization/resources --n_jobs 20

# char
python tokenization/scripts/build_vocab.py --tok_name char --tok_type ko --vocab_size 2000 --input_corpus datasets/wiki/wikiextracted/clean-ko-wiki-20220923.txt --output_dir tokenization/resources --n_jobs 20

# morpheme
python tokenization/scripts/build_vocab.py --tok_name morpheme --tok_type ko --vocab_size 32000 --input_corpus datasets/wiki/wikiextracted/clean-ko-wiki-20220923.txt --output_dir tokenization/resources --n_jobs 20

# subword
python tokenization/scripts/build_vocab.py --tok_name subword --tok_type ko --vocab_size 32000 --input_corpus datasets/wiki/wikiextracted/clean-ko-wiki-20220923.txt --output_dir tokenization/resources --n_jobs 20

# morphemeSubword
python tokenization/scripts/build_vocab.py --tok_name morphemeSubword --tok_type ko --vocab_size 32000 --input_corpus datasets/wiki/wikiextracted/clean-ko-wiki-20220923.txt --output_dir tokenization/resources --n_jobs 20

# word
python tokenization/scripts/build_vocab.py --tok_name word --tok_type ko --vocab_size 64000 --input_corpus datasets/wiki/wikiextracted/clean-ko-wiki-20220923.txt --output_dir tokenization/resources --n_jobs 20

####### For KOMBO #######
# stroke_var
python tokenization/scripts/build_vocab.py --tok_name stroke_var --tok_type ko --vocab_size 200 --input_corpus datasets/wiki/wikiextracted/clean-ko-wiki-20220923.txt --output_dir tokenization/resources --n_jobs 20

# cji_var
python tokenization/scripts/build_vocab.py --tok_name cji_var --tok_type ko --vocab_size 200 --input_corpus datasets/wiki/wikiextracted/clean-ko-wiki-20220923.txt --output_dir tokenization/resources --n_jobs 20

# bts_var
python tokenization/scripts/build_vocab.py --tok_name bts_var --tok_type ko --vocab_size 200 --input_corpus datasets/wiki/wikiextracted/clean-ko-wiki-20220923.txt --output_dir tokenization/resources --n_jobs 20

# jamo_distinct
python tokenization/scripts/build_vocab.py --tok_name jamo_distinct --tok_type ko --vocab_size 200 --input_corpus datasets/wiki/wikiextracted/clean-ko-wiki-20220923.txt --output_dir tokenization/resources --n_jobs 20
