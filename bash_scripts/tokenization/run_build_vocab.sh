# jamo
python tokenization/scripts/build_vocab.py --tok_name jamo --tok_type ko --vocab_size 200 --input_corpus datasets/wiki/wikiextracted/clean-ko-wiki-20220923.txt --output_dir tokenization/resources --n_jobs 20

# char
python tokenization/scripts/build_vocab.py --tok_name char --tok_type ko --vocab_size 2000 --input_corpus datasets/wiki/wikiextracted/clean-ko-wiki-20220923.txt --output_dir tokenization/resources --n_jobs 20

# morpheme
for i in 8000 16000 32000 64000
do
	python tokenization/scripts/build_vocab.py --tok_name morpheme --tok_type ko --vocab_size $i --input_corpus datasets/wiki/wikiextracted/clean-ko-wiki-20220923.txt --output_dir tokenization/resources --n_jobs 20
done

# subword
for i in 4000 8000 16000 32000 64000
do
	python tokenization/scripts/build_vocab.py --tok_name subword --tok_type ko --vocab_size $i --input_corpus datasets/wiki/wikiextracted/clean-ko-wiki-20220923.txt --output_dir tokenization/resources --n_jobs 20
done

# morphemeSubword
for i in 4000 8000 16000 32000 64000
do
        python tokenization/scripts/build_vocab.py --tok_name morphemeSubword --tok_type ko --vocab_size $i --input_corpus datasets/wiki/wikiextracted/clean-ko-wiki-20220923.txt --output_dir tokenization/resources --n_jobs 20
done

# word
python tokenization/scripts/build_vocab.py --tok_name word --tok_type ko --vocab_size 64000 --input_corpus datasets/wiki/wikiextracted/clean-ko-wiki-20220923.txt --output_dir tokenization/resources --n_jobs 20

# English subword
python tokenization/scripts/build_vocab.py --tok_name subword --tok_type en --vocab_size 32000 --input_corpus datasets/wiki/wikiextracted/clean-en-wiki-20220923.txt --output_dir tokenization/resources --n_jobs 20
