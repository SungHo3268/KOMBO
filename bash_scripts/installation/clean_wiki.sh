proj_dir=$(pwd)
cd ${proj_dir}

# Clean Korean Wiki
python tokenization/scripts/clean_str.py \
--lang ko --corpus_type wiki \
--input_corpus datasets/wiki/wikiextracted/ko-wiki-20220923.txt
