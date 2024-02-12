proj_dir=$(pwd)
cd ${proj_dir}

python utils/namuwiki_extract.py --input_corpus datasets/namuwiki/raw/namuwiki_20200302.json --output_dir datasets/namuwiki/extracted
python nlu_tasks/preprocess/clean_corpus.py --input_corpus kowiki
python nlu_tasks/preprocess/clean_corpus.py --input_corpus namuwiki

mkdir datasets/nlu_tasks
mkdir datasets/nlu_tasks/pretraining
cat datasets/namuwiki/extracted/clean-doc-namuwiki_20200302.txt datasets/wiki/wikiextracted/clean-doc-ko-wiki-20220923.txt > datasets/nlu_tasks/pretraining/concatenated.txt
