proj_dir=$(pwd)
output_dir=${proj_dir}/datasets/wiki/wikiextracted

pip install gdown
pip install wikiextractor==0.1

mkdir ${proj_dir}/datasets
mkdir ${proj_dir}/datasets/wiki
mkdir ${proj_dir}/datasets/wiki/raw
mkdir ${proj_dir}/datasets/wiki/wikiextracted

# Korean Wiki
cd ${proj_dir}/datasets/wiki/raw
gdown https://drive.google.com/uc?id=1RmpKbimhGkUP-G9GydFtQqQiJOAv-Ah2
mkdir ${output_dir}/ko
python -m wikiextractor.WikiExtractor kowiki-latest-pages-articles-20220923.xml.bz2 -o ${output_dir}/ko -q
cat ${output_dir}/ko/*/* > ${output_dir}/ko-wiki-20220923.txt
rm -rf ${output_dir}/ko
