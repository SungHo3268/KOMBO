proj_dir=$(pwd)

mkdir ${proj_dir}/datasets/namuwiki
mkdir ${proj_dir}/datasets/namuwiki/raw
mkdir ${proj_dir}/datasets/namuwiki/extracted

cd ${proj_dir}/datasets/namuwiki/raw
gdown https://drive.google.com/uc?id=19ztCnuz_4ZUuLS1xorsODaPN-2wzcMNO
7za e namuwiki_20200302.json.7z
rm namuwiki_20200302.json.7z
