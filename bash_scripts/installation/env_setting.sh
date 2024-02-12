proj_dir=$(pwd)

#############################
#      Install packages
#############################
pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
conda install openpyxl


#############################
#      Install MeCab-ko
#############################
cd ${proj_dir}/utils
wget https://bitbucket.org/eunjeon/mecab-ko/downloads/mecab-0.996-ko-0.9.2.tar.gz
tar -zxvf mecab-0.996-ko-0.9.2.tar.gz
rm mecab-0.996-ko-0.9.2.tar.gz
cd ${proj_dir}/utils/mecab-0.996-ko-0.9.2/
path=${proj_dir}/utils
./configure --prefix=${path}/mecab-0.996-ko-0.9.2     # prefix should be an absolute path
make
make check

##### Log in as root #####
su ## with root user
sudo su
sudo make install

##### Check your MeCab-ko #####
mecab --version         # It will show the message like "mecab of 0.996/ko-0.9.2"


#############################
#    Install MeCab-ko dict
#############################
cd ${proj_dir}/utils/mecab-0.996-ko-0.9.2/
path=${proj_dir}/utils
wget https://bitbucket.org/eunjeon/mecab-ko-dic/downloads/mecab-ko-dic-2.1.1-20180720.tar.gz
tar -zxvf mecab-ko-dic-2.1.1-20180720.tar.gz
rm mecab-ko-dic-2.1.1-20180720.tar.gz
cd ${proj_dir}/utils/mecab-0.996-ko-0.9.2/mecab-ko-dic-2.1.1-20180720/
make clean
./autogen.sh

# paths for arguments should be an absolute paths
./configure --prefix=${path}/mecab-0.996-ko-0.9.2/mecab-ko-dic-2.1.1-20180720 --with-dicdir=${path}/mecab-0.996-ko-0.9.2/lib/mecab/dic/mecab-ko-dic --with-mecab-config=${path}/mecab-0.996-ko-0.9.2/bin/mecab-config

make
##### Log in as root #####
su    ## with root user
sudo su
sudo make install   # It will show the error message like "Nothing to be done for 'install-exec-am'.", but it's okay.


##### Test your MeCab-ko #####
path=${proj_dir}/utils
mecab -d ${path}/mecab-0.996-ko-0.9.2/lib/mecab/dic/mecab-ko-dic
설치가 잘 되었습니다. 이제부터 MeCab을 사용할 수 있습니다.
Ctrl + C

##### You should check the directory of mecab-ko-dic where the path is ${path}/mecab-0.996-ko-0.9.2/lib/mecab/dic/mecab-ko-dic
##### If there is no mecab-ko-dic, you should try to install mecab-ko-dic again.


#############################
#    Install mecab-python
#############################
cd ${proj_dir}/utils/mecab-0.996-ko-0.9.2
git clone https://bitbucket.org/eunjeon/mecab-python-0.996.git
cd mecab-python-0.996

python setup.py build
python setup.py install   # Now, you can find 'mecab-python' in pip list and use it in your python code.
