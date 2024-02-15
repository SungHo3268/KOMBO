# KOMBO: Korean Character-level Architecture Based on the Combination Rules of Sub-characters


<!-- TABLE OF CONTENTS -->
<h2>Contents</h2>
<ul>
  <li>
    <a href="#installation">Installation</a>
  </li>
  <li>
    <a href="#experimental-settings">Experimental Settings</a>
    <ul>
      <li><a href="#corpus">Corpus</a></li>
      <li><a href="#tokenization-baselines">Tokenization Baselines</a></li>
      <li><a href="#building-vocabulary">Building vocabulary</a></li>
    </ul>
  </li>
  <li>
    <a href="#Pretraining">Pretraining</a>
    <ul>
      <li><a href="#pretraining_data">Dataset</a></li>
      <li><a href="#pretraining_settings">Settings</a></li>
      <li><a href="#pretrained_models">Baselines</a></li>
    </ul>
  </li>
  <li>
    <a href="#standard-korean-datasets">Standard Korean Datasets</a>
    <ul>
      <li><a href="#dataset">Dataset</a></li>
      <li><a href="#fine_tuning">Fine-tuning</a></li>
    </ul>
  </li>
  <li>
    <a href="#noisy-korean-datasets">Noisy Korean Datasets</a>
    <ul>
      <li><a href="#task1">Typo Data</a></li>
        <ul>
          <li><a href="#dataset1">Dataset</a></li>
          <li><a href="#fine_tuning1">Fine-tuning</a></li>
        </ul>
      <li><a href="#task2">Toxic Data</a></li>
        <ul>
          <li><a href="#dataset2">Dataset</a></li>
          <li><a href="#fine_tuning2">Fine-tuning</a></li>
        </ul>
    </ul>
  </li>
</ul>
<br/>
<br/>

## Installation
- Create your virtual environment.
  ```bash
  conda create -n {your_env} python=3.8
  conda activate {your_env}
  
  pip install --upgrade pip
  ```
- Set the experiment environments.
  - Install the libraries including MeCab. (When you install the MeCab, you might have to join with a root account.)
  ```bash
  bash bash_scripts/installation/env_setting.sh
  ```
<br/>
<br/>



## Experimental Settings
<a id="corpus"></a>
### Corpus

* Dataset preprocessing

  | Corpus                   | Remove HTML tags  | Remove Empty space  | Punctuation Mapping  | ONLY_REMAIN (Korean/English/Punct)          |
  | :----------------------: | :---------------: | :-----------------: | :------------------: | :----------------------------------------:  |
  | Korean Wiki              | O                 | O                   | O                    | O <br/>(for building vocabulary)            |
  | English Wiki             | O                 | O                   | X                    | X                                           |
  | Namu Wiki                | O                 | O                   | O                    | O <br/>(same as Korean Wiki)                |


* The corpus used for building vocabulary and training BPE models is as follows, which was extracted and refined via [attardi/wikiextractor](https://github.com/attardi/wikiextractor).<br/>
  > Notice)<br/>
  > To reproduce the baseline, we use the 0.1 version of the wikiextractor.<br/>
  > We use the latest version of the wiki dump dataset for both Korean and English. We use the data dated 20220923. <br/>
  > You can download the dump data [here](remove_this/https://drive.google.com/drive/folders/1sIrBNRO66xboZPOmpyF5Cn9gQj8uyL6W?usp=share_link).
  
  - Korean Wikipedia: <https://dumps.wikimedia.org/kowiki>
  - English Wikipedia: <https://dumps.wikimedia.org/enwiki>
  <br/>
  
  
  ```bash
  python -m wikiextractor.WikiExtractor {Wiki dump file path} -o {output dir}
  ```
  ```bash
  cat {otuput_dir}/*/* > {final output_file}.txt
  ```
  
* After applying the wikiextractor to wiki dump data, we appy the [clean_str.py](tokenization/scripts/clean_str.py) for preprocessing the corpus.<br/>
  It returns the **clean-{output_file}.txt**
  ```bash
  python tokenization/bash_scripts/clean_str.py \
  --lang ko --corpus_type wiki \
  --input_corpus datasets/wiki/wikiextracted/ko-wiki-20220923.txt
  ```
  
* You can also download and preprocess all datasets at once.
  ```bash
  bash install_pipeline.sh
  ```
  If you want to do each steps separately, then you just follow the below steps.<br/>
  - Wiki dataset
    ```bash
    bash bash_scripts/installation/download_wiki.sh

    bash bash_scripts/installation/clean_wiki.sh
    ```
  - AIHub news dataset
    ```bash
    bash bash_scripts/installation/download_aihub.sh

    bash bash_scripts/installation/clean_aihub.sh
    ```
  - Namu Wiki dataset
    ```bash
    bash bash_scripts/installation/download_namu.sh

    bash bash_scripts/installation/clean_namu.sh
    ```
  - NLU task datasets
    ```bash
    bash bash_scripts/installation/download_tasks.sh
    ```

<a id="tokenization-baselines"></a>
### Tokenization Baselines 

* There are 13 tokenization strategies for Korean. See [here](tokenization/) to prepare and use each strategy.
  
  * **Stroke** - 130
  * **Cji** - 136
  * **BTS** - 112
  * **Jamo(Consonant and Vowel)** - 170
  * **Character(Syllable)** - 2k
  * **Morpheme** - 32k - using MeCab-ko
  * **Subword** - 32k
  * **Morpheme-aware Subword** - 32k
  * **Word** - 64k - using Moses
  * **KOMBO(Stroke)** - 130
  * **KOMBO(Cji)** - 136
  * **KOMBO(BTS)** - 112
  * **KOMBO(Jamo)** - 170


<a id="building-vocabulary"></a>
### Building vocabulary

- This return the <code>tok.vocab</code> and <code>fairseq.vocab</code> (and <code>tok.model</code> @subword, morpphemeSubword) <br/>
  ```bash
  python tokenization/bash_scripts/build_vocab.py \
  --tok_type en --tok_name subword --vocab_size 32000 \
  --config_path tokenization/utils/tok_info.cfg \
  --input_corpus datasets/wiki/wikiextracted/clean-en-wiki-20220923.txt \
  --output_dir tokenization/resources/ \
  --n_jobs 20  
  ```
  You can also build all vocabularies with [installation code](bash_scripts/tokenization/run_build_vocab.sh).
  ```bash
  bash bash_scripts/tokenization/run_build_vocab.sh
  ```

<br/>
<br/>

<a id="Pretraining"></a>
## Pre-training

For  each  tokenization  strategy,  pre-training of BERT-Base model (Devlin et al., 2019)  was  performed with a Huggingface and Pytorch library.
- Trim original dataset (remove doc seperator, space) and Make the corpus and labels for NSP task for each tasks. It returns the `sentence_as.txt`, `sentence_bs.txt`, and `nsp_labels.txt`. <br/>
We preprocessed the input segments by referring the [official BERT code](https://github.com/google-research/bert/blob/master/create_pretraining_data.py).<br/>
    
    
- We set the training hyper-parameters of all models as follows:<br/>
`batch_size=128`, `max_sequence_length=128`, `learning_rate=5e-5`, `total_steps=1_000_000` `warm_up_steps=10_000`
    
- Run Training
  ```bash
  python nlu_tasks/bash_scripts/run_pretraining.py --random_seed 42 --multi_gpu False \
  --tok_type ${TOKENIZER} --lang ${LANGUAGE} --tok_vocab_size ${TOK_VOCAB} --bpe_corpus ${BPE_CORPUS} \
  --bert_config_path nlu_tasks/utils/bert_config.json --max_seq_len 128 --reload False \
  --batch_size 128 --gradient_accumulation_steps 1 --total_steps 1_000_000 --num_warmup_steps 10_000 \
  --learninig_rate 5e-05 --max_grad_norm 1.
  
  
  # TOKENIZER = {jamo, char, morpheme, subword, morphemeSubword, word, stroke, cji, bts}
  # LANGUAGE = {ko, en}
  # TOK_VOCAB = {200, 2k, 4k, 8k, 16k, 32k, 64k}
  # BPE_CORPUS = {wiki, aihub}
  ```


  If you want to resume the pre-training, you should set the reload and ckpt_dir argument additionaly.
  ```bash
  python nlu_tasks/bash_scripts/run_pretraining.py --tok_type ${TOKENIZER} --tok_vocab_size ${TOK_VOCAB} \
  --reload True --ckpt_dir ${CKPT_DIR}
  
  
  # TOKENIZER = {jamo, char, morpheme, subword, morphemeSubword, word, stroke, cji, bts}
  # LANGUAGE = {ko, en}
  # TOK_VOCAB = {200, 2k, 4k, 8k, 16k, 32k, 64k}
  # BPE_CORPUS = {wiki, aihub}
  # CKPT_DIR = {checkpoint-300000, ckpt}
  ```

<a id="pretraining_data"></a>
### Dataset

- Because the Korean Wiki corpus(20220923) (753 MB) is not enough in volume for the pre-training purpose, we additionally downloaded the recent dump of [Namuwiki corpus(20190312) (5.5 GB)](https://namu.wiki/w/%EB%82%98%EB%AC%B4%EC%9C%84%ED%82%A4:%EB%8D%B0%EC%9D%B4%ED%84%B0%EB%B2%A0%EC%9D%B4%EC%8A%A4%20%EB%8D%A4%ED%94%84) and extracted plain texts using [Namu Wiki Extractor](https://github.com/jonghwanhyeon/namu-wiki-extractor/tree/4d864d2f7da1d4cb767c22d82f91fe2912007e4b) with adding document seperation(<code>'\n'</code>) per article.
  ```bash
  python utils/namuwiki_extract.py --input_corpus datasets/namuwiki/raw/namuwiki_20200302.json --output_dir datasets/namuwiki/extracted
  ```

- Then, we [preprocess](nlu_tasks/preprocess/clean_corpus.py) the corpus as we mentioned above. It returns the file name pattern like "clean-doc-*.txt"<br/>
  ```bash
  python nlu_tasks/preprocess/clean_corpus.py --input_corpus {CORPUS}


  # CORPUS = { kowiki, namuwiki }
  ```
- Concatenate the Korean Wiki corpus and the Namu Wiki corpus.
  ```bash
  mkdir datasets/nlu_tasks/pretraining
  
  cat datasets/namuwiki/extracted/clean-doc-namuwiki_20200302.txt datasets/wiki/wikiextracted/clean-doc-ko-wiki-20220923.txt > datasets/nlu_tasks/pretraining/concatenated.txt
  ```


<a id="pretraining_settings"></a>
### Settings
abcde

<a id="pretrained_models"></a>
### Baselines
abcde

<br/>
<br/>



## Standard Korean Datasets

| Tokenization           | Vocab Size                                                                                                  | KorQuAD               | KorNLI    | KorSTS    | NSMC      | PAWS-X    |
| ---------------------- | ----------------------------------------------------------------------------------------------------------- | --------------------- | --------- | --------- | --------- | --------- |
| Stroke                 | [130](https://www.dropbox.com/s/0ovofnol5j1g5ha/Dawoon%20Jung%20-%20word-64k-pretrained-bert.tar?dl=0)      | 1.54 / 8.86           | 64.06     | 65.83     | 69        | 60.41     | 70.1      | 70.58     | 58.25     | 55.30     |
| Cji                    | [136](https://www.dropbox.com/s/0ovofnol5j1g5ha/Dawoon%20Jung%20-%20word-64k-pretrained-bert.tar?dl=0)      | 1.54 / 8.86           | 64.06     | 65.83     | 69        | 60.41     | 70.1      | 70.58     | 58.25     | 55.30     |
| BTS                    | [112](https://www.dropbox.com/s/0ovofnol5j1g5ha/Dawoon%20Jung%20-%20word-64k-pretrained-bert.tar?dl=0)      | 1.54 / 8.86           | 64.06     | 65.83     | 69        | 60.41     | 70.1      | 70.58     | 58.25     | 55.30     |
| Jamo                   | [170](https://www.dropbox.com/s/m840vmgwis9glzq/Dawoon%20Jung%20-%20jamo-200-pretrained-bert.tar?dl=0)      | 59.66 / 73.91         | 70.6      | 71.2      | 77.22     | 71.47     | 87.97     | 87.89     | 58        | 55.20     |
| Character              | [2K](https://www.dropbox.com/s/560c8lehfijn0mr/Dawoon%20Jung%20-%20char-2k-pretrained-bert.tar?dl=0)        | 69.10 / 83.29         | 73.98     | 73.47     | 82.7      | 75.86     | 88.94     | 89.07     | 68.65     | 67.20     |
| Morpheme               | [32K](https://www.dropbox.com/s/215tb4ublea3orp/Dawoon%20Jung%20-%20mecab-32k-pretrained-bert.tar?dl=0)     | 68.05 / 83.82         | 74.86     | 74.37     | 82.37     | 76.83     | 87.87     | 88.04     | 69.3      | 67.20     |
| Subword                | [32K](https://www.dropbox.com/s/6n1dp2dhjneb5hd/Dawoon%20Jung%20-%20bpe-32k-pretrained-bert.tar?dl=0)       | **74.04** / 86.30     | *74.74*   | 74.29     | 83.02     | 77.01     | *89.39*   | *89.38*   | 74.05     | 70.95     |
| Morpheme-aware Subword | [32K](https://www.dropbox.com/s/mczbb3kf7fzt9l3/Dawoon%20Jung%20-%20mecab_bpe-32k-pretrained-bert.tar?dl=0) | *72.65* / *86.35*     | 74.1      | 75.13     | 83.65     | **78.11** | 89.53     | 89.65     | 74.6      | 71.60     |
| Word                   | [64K](https://www.dropbox.com/s/0ovofnol5j1g5ha/Dawoon%20Jung%20-%20word-64k-pretrained-bert.tar?dl=0)      | 1.54 / 8.86           | 64.06     | 65.83     | 69        | 60.41     | 70.1      | 70.58     | 58.25     | 55.30     |
| KOMBO(Stroke)          | [130](https://www.dropbox.com/s/0ovofnol5j1g5ha/Dawoon%20Jung%20-%20word-64k-pretrained-bert.tar?dl=0)      | 1.54 / 8.86           | 64.06     | 65.83     | 69        | 60.41     | 70.1      | 70.58     | 58.25     | 55.30     |
| KOMBO(Cji)             | [136](https://www.dropbox.com/s/0ovofnol5j1g5ha/Dawoon%20Jung%20-%20word-64k-pretrained-bert.tar?dl=0)      | 1.54 / 8.86           | 64.06     | 65.83     | 69        | 60.41     | 70.1      | 70.58     | 58.25     | 55.30     |
| KOMBO(BTS)             | [112](https://www.dropbox.com/s/0ovofnol5j1g5ha/Dawoon%20Jung%20-%20word-64k-pretrained-bert.tar?dl=0)      | 1.54 / 8.86           | 64.06     | 65.83     | 69        | 60.41     | 70.1      | 70.58     | 58.25     | 55.30     |
| KOMBO(Jamo)            | [170](https://www.dropbox.com/s/m840vmgwis9glzq/Dawoon%20Jung%20-%20jamo-200-pretrained-bert.tar?dl=0)      | 59.66 / 73.91         | 70.6      | 71.2      | 77.22     | 71.47     | 87.97     | 87.89     | 58        | 55.20     |
  

<a id="dataset"></a>
### Dataset
abcde

<a id="fine_tuning"></a>
### Fine-tuning

For each tokenization strategy, fine-tuning of 5 Korean NLU tasks, KorQuAD, KorNLI, KorSTS, NSMC, and PAWS_X, was performed.<br/>
- All tasks shared the files corresponding to [bert config](nlu_tasks/utils/bert_config.json), [models](nlu_tasks/srcs/models.py), [trainer](nlu_tasks/srcs/task_trainer.py), and [running code](nlu_tasks/scripts/run_task.py) and used the individual [config](nlu_tasks/tasks) and [data preprocessing code](nlu_tasks/tasks) files for each tasks. <br/>
  You can run the fine-tuning of tokenization models for each tasks you want as follows:

  ```bash
  python nlu_tasks/bash_scripts/run_task.py --random_seed 42 \
  --tok_type ${TOKENIZER} --lang ${LANGUAGE} --tok_vocab_size ${TOK_VOCAB} --bpe_corpus ${BPE_CORPUS} \
  --ckpt_dir ${CKPT_DIR} --ckpt_file_name {CKPT_FILE} \
  --task_name ${TASKS}


  # TOKENIZER = {jamo, char, morpheme, subword, morphemeSubword, word, stroke, cji, bts}
  # LANGUAGE = {ko, en}
  # TOK_VOCAB = {200, 2k, 4k, 8k, 16k, 32k, 64k}
  # BPE_CORPUS = {wiki, aihub}

  # CKPT_DIR = {Pre-training: ckpt  |  checkpoint-300000
  #             Fine-tuning: 128b_1s_42rs/ckpt  |  128b_1s_42rs/checkpoint-300000}
  # CKPT_FILE = {pytorch_model.bin  |  final_steps.ckpt}

  # TASKS = {KorQuAD, KorNLI, KorSTS, NSMC, PAWS_X}
  ```

- If you want to do all 5 Korean NLU tasks, you can do it by runnig the following.
  ```bash
  bash bash_scripts/nlu_tasks/all_tasks.sh ${CUDA} ${TOKENIZER} ${TOK_VOCAB} ${CKPT_DIR} ${CKPT_FILE}

  # options are the same as above.
  ```
  
<br/>
<br/>



## Noisy Korean Datasets

  <a id="task1"></a>
  ### Typo Data
  abcdef
    <a id="dataset1"></a>
    #### Dataset
    abcdef
    <a id="fine_tuning1"></a>
    #### Fine-tuning
    abcdef

  <a id="task2"></a>
  ### Toxic Data
  abcdef
    <a id="dataset2"></a>
    #### Dataset
    abcdef
    <a id="fine_tuning2"></a>
    #### Fine-tuning
    abcdef

<br/>
<br/>



## Citation

```plain
.
```

## Acknowledgements

.
