import os
import random
import numpy as np
import pandas as pd
from tqdm.auto import tqdm


random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)


save_dir = "datasets/aihub/"
os.makedirs(save_dir, exist_ok=True)


korean = []
english = []
for i in tqdm(range(1, 5), desc="Reading files...", bar_format="{l_bar}{bar:15}{r_bar}"):
    raw_data = pd.read_excel(os.path.join(save_dir, f"raw/aihub_news_corpus_{i}.xlsx"))
    df_data = pd.DataFrame(raw_data, columns=['원문', '번역문'])
    ko = list(df_data['원문'])
    en = list(df_data['번역문'])
    korean += ko
    english += en

assert len(korean) == len(english)
data_size = len(korean)
print(f"Total {data_size} sentences.")
per = np.random.permutation(np.arange(data_size))
train_idx = per[:-16000]
dev_idx = per[-16000: -8000]
test_idx = per[-8000:]


with open(os.path.join(save_dir, "aihub_news_ko_train.txt"), 'w', encoding='utf-8') as ftr:
    with open(os.path.join(save_dir, "aihub_news_ko_dev.txt"), 'w', encoding='utf-8') as fde:
        with open(os.path.join(save_dir, "aihub_news_ko_test.txt"), 'w', encoding='utf-8') as fte:
            for i in tqdm(range(data_size), desc="Writing Korean files...", bar_format="{l_bar}{bar:15}{r_bar}"):
                if i in train_idx:
                    ftr.write(f"{korean[i]}\n")
                elif i in dev_idx:
                    fde.write(f"{korean[i]}\n")
                elif i in test_idx:
                    fte.write(f"{korean[i]}\n")

with open(os.path.join(save_dir, "aihub_news_en_train.txt"), 'w', encoding='utf-8') as ftr:
    with open(os.path.join(save_dir, "aihub_news_en_dev.txt"), 'w', encoding='utf-8') as fde:
        with open(os.path.join(save_dir, "aihub_news_en_test.txt"), 'w', encoding='utf-8') as fte:
            for i in tqdm(range(data_size), desc="Writing English files...", bar_format="{l_bar}{bar:15}{r_bar}"):
                if i in train_idx:
                    ftr.write(f"{english[i]}\n")
                elif i in dev_idx:
                    fde.write(f"{english[i]}\n")
                elif i in test_idx:
                    fte.write(f"{english[i]}\n")
