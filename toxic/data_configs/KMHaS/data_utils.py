import os
import json
from nlu_tasks.srcs.preprocess import clean_text


def load_task_dataset(args):
    data_dir = f"datasets/toxic/K_MHaS/"
    if args.do_hangeulize:
        data_path = os.path.join(data_dir, f'processed_data_{args.remain_lang}_hangeulized.json')
    else:
        data_path = os.path.join(data_dir, f'processed_data_{args.remain_lang}.json')
    if args.data_remove:
        data_path = data_path.replace(".json", "_dr.json")

    print(f"\n##### Loading the K_MHaS dataset #####")
    print(f"Data path: {data_path}\n")

    if os.path.exists(data_path):
        total_dataset = json.load(open(data_path, "r"))
    
    else:
        os.makedirs(data_dir, exist_ok=True)
        total_dataset = {'train': dict(),
                         'validation': dict(),
                         'test': dict()
                         }

        for d_type in total_dataset:
            raw_dataset = load_dataset("jeanlee/kmhas_korean_hate_speech", split=d_type)
            dataset = {'sentence': [], 'label': []}
            label = [0] * 9
            for i, indices in enumerate(raw_dataset['label']):
                text = raw_dataset[i]['text']
                text = clean_text(text, args.remain_lang, args.do_hangeulize, args.data_remove)
                if text is None or len(text) == 0:
                    continue
                dataset['sentence'].append(text)
                new_label = label.copy()
                for index in indices:
                    new_label[index] = 1
                dataset['label'].append(new_label)

            total_dataset[d_type] = dataset

        label_map = {'Origin': 0,
                     'Physical': 1,
                     'Politics': 2,
                     'Profanity': 3,
                     'Age': 4,
                     'Gender': 5,
                     'Race': 6,
                     'Religion': 7,
                     'None': 8}

        total_dataset['label_map'] = label_map
        json.dump(total_dataset, open(data_path, "w"))
    return total_dataset
