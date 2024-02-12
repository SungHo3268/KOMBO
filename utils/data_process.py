import re
import os
import sys
sys.path.append(os.getcwd())
import string
from g2pK.g2pk.g2pk import G2p

g2p = G2p()

single_eng_to_hangeul = {'a': '에이',
                         'b': '비',
                         'c': '씨',
                         'd': '디',
                         'e': '이',
                         'f': '에프',
                         'g': '쥐',
                         'h': '에이치',
                         'i': '아이',
                         'j': '제이',
                         'k': '케이',
                         'l': '엘',
                         'm': '엠',
                         'n': '엔',
                         'o': '오',
                         'p': '피',
                         'q': '큐',
                         'r': '알',
                         's': '에스',
                         't': '티',
                         'u': '유',
                         'v': '브이',
                         'w': '더블유',
                         'x': '엑스',
                         'y': '와이',
                         'z': '지',
                         }

def to_hangeul(sentence):
    hangeul_sentence = []
    for word in sentence.strip().split():
        if word.upper() != word.lower():
            hangeul_sentence.append(g2p(word))
        else:
            hangeul_sentence.append(word)
    hangeul_sentence = ' '.join(hangeul_sentence)

    final_eng_check = re.findall("[A-Za-z]+", hangeul_sentence)
    if final_eng_check:
        for word in final_eng_check:
            g2p_word = g2p(word)
            if g2p_word != word:
                hangeul = g2p_word
            else:
                hangeul = ''
                for ch in word.lower():
                    hangeul += single_eng_to_hangeul[ch]
            hangeul_sentence = hangeul_sentence.replace(word, hangeul)
    return hangeul_sentence


def clean_text(sentence, remain_lang="ko_en_punc", do_hangeulize=True, data_remove=False):
    PUNCT = '\\'.join(string.punctuation)
    PUNCT_MAPPING = {"‘": "'", "₹": "e", "´": "'", "°": "", "€": "e", "™": "tm", "√": " sqrt ", "×": "x", "²": "2",
                     "—": "-", "–": "-", "’": "'", '”': '"', '“': '"', "£": "e", '∞': 'infinity',
                     'θ': 'theta', '÷': '/', 'α': 'alpha', '•': '.', 'à': 'a', '−': '-', 'β': 'beta', '∅': '', '³': '3',
                     'π': 'pi',
                     '\u200b': ' ', '…': ' ... ', '\ufeff': '', 'करना': '', 'है': '', '·': '.'}
    remain_lang = remain_lang.split("_")
    remain_re = ""
    for lang in remain_lang:
        if lang == "ko":
            remain_re += "ㄱ-ㅣ가-힣"
        elif lang == "en":
            remain_re += "A-Za-z"
        elif lang == "punc":
            remain_re += PUNCT
    remain_re = "[^" + remain_re + "\s]"
    ONLY_REMAIN = re.compile(rf"{remain_re}")

    for p in PUNCT_MAPPING:
        sentence = sentence.replace(p, PUNCT_MAPPING[p])
    sentence = sentence.strip()

    if data_remove:
        kor_sentence = re.sub(re.compile(rf"[^ㄱ-ㅣ가-힣\s]"), "", sentence)
        kor_sentence_ = kor_sentence.strip()
        if len(kor_sentence_) == 0:
            sentence = None
            return sentence

    re_sentence = re.sub(ONLY_REMAIN, "", sentence)
    re_sentence = re_sentence.strip()

    if data_remove and (len(re_sentence) != len(sentence)):
        sentence = None
        return sentence
    else:
        if do_hangeulize:
            sentence = to_hangeul(sentence)
        return sentence


if __name__ == '__main__':
    text = "This is the world Championship tournament"
    print(f"* Original: {text}")
    print(f"* Hangeulized: {to_hangeul(text)}\n")

    text = "어제 밥 먹은거 really good 이었어!"
    print(f"* Original: {text}")
    print(f"* Hangeulized: {to_hangeul(text)}\n")

    text = "'the shawshank redemption'이다. 언뜻 생각하면 'escape'를 썼을 법한데 'redemption'을 썼다. redemption의 사전적 의미는 구원, 속죄, 회복이다."
    print(f"* Original: {text}")
    print(f"* Hangeulized: {to_hangeul(text)}\n")

    text = 'GDNTOPCLASSINTHECLUB'
    print(f"* Original: {text}")
    print(f"* Hangeulized: {to_hangeul(text)}\n")

    text = "무기에 유치한cg남무"
    print(f"* Original: {text}")
    print(f"* Hangeulized: {to_hangeul(text)}\n")
