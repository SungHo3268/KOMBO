from typing import List


ADD_LINE = '-'

subchar_dict = {
    # 초성/종성 공통
    'ㄱ': 'ㄱ',
    'ㄲ': 'ㄱㄱ',
    'ㄴ': 'ㄴ',
    'ㄷ': 'ㄴ'+ADD_LINE,
    'ㄸ': 'ㄴ'+ADD_LINE+'ㄴ'+ADD_LINE,
    'ㄹ': 'ㄹ',
    'ㅁ': 'ㅁ',
    'ㅂ': 'ㅁ'+ADD_LINE,
    'ㅃ': 'ㅁ'+ADD_LINE+'ㅁ'+ADD_LINE,
    'ㅅ': 'ㅅ',
    'ㅆ': 'ㅅㅅ',
    'ㅇ': 'ㅇ',
    'ㅈ': 'ㅅ'+ADD_LINE,
    'ㅉ': 'ㅅ'+ADD_LINE+'ㅅ'+ADD_LINE,
    'ㅊ': 'ㅅ'+ADD_LINE+ADD_LINE,
    'ㅋ': 'ㄱ'+ADD_LINE,
    'ㅌ': 'ㄴ'+ADD_LINE+ADD_LINE,
    'ㅍ': 'ㅁ'+ADD_LINE+ADD_LINE,
    'ㅎ': 'ㅇ'+ADD_LINE,
    # 중성
    'ㅏ': 'ㅣ'+'ㆍ',
    'ㅐ': 'ㅣ'+'ㆍ'+'ㅣ',
    'ㅑ': 'ㅣ'+'ㆍ'+'ㆍ',
    'ㅒ': 'ㅣ'+'ㆍ'+'ㆍ'+'ㅣ',
    'ㅓ': 'ㆍ'+'ㅣ',
    'ㅔ': 'ㆍ'+'ㅣ'+'ㅣ',
    'ㅕ': 'ㆍ'+'ㆍ'+'ㅣ',
    'ㅖ': 'ㆍ'+'ㆍ'+'ㅣ'+'ㅣ',
    'ㅗ': 'ㆍ'+'ㅡ',
    'ㅘ': 'ㆍ'+'ㅡ'+'ㅣ'+'ㆍ',
    'ㅙ': 'ㆍ'+'ㅡ'+'ㅣ'+'ㆍ'+'ㅣ',
    'ㅚ': 'ㆍ'+'ㅡ'+'ㅣ',
    'ㅛ': 'ㆍ'+'ㆍ'+'ㅡ',
    'ㅜ': 'ㅡ'+'ㆍ',
    'ㅝ': 'ㅡ'+'ㆍ'+'ㆍ'+'ㅣ',
    'ㅞ': 'ㅡ'+'ㆍ'+'ㆍ'+'ㅣ'+'ㅣ',
    'ㅟ': 'ㅡ'+'ㆍ'+'ㅣ',
    'ㅠ': 'ㅡ'+'ㆍ'+'ㆍ',
    'ㅡ': 'ㅡ',
    'ㅢ': 'ㅡ'+'ㅣ',
    'ㅣ': 'ㅣ',
    # 종성 단독
    '': '',
    'ㄳ': 'ㄱㅅ',
    'ㄵ': 'ㄴㅅ'+ADD_LINE,
    'ㄶ': 'ㄴㅇ'+ADD_LINE,
    'ㄺ': 'ㄹㄱ',
    'ㄻ': 'ㄹㅁ',
    'ㄼ': 'ㄹㅁ'+ADD_LINE,
    'ㄽ': 'ㄹㅅ',
    'ㄾ': 'ㄹㄴ'+ADD_LINE+ADD_LINE,
    'ㄿ': 'ㄹㅁ'+ADD_LINE+ADD_LINE,
    'ㅀ': 'ㄹㅇ'+ADD_LINE,
    'ㅄ': 'ㅁ'+ADD_LINE+'ㅅ',
}

subchar_reverse_dict = {val: key for key, val in subchar_dict.items()}


# basic_consonant = ['ㄱ', 'ㄴ', 'ㅁ', 'ㅅ', 'ㅇ'] + ['ㄹ'] + [ADD_LINE]
# basic_vowel = ['ㅡ', 'ㆍ', 'ㅣ']
# basic_cv = [basic_consonant,
#             basic_vowel]

consonants = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ', 'ㄳ', 'ㄵ', 'ㄶ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅄ', ''] \
             + ['\u11A8', '\u11A9', '\u11AB', '\u11AE', '\u11AF', '\u11B7', '\u11B8', '\u11BA', '\u11BB', '\u11BC', '\u11BD', '\u11BE', '\u11BF', '\u11C0', '\u11C1', '\u11C2'] \
             + [ADD_LINE]
vowels = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ'] \
         + ['ㆍ']
cv = [consonants, vowels]

def jamo_seperator(char: str) -> List[str]:
    """
    :param char: the string split into BTS units (stroke, cji, and bts)
    :return: [chosung, joongsung, jongsung]
    """

    seperated_jamo = []
    cur_jamo = ''
    mode = 0        # if mode is 0, it means the consonant,
                    # if mode is 1, it means the vowel,
    if (len(char) > 0) and (char[0] not in cv[0] + cv[1]):
        cur_jamo = char
    else:
        for c in char:
            if c in cv[mode]:
                cur_jamo += c
            elif c in cv[(mode+1) % 2]:
                seperated_jamo.append(cur_jamo)
                cur_jamo = c
                mode = (mode+1) % 2
            else:
                seperated_jamo.append(c)
    seperated_jamo.append(cur_jamo)
    return seperated_jamo


def jamo_compatible_mapping(jamo):
    jamo_map = {'\u11A8': '\u1100',     # ㄱ
                '\u11A9': '\u1101',     # ㄲ
                '\u11AB': '\u1102',     # ㄴ
                '\u11AE': '\u1103',     # ㄷ
                '\u11AF': '\u1105',     # ㄹ
                '\u11B7': '\u1106',     # ㅁ
                '\u11B8': '\u1107',     # ㅂ
                '\u11BA': '\u1109',     # ㅅ
                '\u11BB': '\u110A',     # ㅆ
                '\u11BC': '\u110B',     # ㅇ
                '\u11BD': '\u110C',     # ㅈ
                '\u11BE': '\u110E',     # ㅊ
                '\u11BF': '\u110F',     # ㅋ
                '\u11C0': '\u1110',     # ㅌ
                '\u11C1': '\u1111',     # ㅍ
                '\u11C2': '\u1112',     # ㅎ
                }
    if jamo in jamo_map:
        modified = jamo_map[jamo]
    else:
        modified = jamo
    return modified

