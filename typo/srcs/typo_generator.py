import random
import hgtk
import jamotools
from collections import defaultdict

chosung_list = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
joongsung_list = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅜ', 'ㅢ', 'ㅣ']
jongsung_list = ['', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
keyboard_mappings = None


def insert_jamo_keyboard(word):
    rand_idx = random.randint(1, 3 * len(word)) - 1
    try:
        jamos = list(hgtk.letter.decompose(word[rand_idx // 3]))
        new_jamo = get_keyboard_neighbors(jamos[rand_idx % 3])
        jamos = jamos[:rand_idx % 3] + [new_jamo] + jamos[rand_idx % 3:]
    except hgtk.exception.NotHangulException:
        jamos = word[rand_idx // 3]

    new_char = jamotools.join_jamos("".join(jamos))
    new_word = word[:rand_idx // 3] + new_char + word[rand_idx // 3 + 1:]
    return new_word


def transpose_jamo(word):
    chars = []
    for idx in range(len(word)):
        try:
            jamos = hgtk.letter.decompose(word[idx])
            chars.extend(list(jamos))
        except hgtk.exception.NotHangulException:
            chars.append(word[idx])
    rand_idx = random.randint(1, len(chars)) - 1
    new_words = "".join(chars[:rand_idx] + chars[rand_idx:rand_idx + 2][::-1] + chars[rand_idx + 2:])
    new_words = jamotools.join_jamos(new_words)
    return new_words


def substitute_jamo_keyboard(word):
    rand_idx = random.randint(1, 3 * len(word)) - 1
    try:
        jamos = list(hgtk.letter.decompose(word[rand_idx // 3]))
        jamos[rand_idx % 3] = get_keyboard_neighbors(jamos[rand_idx % 3])

    except hgtk.exception.NotHangulException:
        jamos = word[rand_idx // 3]

    new_char = jamotools.join_jamos("".join(jamos))
    new_word = word[:rand_idx // 3] + new_char + word[rand_idx // 3 + 1:]
    return new_word


def delete_jamo(word):
    rand_idx = random.randint(1, len(word)) - 1
    try:
        jamos = list(hgtk.letter.decompose(word[rand_idx]))
        jamo = random.choice(jamos)
        jamos.remove(jamo)
    except hgtk.exception.NotHangulException:
        jamos = word[rand_idx]

    new_char = jamotools.join_jamos(jamos)
    new_word = word[:rand_idx] + new_char + word[rand_idx + 1:]
    return new_word


def get_keyboard_neighbors(jamo):
    global keyboard_mappings
    if keyboard_mappings is None or len(keyboard_mappings) != 26:
        keyboard_mappings = defaultdict(lambda: [])
        keyboard = ['ㅂㅈㄷㄱㅅㅛㅕㅑㅐㅔ', 'ㅁㄴㅇㄹㅎㅗㅓㅏㅣ*', 'ㅋㅌㅊㅍㅠㅜㅡ***']
        shift_keyboard = 'ㅃㅉㄸㄲㅆ***ㅒㅖ'
        row = len(keyboard)
        col = len(keyboard[0])

        dx = [-1, 1, 0, 0]
        dy = [0, 0, -1, 1]

        for i in range(row):
            for j in range(col):
                for k in range(4):
                    x_, y_ = i + dx[k], j + dy[k]
                    if (0 <= x_ < row) and (0 <= y_ < col):
                        if keyboard[x_][y_] == '*': continue
                        if keyboard[i][j] == '*': continue
                        keyboard_mappings[keyboard[i][j]].append(keyboard[x_][y_])

        for i in range(len(shift_keyboard)):
            keyboard_mappings[shift_keyboard[i]].append(keyboard[0][i])

    if jamo not in keyboard_mappings: return jamo
    return random.choice(keyboard_mappings[jamo])


def generate_typo(sentence, typo_type="random", typo_rate=0.):
    insert_typo = eval(f"insert_jamo_keyboard")
    substitute_typo = eval(f"substitute_jamo_keyboard")
    delete_typo = eval(f"delete_jamo")
    transpose_typo = eval(f"transpose_jamo")

    new_sentence = []
    for word in sentence.split():
        if typo_rate == 0:
            new_sentence.append(word)

        elif random.random() < typo_rate:
            if typo_type == "random":
                what_typo = random.random()
                if what_typo < 0.25:
                    new_sentence.append(insert_typo(word))
                elif what_typo < 0.5:
                    new_sentence.append(transpose_typo(word))
                elif what_typo < 0.75:
                    new_sentence.append(substitute_typo(word))
                else:
                    new_sentence.append(delete_typo(word))
            elif typo_type == "insert":
                new_sentence.append(insert_typo(word))
            elif typo_type == "transpose":
                new_sentence.append(substitute_typo(word))
            elif typo_type == "substitute":
                new_sentence.append(substitute_typo(word))
            elif typo_type == "delete":
                new_sentence.append(substitute_typo(word))
            else:
                raise NotImplementedError
        else:
            new_sentence.append(word)
    return " ".join(new_sentence)


if __name__ == "__main__":
    text = "안녕하세요"
    print(text)
    print(generate_typo(text, "random", 0.9))
