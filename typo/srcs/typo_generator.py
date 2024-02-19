import random
import hgtk
import jamotools
from collections import defaultdict

chosung_list = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
joongsung_list = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅜ', 'ㅢ', 'ㅣ']
jongsung_list = ['', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
keyboard_mappings = None


def add_character(word):
    cho, joong, jong = random.choice(chosung_list), random.choice(joongsung_list), random.choice(jongsung_list)
    new_char = jamotools.join_jamos(cho + joong + jong)
    rand_idx = random.randint(1, len(word)) - 1
    new_word = word[:rand_idx] + new_char + word[rand_idx:]
    return new_word


def add_jamo(word):
    new_jamo = random.choice(chosung_list + joongsung_list + jongsung_list)
    rand_idx = random.randint(1, len(word)) - 1
    new_word = word[:rand_idx] + new_jamo + word[rand_idx:]
    return new_word


def add_keyboard(word):
    rand_idx = random.randint(1, 3 * len(word)) - 1
    try:
        jamos = list(hgtk.letter.decompose(word[rand_idx // 3]))
        new_jamo = get_keyboard_neighbors(jamos[rand_idx % 3])
        jamos = jamos[:rand_idx % 3] + [new_jamo] + jamos[rand_idx % 3:]

    except:
        jamos = word[rand_idx // 3]

    new_char = jamotools.join_jamos("".join(jamos))
    new_word = word[:rand_idx // 3] + new_char + word[rand_idx // 3 + 1:]
    return new_word


def replace_character(word):
    cho, joong, jong = random.choice(chosung_list), random.choice(joongsung_list), random.choice(jongsung_list)
    new_char = jamotools.join_jamos(cho + joong + jong)
    rand_idx = random.randint(1, len(word)) - 1
    new_word = word[: rand_idx] + new_char + word[rand_idx+1:]
    return new_word


def replace_jamo(word):
    new_jamo = random.choice(chosung_list + joongsung_list + jongsung_list)
    rand_idx = random.randint(1, len(word)) - 1
    try:
        cho, joong, jong = hgtk.letter.decompose(word[rand_idx])
        if new_jamo in chosung_list:
            cho = new_jamo
        elif new_jamo in joongsung_list:
            joong = new_jamo
        elif new_jamo in jongsung_list:
            jong = new_jamo
        
        new_char = jamotools.join_jamos(cho + joong + jong)
        new_word = word[: rand_idx] + new_char + word[rand_idx+1:]
        return new_word
    except:
        return word


def replace_keyboard(word):
    rand_idx = random.randint(1, 3 * len(word)) - 1
    try:
        jamos = list(hgtk.letter.decompose(word[rand_idx // 3]))
        jamos[rand_idx % 3] = get_keyboard_neighbors(jamos[rand_idx % 3])

    except:
        jamos = word[rand_idx // 3]

    new_char = jamotools.join_jamos("".join(jamos))
    new_word = word[:rand_idx // 3] + new_char + word[rand_idx // 3 + 1:]
    return new_word


def drop_character(word):
    rand_idx = random.randint(1, len(word)) - 1
    new_word = word[:rand_idx] + word[rand_idx + 1:]
    return new_word


def drop_jamo(word):
    rand_idx = random.randint(1, len(word)) - 1
    try:
        jamos = list(hgtk.letter.decompose(word[rand_idx]))
        jamo = random.choice(jamos)
        jamos.remove(jamo)
    except:
        jamos = word[rand_idx]

    new_char = jamotools.join_jamos(jamos)
    new_word = word[:rand_idx] + new_char + word[rand_idx + 1:]
    return new_word


def swap_jamo(word):
    chars = []
    for idx in range(len(word)):
        try:
            jamos = hgtk.letter.decompose(word[idx])
            chars.extend(list(jamos))
        except:
            chars.append(word[idx])
    rand_idx = random.randint(1, len(chars)) - 1
    new_words = "".join(chars[:rand_idx] + chars[rand_idx:rand_idx + 2][::-1] + chars[rand_idx + 2:])
    new_words = jamotools.join_jamos(new_words)
    return new_words


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


def generate_typo(sent, typo_type="all", typo_level="jamo", typo_rate=0):
    # add_typo = eval(f"add_{typo_level}")
    add_typo = eval(f"add_keyboard")
    replace_typo = eval(f"replace_keyboard")
    drop_typo = eval(f"drop_{typo_level}")
    swap_typo = eval(f"swap_{typo_level}")

    new_sent = []
    for word in sent.split():
        if typo_rate == 0:
            new_sent.append(word)

        elif random.random() < typo_rate:
            if typo_type == "all":
                what_typo = random.random()
                if what_typo < 0.25:
                    new_sent.append(add_typo(word))
                elif what_typo < 0.5:
                    new_sent.append(replace_typo(word))
                elif what_typo < 0.75:
                    new_sent.append(swap_typo(word))
                else:
                    new_sent.append(drop_typo(word))
            elif typo_type == "add":
                new_sent.append(add_typo(word))
            elif typo_type == "replace":
                new_sent.append(replace_typo(word))
            else:
                select_typo = eval(f"{typo_type}_{typo_level}")
                new_sent.append(select_typo(typo_level))

        else:
            new_sent.append(word)

    return " ".join(new_sent)


if __name__ == "__main__":
    text = "안녕하세요"
    print(text)
    print(generate_typo(text, "all", "jamo", 0.9))
