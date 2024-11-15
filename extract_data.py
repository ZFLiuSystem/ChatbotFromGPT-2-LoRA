import pandas as pd


def remove_quotes(s: str):
    if s.startswith(("'", '"')) and s.endswith(("'", '"')):
        s = s[1:-2]
    return s


def main(file_path=None, save_path=None):
    raw_set = pd.read_csv(filepath_or_buffer=file_path, header=None)
    dialogue_set = raw_set[0]
    dialogue_list = []
    for a_dialog in dialogue_set:
        sentences = a_dialog.split('\n')
        length = len(sentences)
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip('[').strip(']').strip(" ")
            sentence = remove_quotes(sentence)
            if " ' " in sentence:
                sub_sentences = sentence.split(" ' ")
                marker = 0
                for ids, sub_sens in enumerate(sub_sentences):
                    marker += 1
                    if sub_sens.startswith(("' ", '" ')):
                        sub_sens = sub_sens[2:]
                    elif sub_sens.endswith((" '", ' "')):
                        sub_sens = sub_sens[:-2]
                    sub_sentences[ids] = sub_sens
                sub_sentences = [sub_sens.strip(" ") + "\n" for sub_sens in sub_sentences]
                if i == length - 1:
                    sub_sentences[marker - 1] += '\n'
                dialogue_list.extend(sub_sentences)
            else:
                sentence = sentence.strip(" ") + "\n"
                if i == length - 1:
                    sentence = sentence + "\n"
                dialogue_list.append(sentence.strip(" "))
    dialogue_list.pop(0)
    with open(save_path, 'w', encoding='utf-8') as f:
        f.writelines(dialogue_list)
    pass


if __name__ == '__main__':
    main(file_path="DailyDialog/validation.csv",
         save_path="data/valid_dialog.txt")
