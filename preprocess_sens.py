import torch
from transformers import GPT2TokenizerFast
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from config import Args


def tokenizer_(filepath):
    assert filepath is not None
    tokenizer = GPT2TokenizerFast.from_pretrained(filepath)
    return tokenizer
    pass


class LoadSet(Dataset):
    def __init__(self, samples_set):
        self.samples_set = samples_set
        pass

    def __len__(self):
        return len(self.samples_set)

    def __getitem__(self, index):
        return self.samples_set[index]
        pass


class ChatSet:
    def __init__(self, raw_set_path, args):
        self.raw_set_path = raw_set_path
        self.tokenizer = tokenizer_(args.tokenizer_path)
        self.load_set = LoadSet
        self.args = args
        pass

    def read_raw_data(self):
        with open(self.raw_set_path, 'r', encoding='utf-8') as f:
            raw_set = f.read()
            if "\r\n" in raw_set:
                raw_set = raw_set.split('\r\n\r\n')
            else:
                raw_set = raw_set.split('\n\n')

        dialogue_list = []
        for index, a_dialogue in enumerate(tqdm(raw_set)):
            if '\r\n' in a_dialogue:
                group_sens = a_dialogue.split('\r\n')
            else:
                group_sens = a_dialogue.split('\n')
            input_ids = [self.tokenizer.bos_token_id]
            for a_sentence in group_sens:
                word_list = self.tokenizer.tokenize(a_sentence)
                ids_list = self.tokenizer.convert_tokens_to_ids(word_list) + [self.tokenizer.eos_token_id]
                input_ids += ids_list
            dialogue_list.append(input_ids)
            pass
        return dialogue_list
        pass

    def batch_function(self, batch):
        new_batch = []
        for a_dialogue in batch:
            if len(a_dialogue) > self.args.max_len:
                a_dialogue = a_dialogue[:self.args.max_len]
            new_batch.append(torch.tensor(a_dialogue, dtype=torch.long))
        inputs = torch.nn.utils.rnn.pad_sequence(new_batch,
                                                 batch_first=True, padding_value=0)
        labels = torch.nn.utils.rnn.pad_sequence(new_batch,
                                                 batch_first=True,
                                                 padding_value=self.args.label_padding)
        return inputs, labels
        pass

    def get_torch_set(self, shuffle=True):
        dialogue_set = self.read_raw_data()
        dialogue_set = self.load_set(dialogue_set)
        torch_loader = DataLoader(dataset=dialogue_set, batch_size=self.args.batch_size, num_workers=0,
                                  shuffle=shuffle, collate_fn=self.batch_function)
        return torch_loader
        pass


if __name__ == '__main__':
    args_instance = Args()
    args = args_instance.get_args()
    set_instance = ChatSet(args.train_set, args)
    train_loader = set_instance.get_torch_set()
    pass
