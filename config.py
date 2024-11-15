import argparse


class Args:
    @staticmethod
    def parser():
        parser = argparse.ArgumentParser()
        return parser
        pass

    @staticmethod
    def add_params(parser):
        filepath = 'your address'
        parser.add_argument('--train_set', default=filepath + 'data/train_dialog.txt')
        parser.add_argument('--valid_set', default=filepath + 'data/valid_dialog.txt')
        parser.add_argument('--tokenizer_path',
                            default='model/tokenizer.json', type=str)
        parser.add_argument('--checkpoint', default='model/pytorch_model.bin',
                            type=str)
        parser.add_argument('--checkpoint_path', default=filepath + 'checkpoint/fine_tuning_gpt2.pth', type=str)

        parser.add_argument('--d_model', default=768, type=int)
        parser.add_argument('--lora_r', default=8, type=int)
        parser.add_argument('--lora_alpha', default=32, type=int)
        parser.add_argument('--lora_dropout', default=0.1, type=float)

        parser.add_argument('--vocab_size', default=50257, type=int)
        parser.add_argument('--max_len', default=180, type=int)
        parser.add_argument('--label_padding', default=-100, type=int)
        parser.add_argument('--train_shuffle', default=True, type=bool)
        parser.add_argument('--batch_size', default=4, type=int)
        parser.add_argument('--epoch', default=30, type=int)
        parser.add_argument('--lr', default=0.0005, type=float)
        parser.add_argument('--betas', default=(0.998, 0.99), type=tuple)
        parser.add_argument('--eps', default=1e-6, type=float)
        parser.add_argument('--num_warmup', default=1000, type=int)
        parser.add_argument('--weight_decay', default=0.0001, type=float)
        parser.add_argument('--accumulated_steps', default=4, type=int)
        parser.add_argument('--grad_norm', default=1.0, type=float)
        parser.add_argument('--device', default='cuda:0', type=str)
        return parser
        pass

    def get_args(self):
        parser = self.parser()
        args = self.add_params(parser)
        return args.parse_args()
