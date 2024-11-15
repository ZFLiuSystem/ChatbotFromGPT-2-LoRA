import torch
from torch.nn import functional as F
from model import ChatModel
from transformers import AdamW, get_scheduler
from preprocess_sens import ChatSet
from config import Args
import time
from tqdm import tqdm


class Trainer:
    def __init__(self, args, train_loader, val_loader,):
        self.args = args

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.model = ChatModel(model_path=args.checkpoint, embed_dim=args.d_model, vocab_size=args.vocab_size,
                               lora_r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout)
        model_parameters = self.set_parameters()

        self.device = torch.device(args.device)
        self.optimizer = AdamW(params=model_parameters, lr=args.lr, betas=args.betas, eps=args.eps,
                               no_deprecation_warning=True)
        self.scheduler = get_scheduler(name="linear", optimizer=self.optimizer,
                                       num_training_steps=len(train_loader) * args.epoch // args.accumulated_steps,
                                       num_warmup_steps=args.num_warmup // args.accumulated_steps)
        pass

    def set_parameters(self):
        model_params = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimized_params = [
            {'params': [p for p_name, p in model_params if not any(nd in p_name for nd in no_decay)],
             'weight_decay': self.args.weight_decay},
            {'params': [p for p_name, p in model_params if any(nd in p_name for nd in no_decay)],
             'weight_decay': .0}]
        return optimized_params
        pass

    @staticmethod
    def calculate_loss(logits, labels_ids, ignore_index=-100):
        loss_fn = F.cross_entropy
        lan_logits = logits[:, :-1, :].contiguous()
        lan_labels = labels_ids[:, 1:].contiguous()
        lan_logits = lan_logits.view(-1, lan_logits.size(-1))
        lan_labels = lan_labels.view(-1)
        loss = loss_fn(lan_logits, lan_labels, ignore_index=ignore_index)
        return loss
        pass

    def generate_attn_mask(self, input_ids):
        init_mask = torch.ones([input_ids.size(0), input_ids.size(-1)], device=self.device)
        mask_marker = (input_ids != 0).to(self.device)
        attn_mask = torch.where(mask_marker, init_mask, 0)
        return attn_mask
        pass

    @staticmethod
    def calculate_acc(logits, labels, ignore_index=-100):
        logits = logits[:, :-1, :].cpu().contiguous().view(-1, logits.size(-1))
        labels = labels[:, 1:].cpu().contiguous().view(-1)
        token_ids = torch.argmax(logits, dim=1)
        non_pad_mask = labels.ne(ignore_index)
        num_correct_ids = token_ids.eq(labels).masked_select(non_pad_mask).cpu().sum().item()
        num_words = non_pad_mask.sum().item()
        return num_correct_ids, num_words
        pass

    def save_checkpoint(self, checkpoint: dict):
        torch.save(checkpoint, f=self.args.checkpoint_path)
        pass

    def train(self):
        self.model.train()
        self.model.to(self.device)
        best_acc = .0
        for epoch in range(1, self.args.epoch + 1):
            total_acc = .0
            total_correct_ids = 0
            total_loss = .0
            gradient_accumulation_counter = 0
            print(80 * '-')
            start_time = time.time()
            for batch_id, (input_ids, labels_ids) in tqdm(enumerate(self.train_loader)):
                attention_mask = self.generate_attn_mask(input_ids)
                input_ids = input_ids.to(self.device)
                labels_ids = labels_ids.to(self.device)

                logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
                loss = self.calculate_loss(logits=logits, labels_ids=labels_ids)
                total_loss += loss.cpu().item()

                batch_correct_ids, batch_total_ids = self.calculate_acc(logits, labels_ids)
                total_acc += (batch_correct_ids / batch_total_ids)
                total_correct_ids += batch_correct_ids

                '''Loss equalization, back propagation and gradient accumulation'''
                assert self.args.accumulated_steps > 1
                # Loss equalization
                loss /= self.args.accumulated_steps
                # Back propagation
                loss.backward()
                # Gradient accumulation
                gradient_accumulation_counter += 1
                '''Gradient clipping, parameter updating and learning rate updating'''
                if gradient_accumulation_counter % self.args.accumulated_steps == 0:
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(),
                                                   max_norm=self.args.grad_norm)
                    # Parameter updating
                    self.optimizer.step()
                    # Learning rate updating
                    self.scheduler.step()
                    # Gradient clearing
                    self.optimizer.zero_grad()
                    # Reset gradient accumulation
                    gradient_accumulation_counter = 0
                pass
            end_time = time.time()
            epoch_span = end_time - start_time
            epoch_train_loss = total_loss / len(self.train_loader)
            epoch_train_acc = total_acc / len(self.train_loader)
            epoch_train_correct_ids = total_correct_ids / len(self.train_loader)
            prompts = ('Epoch[{:03d}/{:03d}], Train Time[{:.2f} seconds]:\n'
                       'Train Loss: {:.2f}, Train_ids_acc: {:.2f}, Train_correct_ids: {:.2f}.')
            print(prompts.format(self.args.epoch, epoch, epoch_span,
                                 epoch_train_loss, epoch_train_acc, epoch_train_correct_ids))
            best_acc = self.valid(epoch, best_acc=best_acc)
            print(80 * '-')
        pass

    def valid(self, epoch, best_acc=None):
        self.model.eval()
        total_loss = .0
        total_acc = .0
        total_correct_ids = 0
        start_time = time.time()
        for batch_id, (inputs_ids, labels_ids) in tqdm(enumerate(self.val_loader)):
            attention_masks = self.generate_attn_mask(inputs_ids)
            inputs_ids = inputs_ids.to(self.device)
            labels_ids = labels_ids.to(self.device)

            logits = self.model(input_ids=inputs_ids, attention_mask=attention_masks)
            loss = self.calculate_loss(logits, labels_ids=labels_ids)
            total_loss += loss.cpu().item()

            batch_correct_ids, batch_total_ids = self.calculate_acc(logits, labels_ids)
            total_acc += (batch_correct_ids / batch_total_ids)
            total_correct_ids += batch_correct_ids
        end_time = time.time()
        span_time = end_time - start_time
        epoch_val_loss = total_loss / len(self.val_loader)
        epoch_val_acc = total_acc / len(self.val_loader)
        epoch_val_correct_ids = total_correct_ids / len(self.val_loader)
        prompts = ('Epoch[{:03d}/{:03d}], Valid Time[{:.2f} seconds]:\n'
                   'Valid Loss: {:.2f}, Valid_ids_acc: {:.2f}, Valid_correct_ids: {:.2f}.')
        print(prompts.format(self.args.epoch, epoch, span_time,
                             epoch_val_loss, epoch_val_acc, epoch_val_correct_ids))
        if best_acc is not None:
            if epoch_val_acc > best_acc:
                best_acc = epoch_val_acc
                checkpoint = {'model': self.model.state_dict(),
                              'optimizer': self.optimizer.state_dict(),
                              'loss': epoch_val_loss,
                              'epoch': epoch,
                              }
                self.save_checkpoint(checkpoint)
        return best_acc
        pass


def main():
    args_instance = Args()
    args = args_instance.get_args()
    train_set = ChatSet(raw_set_path=args.train_set, args=args)
    val_set = ChatSet(raw_set_path=args.valid_set, args=args)
    train_loader = train_set.get_torch_set(shuffle=args.train_shuffle)
    valid_loader = val_set.get_torch_set(shuffle=False)
    trainer = Trainer(args, train_loader, valid_loader)
    trainer.train()
    pass


if __name__ == "__main__":
    main()

