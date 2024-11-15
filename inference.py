import argparse
import torch
from torch.nn import functional as f
from model import ChatModel
from preprocess_sens import tokenizer_
import copy


def set_infer_args():
    filepath = 'your address'
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', default=True, type=bool)
    parser.add_argument('--tokenizer_path', default=filepath + 'model', type=str)
    parser.add_argument('--checkpoint', default=filepath + 'model', type=str)
    parser.add_argument('--checkpoint_path', default=filepath + 'checkpoint/fine_tuning_gpt2.pth', type=str)
    parser.add_argument('--vocab_size', default=50257, type=int)
    parser.add_argument('--max_history_len', default=18, type=int)

    parser.add_argument('--d_model', default=768, type=int)
    parser.add_argument('--lora_r', default=8, type=int)
    parser.add_argument('--lora_alpha', default=32, type=int)
    parser.add_argument('--lora_dropout', default=0.1, type=float)

    parser.add_argument('--batch_size', default=6, type=int)
    parser.add_argument('--max_sens_len', default=50, type=int)
    parser.add_argument('--penalty', default=1.5, type=float)
    parser.add_argument('--temperature', default=0.8, type=float)
    parser.add_argument('--top_k', default=8, type=int)
    parser.add_argument('--top_p', default=0.0, type=float)
    return parser.parse_args()
    pass


def top_k_top_p_filter(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    assert logits.dim() == 2
    top_k = min(top_k, logits[0].size(-1))
    if top_k > 0:
        _, topk_indices = torch.topk(logits, top_k, dim=1)
        mask = torch.zeros_like(logits, dtype=torch.bool)
        mask.scatter_(dim=1, index=topk_indices, value=True)
        logits = torch.where(condition=mask, input=logits, other=torch.full_like(logits, filter_value))
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(f.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        for ids, logit in enumerate(logits):
            indices_to_remove = sorted_indices[ids][sorted_indices_to_remove[ids]]
            logit[indices_to_remove] = filter_value
            pass
    return logits
    pass


def inference_major(args):
    device = 'cuda:0' if args.cuda else 'cpu'

    tokenizer = tokenizer_(args.tokenizer_path)
    checkpoint = torch.load(args.checkpoint_path)
    dialogue_model = ChatModel(model_path=args.checkpoint, embed_dim=args.d_model, vocab_size=args.vocab_size,
                               lora_r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout)
    dialogue_model.load_state_dict(checkpoint['model'])
    dialogue_model.eval()
    dialogue_model.to(device)

    eval_model = ChatModel(model_path=args.checkpoint, embed_dim=args.d_model, vocab_size=args.vocab_size,
                           lora_r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout)
    eval_model.load_state_dict(checkpoint['model'])

    print(90 * '-')
    print(eval_model)
    print(90 * '-')

    eval_model.eval()
    eval_model.to(device)
    loss_fn = f.cross_entropy
    print("Please chat with the chatbot.")

    history = []
    start_id = 1
    while True:
        try:
            user_cont = input("User: ")
            if user_cont in ['Goodbye', 'Bye']:
                break
            cont_ids = tokenizer.encode(user_cont)
            history.append(cont_ids)

            if len(history) > 8:
                history = history[start_id:]
                start_id += 1

            input_ids = [tokenizer.bos_token_id]
            for history_id, history_cont in enumerate(history[-args.max_history_len:]):
                input_ids.extend(history_cont)
                input_ids.append(tokenizer.eos_token_id)
            input_ids = [copy.deepcopy(input_ids) for _ in range(args.batch_size)]
            inputs_ids_tensors = torch.tensor(input_ids, dtype=torch.long, device=device)

            generated_ids = []
            finish_set = set()
            for _ in range(args.max_sens_len):
                outputs = dialogue_model(input_ids=inputs_ids_tensors)
                later_token_logits = outputs[:, -1, :]

                for ids in range(args.batch_size):
                    for token_id in set(token_ids[ids] for token_ids in generated_ids):
                        if later_token_logits[ids][token_id] > 0:
                            later_token_logits[ids][token_id] /= args.penalty
                        else:
                            later_token_logits[ids][token_id] *= args.penalty
                later_token_logits = later_token_logits / args.temperature

                filtered_logits = top_k_top_p_filter(later_token_logits, args.top_k, args.top_p)
                filtered_logits = f.softmax(filtered_logits, dim=-1)
                next_token = torch.multinomial(input=filtered_logits, num_samples=1, replacement=False)

                for ids, token_ids in enumerate(next_token[:, 0]):
                    if token_ids == tokenizer.eos_token_id:
                        finish_set.add(ids)
                    pass
                finish_cycle = True
                for ids in range(args.batch_size):
                    if ids not in finish_set:
                        finish_cycle = False
                        break
                if finish_cycle:
                    break

                generated_ids.append([token.item() for token in next_token[:, 0]])
                inputs_ids_tensors = torch.cat([inputs_ids_tensors, next_token], dim=-1)

            candidate_responses = []
            for batch_ids in range(args.batch_size):
                response = []
                for token_ids in range(len(generated_ids)):
                    if generated_ids[token_ids][batch_ids] != tokenizer.eos_token_id:
                        response.append(generated_ids[token_ids][batch_ids])
                    else:
                        break
                    pass
                candidate_responses.append(response)

            min_loss = float('Inf')
            best_response = ""
            for response in candidate_responses:
                eval_input_ids = [tokenizer.bos_token_id]
                eval_input_ids.extend(response)
                eval_input_ids += [tokenizer.eos_token_id]
                for history_ids in reversed(history[-args.max_history_len:]):
                    eval_input_ids.extend(history_ids)
                    eval_input_ids.append(tokenizer.eos_token_id)
                eval_input_tensor = torch.tensor(eval_input_ids, dtype=torch.long, device=device).unsqueeze(0)
                logits = eval_model(input_ids=eval_input_tensor)[..., :-1, :].contiguous()
                labels = eval_input_tensor[..., 1:].contiguous()
                loss = loss_fn(input=logits.view(-1, logits.size(-1)), target=labels.view(-1))
                if loss.cpu().item() < min_loss:
                    best_response = response
                    min_loss = loss.cpu().item()
            history.append(best_response)
            chatbot_tokens = tokenizer.convert_ids_to_tokens(best_response)
            chatbot_cont = tokenizer.convert_tokens_to_string(chatbot_tokens)
            print("Chatbot: " + chatbot_cont)
            pass

        except KeyboardInterrupt:
            break
    pass


if __name__ == '__main__':
    args = set_infer_args()
    inference_major(args)
    pass

