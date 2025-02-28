"""Recompute BERT predictions on UKP dev/test without topic information."""
import os
import csv
import itertools

from tqdm import tqdm
from pytorch_pretrained_bert.tokenization import BertTokenizer
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from train import InputExample, convert_examples_to_features
from SigmoidBERT import SigmoidBERT


def inference(bert_output, test_file, eval_batch_size=32):
    """Perform inference."""
    # Import fine-tuned BERT model
    max_seq_length = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained(bert_output, do_lower_case=True)
    model = SigmoidBERT.from_pretrained(bert_output)
    model.to(device)
    model.eval()

    # Import test data
    test_sentences = set()
    with open(test_file, 'r') as csvfile:
        csvreader = csv.reader(csvfile, delimiter='\t', quotechar=None)
        for splits in csvreader:
            splits = map(str.strip, splits)
            __, sentence_a, sentence_b, __ = splits
            test_sentences.add(sentence_a)
            test_sentences.add(sentence_b)
    comb_iter = itertools.combinations(test_sentences, 2)

    input_examples = []
    output_examples = []
    for sentence_a, sentence_b in comb_iter:
        input_examples.append(InputExample(text_a=sentence_a,
                                           text_b=sentence_b,
                                           label=-1))
        output_examples.append([sentence_a, sentence_b, -1])

    eval_features = convert_examples_to_features(input_examples, max_seq_length, tokenizer)

    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)

    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=eval_batch_size)

    # Inference
    predicted_logits = []
    with torch.no_grad():
        for input_ids, input_mask, segment_ids in tqdm(eval_dataloader, desc="Batch"):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)

            logits = model(input_ids, segment_ids, input_mask)
            logits = logits.detach().cpu().numpy()
            predicted_logits.extend(logits[:, 0])

    for idx, logit in enumerate(predicted_logits):
        output_examples[idx].append(logit)

    # Export results
    eval_mode = os.path.basename(test_file).split(".")[0]
    output_pred_file = os.path.join(bert_output,
                                    f"{eval_mode}_predictions_epoch_3_no_topic_info.tsv")
    with open(output_pred_file, "w") as writer:
        for idx, example in enumerate(output_examples):
            sentence_a, sentence_b, gold_label, pred_logit = example
            writer.write("\t".join([sentence_a.replace("\n", " ").replace("\t", " "),
                                    sentence_b.replace("\n", " ").replace("\t", " "),
                                    str(gold_label), str(pred_logit)]))
            writer.write("\n")
