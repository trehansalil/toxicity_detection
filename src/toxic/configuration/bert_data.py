import torch
from torch.utils.data import Dataset


class BertDataSet(Dataset):

    def __init__(self, sentences, toxic_labels, tokenizer, max_len: int):
        self.sentences = sentences
        self.targets = toxic_labels.to_numpy()
        self.tokenizer, self.max_len = tokenizer, max_len

    def __len__(self):
        return len(self.sentences)


    def __getitem__(self, idx):
        
        sentence = self.sentences[idx]
        bert_senten = self.tokenizer.encode_plus(
            sentence,
            add_special_tokens = True, # [CLS],[SEP]
            max_length = self.max_len,
            pad_to_max_length = True,
            truncation = True,
            return_attention_mask = True            
        )
            
        ids = torch.tensor(bert_senten['input_ids'], dtype = torch.long)
        mask = torch.tensor(bert_senten['attention_mask'], dtype = torch.long)
        toxic_label = torch.tensor(self.targets[idx], dtype = torch.float)

        return {
            'ids' : ids,
            'mask' : mask,
            'toxic_label':toxic_label
        }