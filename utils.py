from torch.nn.utils.rnn import pad_sequence 
from torch import stack

# Objects
choices = { "Is Automatically Generated": (0, {'False': 0, 'True': 1}), "Needs Action from User": (1, {'False': 0, 'True': 1}),
            "Is SPAM": (2, {'False': 0, 'True': 1}), "Is Business-Related": (3, {'False': 0, 'True': 1}), 
            "How is the Writing Style": (4, {'Informal': 0, 'Neutral': 1, 'Formal': 2})}


# Frunctions
def collate_fn_robert(examples):
    dataset_ids = [example['dataset_id'] for example in examples]
    personal_ids = [example['personal_id'] for example in examples]
    
    input_ids = pad_sequence([example['input_ids'] for example in examples], batch_first=True, padding_value=0)
    attention_mask = pad_sequence([example['attention_mask'] for example in examples], batch_first=True, padding_value=0)
    labels = {}
    labels['labels_choices'] = stack([example['labels_choices'] for example in examples])
    
    return {
        'dataset_id': dataset_ids,
        'personal_id': personal_ids,
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }

def collate_fn_xlm(examples):
    dataset_ids = [example['dataset_id'] for example in examples]
    personal_ids = [example['personal_id'] for example in examples]
    
    input_ids = pad_sequence([example['input_ids'] for example in examples], batch_first=True, padding_value=1)
    attention_mask = pad_sequence([example['attention_mask'] for example in examples], batch_first=True, padding_value=0)
    labels = {}
    labels['labels_choices'] = stack([example['labels_choices'] for example in examples])
    
    return {
        'dataset_id': dataset_ids,
        'personal_id': personal_ids,
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }