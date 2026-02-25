import dataclasses 

@dataclasses.dataclass 
class BertConfig:
    vocab_size = 30522
    num_layers = 12 
    hidden_size = 768 
    num_heads = 12 
    dropout_prob = 0.1 
    pad_id = 0 
    max_seq_len = 512
    num_types = 2 
    
     
    