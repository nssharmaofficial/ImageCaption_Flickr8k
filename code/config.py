import os
import torch

class Config(object):
    
    def __init__(self) -> None:

        self.DEVICE = torch.device("cpu")
        
        self.BATCH = 32
        self.EPOCHS = 5
        
        self.VOCAB_FILE = 'word2index3000.txt'
        self.VOCAB_SIZE = 3000
        
        self.NUM_LAYER = 1
        self.IMAGE_EMB_DIM = 256
        self.WORD_EMB_DIM = 256
        self.HIDDEN_DIM = 512
        self.LR = 0.001
        
        self.EMBEDDING_WEIGHT_FILE = 'checkpoints/NEW_embeddings-32B-512H-1L-e5.pt'
        self.ENCODER_WEIGHT_FILE = 'checkpoints/NEW_encoder-32B-512H-1L-e5.pt'
        self.DECODER_WEIGHT_FILE = 'checkpoints/NEW_decoder-32B-512H-1L-e5.pt'
        
        self.ROOT = os.path.join(os.path.expanduser('~'), 'NN_projects', 'ImageCaption_Flickr8k') 

        
        
