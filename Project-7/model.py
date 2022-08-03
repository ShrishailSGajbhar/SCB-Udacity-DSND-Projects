import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        """
        Initialize the Decoder class with necessary parameters for generating the image captions.
        
        Args:
        embed_size (int): Dimensionality of image and word embeddings
        hidden_sie (int): Dimensionality of hidden states in the Decoder
        vocab_size (int): Size of the vocabulary
        num_layers (int): number of layers
        """
        
        # call the superclass __init__ method
        super(DecoderRNN, self).__init__()
        
        # Creating the attribute variables
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        
        # Create an embedding layer which converts tokens (words) into vector of specified embed_size
        self.embed_layer = nn.Embedding(vocab_size, embed_size)
        # here vocab_size indicates size of the dictionary of embeddings and embed_size indicates the size of each embedding vector
        
        # Create an LSTM layer which converts the input embedding vector into hidden states
        self.lstm_layer = nn.LSTM(input_size = embed_size, hidden_size = hidden_size, 
                                  num_layers = num_layers, batch_first = True)
        
        # Create a fully connected layer that converts hidden states into output vector
        self.fc_layer = nn.Linear(hidden_size, vocab_size)
        
    
    def forward(self, features, captions):
        """
        Define forward function that computes output Tensors from input Tensors
        
        Args:
        features (tuple): context vector from Encoder of size (batch_size, embed_size)
        captions (tuple): indexes for the vocabulary words of size (batch_size, sentence_length)
        
        Returns:
        outputs (tuple): predicted output of shape (batch_size, captions.shape[1], vocab_size)
        """
        
        # Embed the caption except the last word <end>
        cap_embeddings = self.embed_layer(captions[:,:-1])
        
        # get the proper lstm input
        input_lstm = torch.cat((features.unsqueeze(1), cap_embeddings), 1)
        
        # Get the output of the lstm layer
        lstm_out, hidden = self.lstm_layer(input_lstm)
        
        # Get the final output
        outputs = self.fc_layer(lstm_out)
        
        return outputs
        

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        words = []
        for i in range(max_len):
            # get the hidden states and lstm output for the given iteration
            lstm_out, states = self.lstm_layer(inputs, states)
            # final output
            outputs = self.fc_layer(lstm_out.squeeze(1))
            # get the word index
            word_index = outputs.argmax(dim=1)
            words.append(word_index.item())
            # Get the input for next iteration
            inputs = self.embed_layer(word_index).unsqueeze(1)
            
        return words 