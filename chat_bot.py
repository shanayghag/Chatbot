# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 09:50:37 2019

@author: SHANAY
"""

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import os
import re
import codecs
import csv
import random
import itertools
import unicodedata
import warnings
warnings.filterwarnings("ignore")

device=torch.device("cpu")

#Data preprocessing
line_path=os.path.join("cornell movie-dialogs corpus","movie_lines.txt")
conv_path=os.path.join("cornell movie-dialogs corpus","movie_conversations.txt")

with open(line_path,'r') as file:
    lines=file.readlines()
for line in lines[:8]:
    print(line.strip())
    
# Split each line of file in a dictionary of fields
line_field=["lineID","characterID","movieID","character","text"]
lines={}
with open(line_path,'r',encoding='iso-8859-1') as f:
    for line in f:
        values=line.split(" +++$+++ ")
        lineObj={}
        for i,field in enumerate(line_field):
            lineObj[field]=values[i]
        lines[lineObj["lineID"]]=lineObj
        
# Group fields of lines from 'Load lines' into conversations based on movie_converstaions.txt
conv_field=["character1ID","character2ID","movieID","utteranceIDs"]
conversations=[]
with open(conv_path,'r',encoding='iso-8859-1') as f:
    for line in f:
        values=line.split(" +++$+++ ")
        convObj={}
        for i,field in enumerate(conv_field):
            convObj[field]=values[i]
        lineIds=eval(convObj["utteranceIDs"])
        convObj['line']=[]
        for ids in lineIds:
            convObj['line'].append(lines[ids])
        conversations.append(convObj)
        
qa_pairs=[]
for conversation in conversations:
    for i in range(len(conversation["line"])-1):
        inputLine=conversation["line"][i]["text"].strip()
        targetLine=conversation["line"][i+1]["text"].strip()
        if targetLine and inputLine:
            qa_pairs.append([inputLine,targetLine])
            
#Define path to the new file
datafile_path=os.path.join("cornell movie-dialogs corpus","formatted_text.txt")
delimiter='\t'
delimiter=str(codecs.decode(delimiter,"unicode_escape"))
print("\n Writing a newly formatted File.")
with open(datafile_path,'w',encoding="utf-8") as outputfile:
    writer=csv.writer(outputfile,delimiter=delimiter)
    for pair in qa_pairs:
        writer.writerow(pair)
print("Done writing to the File.....\n")

datafile_path=os.path.join("cornell movie-dialogs corpus","formatted_text.txt")
with open(datafile_path,'rb') as formatted_file:
    new_lines=formatted_file.readlines()
            
# Creating Vocabulary
'''
 Padding token for short sentences.
 Start of sentence token.
 End of Sentence token.
'''
PAD_token=0
SOS_token=1
EOS_token=2
'''
word2index=converts the words from sentence into random indexes
word2count=Checks the occurrence of a particular word
index2word=converts the indexes to word
above 3 are dictionaries
addSentence creates a list of sentences.
'''
class vocabulary:
    def __init__(self,name):
        self.name=name
        self.word2index={}
        self.word2count={}
        self.index2word={PAD_token:"PAD",SOS_token:"SOS",EOS_token:"EOS"}
        self.num_words=3
        
    def addSentence(self,sentence):
        for word in sentence.split():
            self.addWord(word)
            
    def addWord(self,word):
        if word not in self.word2index:
            self.word2index[word]=self.num_words
            self.word2count[word]=1
            self.index2word[self.num_words]=word
            self.num_words +=1
        else:
            self.word2count[word] +=1
            
    # Remove words below certain threshold
    def trim(self,min_count):
        keep_words=[]
        for k,v in self.word2count.items():
            if v>=min_count:
                keep_words.append(k)
        print("Keep words {}/{}={:.4f}".format(len(keep_words),len(self.word2index),len(keep_words)/len(self.word2index)))
        #Reintialise the Dictionaries and store words above a certain threshold 
        self.word2index={}
        self.word2count={}
        self.index2word={PAD_token:"PAD",SOS_token:"SOS",EOS_token:"EOS"}
        self.num_words=3
        for word in keep_words:
            self.addWord(word)
            
def unicode2Ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD',s) if unicodedata.category(c)!='Mn')

def normalizeString(s):
    s=unicode2Ascii(s.lower().strip())
    s=re.sub(r"([.!?])",r" \1",s)
    s=re.sub(r"[^a-zA-Z.!?]+",r" ",s)
    s=re.sub(r"\s+",r" ",s).strip()
    return s

datafile_path=os.path.join("cornell movie-dialogs corpus","formatted_text.txt")
print("Reading and processing the file.....please wait!")
formatted_lines=open(datafile_path,encoding="utf-8").read().strip().split('\n')
#split every line into pairs and normalise it
pairs=[[normalizeString(s) for s in pair.split('\t')] for pair in formatted_lines]
print("Done Reading!")
voc=vocabulary("cornell movie-dialogs corpus")

# Returns true if both the sentences in the pair are below the threshold
Max_Length=10
def  filterpair(p):
    return len(p[0].split())<Max_Length and len(p[1].split())<Max_Length

def filterpairs(pairs):
    return[pair for pair in pairs if filterpair(pair)]
    
pairs=[pair for pair in pairs if len(pair)>1]
print("There are {} conversation pairs in the dataset".format(len(pairs)))
pairs=filterpairs(pairs)
print("After filtering there are {}".format(len(pairs)))

for pair in pairs:
    voc.addSentence(pair[0])
    voc.addSentence(pair[1])
print("Counted Words in Vocabulary:{}".format(voc.num_words))

MIN_COUNT=3
def trimRareWords(voc,pairs,MIN_COUNT):
    voc.trim(MIN_COUNT)
    keep_new_pairs=[]
    for pair in pairs:
        input_sentence=pair[0]
        output_sentence=pair[1]
        keep_input= True
        keep_output= True
        
        #Check input sentence
        for word in input_sentence.split(' '):
            if word not in voc.word2index:
                keep_input=False
                break
        
        #Check output sentence
        for word in output_sentence.split(' '):
            if word not in voc.word2index:
                keep_output= False
                break
            
        if keep_input and keep_output:
            keep_new_pairs.append(pair)
    
    print("Trimmed from {} words to {} words".format(len(pairs),len(keep_new_pairs)))
    return keep_new_pairs

trimmed_pairs=trimRareWords(voc,pairs,MIN_COUNT)

def indexfromSentences(voc,sentence):
    return [voc.word2index[word] for word in sentence.split(' ')] + [EOS_token]
    
#for testing
    test=indexfromSentences(voc,trimmed_pairs[1][0])
    print(test)
    
#Define some sample for testing
inp=[]
out=[]
for pair in trimmed_pairs[:10]:
    inp.append(pair[0])
    out.append(pair[1])
print(inp)
print(len(inp))
indexes=[indexfromSentences(voc,sentence) for sentence in inp]
print(indexes)
    
def zeropadding(l,fillValue=0):
    return list(itertools.zip_longest(*l,fillvalue=fillValue))

length=[len(ind) for ind in indexes]
print(max(length))

#Testing the function
test_result=zeropadding(indexes)
print(test_result)
print(len(test_result))

def binaryMatrix(l,value=0):
    m=[]
    for i,seq in enumerate(l):
        m.append([])
        for token in seq:
            if token==PAD_token:
                m[i].append(0)
            else:
                m[i].append(1)
    return m

def inputVar(l,voc):
    indexes_batch=[indexfromSentences(voc,sentence) for sentence in l]
    lengths=torch.tensor([len(index) for index in indexes_batch])
    padList=zeropadding(indexes_batch)
    padVar=torch.LongTensor(padList)
    return padVar,lengths

def outVar(l,voc):
    indexes_batch=[indexfromSentences(voc,sentence) for sentence in l]
    max_target_len=max([len(indexes) for indexes in indexes_batch])
    padList=zeropadding(indexes_batch)
    mask=binaryMatrix(padList)
    mask=torch.ByteTensor(mask)
    padVar=torch.LongTensor(padList)
    return padVar,mask,max_target_len
 
def batch2TrainData(voc,pair_batch):
    pair_batch.sort(key= lambda x: len(x[0].split()),reverse=True)
    input_batch=[]
    output_batch=[]
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])
    inp,lengths=inputVar(input_batch,voc)
    output,mask,max_target_length=outVar(output_batch,voc)
    return inp,lengths,output,mask,max_target_length

# Test Validations for above function
small_batch_size=64
batches=batch2TrainData(voc,[random.choice(trimmed_pairs) for i in range(small_batch_size)])
input_variable,lengths,target_variable,mask,max_target_length=batches
print("Input variable:")
print(input_variable)
print("Lengths:",lengths)
print("Target Variable:")
print(target_variable)
print("Mask:")
print(mask)
print("Max Length:",max_target_length)
    
# Preparation of the model
class EncoderRNN(nn.Module):
    def __init__(self,hidden_size,embedding,n_layers=1,dropout=0):
        super(EncoderRNN,self).__init__()
        self.n_layers=n_layers
        self.hidden_size=hidden_size
        self.embedding=embedding
        self.gru=nn.GRU(hidden_size,hidden_size,n_layers,dropout=(0 if n_layers==1 else dropout),bidirectional=True)
        
    def forward(self,input_seq,input_lengths,hidden=None):
        embedded=self.embedding(input_seq)
        packed=torch.nn.utils.rnn.pack_padded_sequence(embedded,input_lengths)
        outputs, hidden=self.gru(packed,hidden)
        outputs,_=torch.nn.utils.rnn.pad_packed_sequence(outputs)
        outputs= outputs[:,:,:self.hidden_size] + outputs[:,:,self.hidden_size:]
        return outputs,hidden
    
class Attn(torch.nn.Module):
    def __init__(self,method,hidden_size):
        super(Attn,self).__init__()
        self.method=method
        self.hidden_size=hidden_size
        
    def dot_score(self,hidden,encoder_output):
        #Element wise multiply the current decoder state with the encoder output and then sum it.
        #Dim=2 means sum across the hidden i.e for each row sum across each column.
        return torch.sum(hidden * encoder_output,dim=2)
    
    def forward(self,hidden,encoder_output):
        attn_energies=self.dot_score(hidden,encoder_output)
        attn_energies= attn_energies.t()
        return F.softmax(attn_energies,dim=1).unsqueeze(1)
    
class LuongAttnDecoderRNN(nn.Module):
    def __init__(self,attn_model,embedding,hidden_size,output_size,n_layers=1,dropout=0.1):
        super(LuongAttnDecoderRNN,self).__init__()
        self.attn_model=attn_model
        self.hidden_size=hidden_size
        self.output_size=output_size
        self.n_layers=n_layers
        self.dropout=dropout
        
        
        #Define Layers
        self.embedding=embedding
        self.embedding_dropout=nn.Dropout(dropout)
        self.gru=nn.GRU(hidden_size,hidden_size,n_layers,dropout=(0 if n_layers==1 else dropout))
        self.concat=nn.Linear(hidden_size*2,hidden_size)
        self.out=nn.Linear(hidden_size,output_size)
        
        self.attn=Attn(attn_model,hidden_size)
        
    def forward(self,input_step,last_hidden,encoder_output):
        embedded=self.embedding(input_step)
        embedded=self.embedding_dropout(embedded)
        #dropout reduces over-fitting
        rnn_output,hidden=self.gru(embedded,last_hidden)
        attn_weights=self.attn(rnn_output,encoder_output)
        context=attn_weights.bmm(encoder_output.transpose(0,1))
        rnn_output=rnn_output.squeeze(0)
        context=context.squeeze(1)
        concat_input=torch.cat((rnn_output,context),1)
        concat_output=torch.tanh(self.concat(concat_input))
        output=self.out(concat_output)
        output=F.softmax(output,dim=1)
        return output,hidden
    
#Training starts
#Creating a Loss Function
#NLLLOSS: Negative log likely loss
def maskNLLLoss(decoder_out,target,mask):
    nTotal=mask.sum() #Total number of non-zero elements
    target=target.view(-1,1) #Reshaping the tensor to one column
    gathered_tensor=torch.gather(decoder_out,1,target)
    crossEntropy=-torch.log(gathered_tensor)
    loss=crossEntropy.masked_select(mask)
    loss=loss.mean()
    loss=loss.to(device)
    return loss,nTotal.item()

#Visualizing the training 
'''
Teacher forcing: Supplying the correct word to the LSTM regardless of the previous output. Basically Hardcoding the next
input of the LSTM. If we set Teacher Forcing ratio to 0.5 then for 1/2 of training time teacher forcing will be enabled.

small_batch_size=5
batches=batch2TrainData(voc,[random.choice(trimmed_pairs) for i in range(small_batch_size)])
print(batches)
input_variable,lengths,target_variable,mask,max_target_length=batches
print("Input variable:")
print(input_variable)
print("Lengths:",lengths)
print("Target Variable:")
print(target_variable)
print("Mask:")
print(mask)
print("Max Length:",max_target_length)


#Defining the parameters
hidden_size=500
encoder_n_layers=2
decoder_n_layers=2
dropout=0.1
attn_model='dot'
embedding=nn.Embedding(voc.num_words,hidden_size)


#Define Encoder and Decoder
encoder=EncoderRNN(hidden_size,embedding,encoder_n_layers,dropout)
decoder=LuongAttnDecoderRNN(attn_model,embedding,hidden_size,voc.num_words,decoder_n_layers,dropout)
encoder=encoder.to(device)
decoder=decoder.to(device)
'''

'''
#Training the objects
encoder.train()
decoder.train()
'''

def train(input_variable,lengths,target_variable,mask,max_target_length,encoder,decoder,embedding,encoder_optimizer,decoder_optimizer,batch_size,clip,max_len=Max_Length):
    #Initialize the Optimizers
    #Optimizers are used for back propagation(ADAM instead of Gardient descent)
    encoder_optimizer=optim.Adam(encoder.parameters(),lr=0.0001)
    decoder_optimizer=optim.Adam(decoder.parameters(),lr=0.0001)
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_variable=input_variable.to(device)
    lengths=lengths.to(device)
    target_variable=target_variable.to(device)
    mask=mask.to(device)
    
    loss=0
    print_losses=[]
    n_totals=0

    encoder_outputs,encoder_hidden=encoder(input_variable,lengths)
    #print("Encoder Output Shape:",encoder_outputs.shape)
    #print("Last Encoder Hidden State Shape:",encoder_hidden.shape)

    decoder_input=torch.LongTensor([[SOS_token for i in range(batch_size)]])
    decoder_input=decoder_input.to(device)
    #print("Initial Decoder Input Shape:",decoder_input.shape)
    #print(decoder_input)

    #Set initial decoder hidden state to encoder last hidden state
    decoder_hidden=encoder_hidden[:decoder.n_layers]
   
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    
    
    if use_teacher_forcing:
        for t in range(max_target_length):
            decoder_output,decoder_hidden=decoder(decoder_input,decoder_hidden,encoder_outputs)
            #Teacher forcing: Next input is current Target.
            decoder_input=target_variable[t].view(1,-1)
            #Calculate and Accumulate the Loss
            mask_loss, nTotal=maskNLLLoss(decoder_output,target_variable[t],mask[t])
            loss+=mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals+=nTotal
            
    else:
         for t in range(max_target_length):
            decoder_output,decoder_hidden=decoder(decoder_input,decoder_hidden,encoder_outputs)
            # NO Teacher forcing: Next input is decoders own output.
            _,topi= decoder_output.topk(1)
            decoder_input=torch.LongTensor([[topi[i][0] for i in range (batch_size)]])
            decoder_input= decoder_input.to(device)
            #Calculate and Accumulate the Loss
            mask_loss, nTotal=maskNLLLoss(decoder_output,target_variable[t],mask[t])
            loss+=mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals+=nTotal
            
    #perform back propagation
    loss.backward()
    
     # Clip gradients: gradients are modified in place
    _ = torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    _ = torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)

    encoder_optimizer.step()
    decoder_optimizer.step()
    returned_loss=sum(print_losses)/n_totals
    return returned_loss

# Training Iterations

def trainIters(model_name, voc, trimmed_pairs, encoder, decoder, encoder_optimizer, decoder_optimizer, embedding, encoder_n_layers, decoder_n_layers, save_dir, n_iteration, batch_size, print_every, save_every, clip, corpus_name, loadFilename):

    try:
        # Load batches for each iteration
        training_batches = [batch2TrainData(voc, [random.choice(trimmed_pairs) for _ in range(batch_size)]) for _ in range(n_iteration)]
    
        # Initializations
        print('Initializing ...')
        start_iteration = 1
        print_loss = 0
        if loadFilename:
            start_iteration = checkpoint['iteration'] + 1

        # Training loop
        print("Training...")
        for iteration in range(start_iteration, n_iteration + 1):
            training_batch = training_batches[iteration - 1]
            # Extract fields from batch
            input_variable, lengths, target_variable, mask, max_target_length = training_batch
            
            # Run a training iteration with batch
            loss = train(input_variable, lengths, target_variable, mask, max_target_length, encoder,decoder, embedding, encoder_optimizer, decoder_optimizer, batch_size, clip, max_len=Max_Length)
            print_loss += loss

            # Print progress
            if iteration % print_every == 0:
                print_loss_avg = print_loss / print_every
                print("Iteration: {}; Percent complete: {:.1f}%; Average loss: {:.4f}".format(iteration, iteration / n_iteration * 100, print_loss_avg))
                print_loss = 0

            # Save checkpoint
            if (iteration % save_every == 0):
                directory = os.path.join(save_dir, model_name, corpus_name, '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size))
                if not os.path.exists(directory):
                    os.makedirs(directory)
                    torch.save({
                            'iteration': iteration,
                            'en': encoder.state_dict(),
                            'de': decoder.state_dict(),
                            'en_opt': encoder_optimizer.state_dict(),
                            'de_opt': decoder_optimizer.state_dict(),
                            'loss': loss,
                            'voc_dict': voc.__dict__,
                            'embedding': embedding.state_dict()
                            }, os.path.join(directory, '{}_{}.tar'.format(iteration, 'checkpoint')))
    except KeyError:
            print("Error: Encountered unknown word.")
            
class GreedySearchDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(GreedySearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_seq, input_length, max_length):
        # Forward input through encoder model
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)
        # Prepare encoder's final hidden layer to be first hidden input to the decoder
        decoder_hidden = encoder_hidden[:decoder.n_layers]
        # Initialize decoder input with SOS_token
        decoder_input = torch.ones(1, 1, device=device, dtype=torch.long) * SOS_token
        # Initialize tensors to append decoded words to
        all_tokens = torch.zeros([0], device=device, dtype=torch.long)
        all_scores = torch.zeros([0], device=device)
        # Iteratively decode one word token at a time
        for _ in range(max_length):
            # Forward pass through decoder
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            # Obtain most likely word token and its softmax score
            decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
            # Record token and score
            all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
            all_scores = torch.cat((all_scores, decoder_scores), dim=0)
            # Prepare current token to be next decoder input (add a dimension)
            decoder_input = torch.unsqueeze(decoder_input, 0)
        # Return collections of word tokens and scores
        return all_tokens, all_scores

def evaluate(encoder, decoder, searcher, voc, sentence, max_length=Max_Length):
    ### Format input sentence as a batch
    # words -> indexes
    indexes_batch = [indexfromSentences(voc, sentence)]
    # Create lengths tensor
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    # Transpose dimensions of batch to match models' expectations
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
    # Use appropriate device
    input_batch = input_batch.to(device)
    lengths = lengths.to(device)
    # Decode sentence with searcher
    tokens, scores = searcher(input_batch, lengths, max_length)
    # indexes -> words
    decoded_words = [voc.index2word[token.item()] for token in tokens]
    return decoded_words


def evaluateInput(encoder, decoder, searcher, voc):
    input_sentence = ''
    while(1):
        try:
            # Get input sentence
            input_sentence = input('> ')
            # Check if it is quit case
            if input_sentence == 'q' or input_sentence == 'quit': break
            # Normalize sentence
            input_sentence = normalizeString(input_sentence)
            # Evaluate sentence
            output_words = evaluate(encoder, decoder, searcher, voc, input_sentence)
            # Format and print response sentence
            output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
            print('Bot:', ' '.join(output_words))

        except KeyError:
            print("Error: Encountered unknown word.")
            
# Configure models
model_name = 'cb_model'
attn_model = 'dot'
#attn_model = 'general'
#attn_model = 'concat'
hidden_size = 500
encoder_n_layers = 2
decoder_n_layers = 2
dropout = 0.1
batch_size = 64

# Set checkpoint to load from; set to None if starting from scratch
loadFilename = None
checkpoint_iter = 4000
#loadFilename = os.path.join(save_dir, model_name, corpus_name,
#                            '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size),
#                            '{}_checkpoint.tar'.format(checkpoint_iter))


# Load model if a loadFilename is provided
if loadFilename:
    # If loading on same machine the model was trained on
    checkpoint = torch.load(loadFilename)
    # If loading a model trained on GPU to CPU
    #checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
    encoder_sd = checkpoint['en']
    decoder_sd = checkpoint['de']
    encoder_optimizer_sd = checkpoint['en_opt']
    decoder_optimizer_sd = checkpoint['de_opt']
    embedding_sd = checkpoint['embedding']
    voc.__dict__ = checkpoint['voc_dict']


print('Building encoder and decoder ...')
# Initialize word embeddings
embedding = nn.Embedding(voc.num_words, hidden_size)
if loadFilename:
    embedding.load_state_dict(embedding_sd)
# Initialize encoder & decoder models
encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout)
if loadFilename:
    encoder.load_state_dict(encoder_sd)
    decoder.load_state_dict(decoder_sd)
# Use appropriate device
encoder = encoder.to(device)
decoder = decoder.to(device)
print('Models built and ready to go!')

# Configure training/optimization
clip = 50.0
teacher_forcing_ratio = 1.0
learning_rate = 0.0001
decoder_learning_ratio = 5.0
n_iteration = 4000
print_every = 1
save_every = 500
corpus_name= "cornell movie-dialogs corpus"
save_dir="trained_model"

# Ensure dropout layers are in train mode
encoder.train()
decoder.train()

# Initialize optimizers
print('Building optimizers ...')
encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)
if loadFilename:
    encoder_optimizer.load_state_dict(encoder_optimizer_sd)
    decoder_optimizer.load_state_dict(decoder_optimizer_sd)

# Run training iterations
print("Starting Training!")
trainIters(model_name, voc, trimmed_pairs, encoder, decoder, encoder_optimizer, decoder_optimizer,
           embedding, encoder_n_layers, decoder_n_layers, save_dir, n_iteration, batch_size,
           print_every, save_every, clip, corpus_name, loadFilename)

# Set dropout layers to eval mode
encoder.eval()
decoder.eval()

# Initialize search module
searcher = GreedySearchDecoder(encoder, decoder)

# Uncomment the following line after training the model
#evaluateInput(encoder, decoder, searcher, voc)
    
    
    
        
        
                
    
    
    
        
        
