import re
import nltk
import torch
import string
import numpy as np
import pandas as pd
import seaborn as sns
import torch.nn as nn
import matplotlib.pyplot as plt

from constants import *
from BERT_Arch import BERT_Arch
from transformers import BertModel, BertTokenizerFast, AdamW
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from timeit import default_timer as timer

def clean_content(s):
	"""Given a sentence remove its punctuation and stop words"""
	if not isinstance(s,str):
		s = str(s)																				# Convert to string
	s = s.lower()																				# Convert to lowercase
	s = s.translate(str.maketrans('','',string.punctuation))									# Remove punctuation
	s = re.sub(r'([\;\:\|â€¢Â«\n])', ' ', s)													# Remove special characters
	s = re.sub(r'(@.*?)[\s]', ' ', s)															# Remove '@name'
	s = re.sub(r'&amp;', '&', s)																# Replace '&amp;' with '&'
	
	tokens = word_tokenize(s)
	stop_words = stopwords.words('english')
	ps = PorterStemmer()
	cleaned_s = ' '.join([ps.stem(w) for w in tokens if w not in stop_words or w in ['not', 'can']])		# Remove stop-words and stem
	cleaned_s = re.sub(r'\s+', ' ', cleaned_s).strip()														# Replace multi whitespace with single whitespace
	return cleaned_s

def show_preds(preds, test_labels):
	print(f"Accuray: {round(accuracy_score(test_labels, preds), 5) * 100}%")
	print(f"ROC-AUC: {round(roc_auc_score(test_labels, preds), 5) * 100}%")

	fig = plt.figure(figsize=(10,4))
	heatmap = sns.heatmap(data = pd.DataFrame(confusion_matrix(test_labels, preds)), annot = True, fmt = "d", cmap=sns.color_palette("Reds", 50))
	heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha="right", fontsize=14)
	heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha="right", fontsize=14)
	plt.ylabel("Ground Truth")
	plt.xlabel("Prediction")
	plt.show()

def load_data(path):
	df = pd.read_csv(path)
	print(f"Number of training rows: {df.shape[0]:,}\n")

	# check class distribution
	print(df["tag"].value_counts(normalize = True))
	df["tag"].replace({"legit": 0, "spam":1}, inplace=True)

	# drop row with nan value
	print("\nDrop {:,} row with null value\n".format(df["body"].isnull().sum()))
	df.dropna(subset=["body"], inplace=True)
	print(f"Number of remain rows: {df.shape[0]:,}\n")

	start = timer()
	print(format("Clean data", '18s'), end='...')

	# clean text and drop row with empty value
	df["body"] = df["body"].apply(clean_content)
	df.replace("", float("NaN"), inplace=True)
	df.dropna(subset = ["body"], inplace=True)

	# drop duplicates
	df.drop_duplicates(subset=['body'],keep='first',inplace=True)

	print(f" Elapsed time: {timer()-start:.3f}")
	print(f"Number of remain rows: {df.shape[0]:,}\n")

	# shuffle dataset row
	df = df.sample(frac = 1).reset_index(drop=True)

	return df

def split_data(body, tag):
	state = 2020
	start = timer()
	print(format("Split data", '18s'), end='...')

	train_text, temp_text, train_labels, temp_labels = train_test_split(
		body, tag,
		random_state = state,
		test_size = 0.3,
		stratify = tag)

	# use temp set to create validation and test set
	val_text, test_text, val_labels, test_labels = train_test_split(
		temp_text, temp_labels,
		random_state = state,
		test_size = 0.5,
		stratify = temp_labels)

	print(f" Elapsed time: {timer()-start:.3f}")

	return train_text, train_labels, val_text, val_labels, test_text, test_labels

def tokens(tokenizer, data, max_len):
	token_ = tokenizer.batch_encode_plus(
		data.tolist(),
		max_length = max_len,
		padding=True,
		truncation=True,
		return_token_type_ids=False
	)
	return token_

def tokenize(train_text, val_text, test_text, max_len):
	print(format("Load Tokenizer", '18s'), end='...\n')
	tokenizer = BertTokenizerFast.from_pretrained(bert_pretrain)

	start = timer()
	print(format("Tokenize", '18s'), end='...')

	# tokenize and encode sequences
	tokens_train = tokens(tokenizer, train_text, max_len)
	tokens_val = tokens(tokenizer, val_text, max_len)
	tokens_test = tokens(tokenizer, test_text, max_len)

	print(f" Elapsed time: {timer()-start:.3f}")

	return tokens_train, tokens_val, tokens_test

def create_loader(tokens, labels, batch_size, istrain):
	seq = torch.tensor(tokens['input_ids'])
	mask = torch.tensor(tokens['attention_mask'])
	label_tensor = torch.tensor(labels.tolist())

	data = TensorDataset(seq, mask, label_tensor)								# wrap tensors
	sampler = RandomSampler(data) if istrain else SequentialSampler(data)		# sampler for sampling the data during training
	dataloader = DataLoader(data, sampler = sampler, batch_size = batch_size)

	return dataloader

def data_loader(tokens_train, train_labels, tokens_val, val_labels, tokens_test, test_labels, batch_size):
	start = timer()
	print(format("Create DataLoader", '18s'), end='...')

	train_dataloader = create_loader(tokens_train, train_labels, batch_size, True)
	val_dataloader = create_loader(tokens_val, val_labels, batch_size, False)
	test_dataloader = create_loader(tokens_test, test_labels, batch_size, False)

	print(f" Elapsed time: {timer()-start:.3f}")
	return train_dataloader, val_dataloader, test_dataloader

def loss_func(device, train_labels):
	start = timer()
	print(format("Loss Function", '18s'), end='...')
	# compute class weight
	class_wts = compute_class_weight('balanced', np.unique(train_labels), train_labels)
	# convert class weight to tensor
	weights= torch.tensor(class_wts, dtype = torch.float)
	weights = weights.to(device)
	# loss function
	cross_entropy  = nn.NLLLoss(weight = weights)
	print(f" Elapsed time: {timer()-start:.3f}")
	return cross_entropy

def init_optimizer(model, learning_rate):
	start = timer()
	print(format("Init Optimizer", '18s'), end='...')
	# define the optimizer
	optimizer = AdamW(model.parameters(), lr = learning_rate)
	print(f" Elapsed time: {timer()-start:.3f}")
	return optimizer

def init_model():
	start = timer()
	print(format("Init Model", '18s'), end='...')
	try:
		model = torch.load(data_path/model_file)
	except:
		bert = BertModel.from_pretrained(bert_pretrain)
		for param in bert.parameters():
			param.requires_grad = False

		# pass the pre-trained BERT to our define architecture
		model = BERT_Arch(bert)
		torch.save(model, data_path/model_file)

	print(f" Elapsed time: {timer()-start:.3f}")
	return model

# function to predict the model
def predict_model(device, model, test_dataloader):
	total_preds = []
	print("\nPredicting...")
	start = timer()
	pre_time = start
	# iterate over batches
	for step,batch in enumerate(test_dataloader):

		# Progress update every 50 batches.
		if step % 50 == 0 and not step == 0:

			# Report progress.
			print(f"  Batch {step:>5,}  of  {len(test_dataloader):>5,}.  Time: {timer()-pre_time:.3f}")
			pre_time = timer()

		# push the batch to gpu
		batch = [t.to(device) for t in batch]

		sent_id, mask, labels = batch
		with torch.no_grad():
			preds = model(sent_id, mask)
			preds = preds.detach().cpu().numpy()
			total_preds.extend(preds)
	print(f"Predict time: {timer()-start:.3f}")

	predictions = np.argmax(total_preds, axis = 1)

	start = timer()
	print(format("Save Predictions", '18s'), end='...')

	np.save(data_path/"saved_pred", predictions)

	print(f" Elapsed time: {timer()-start:.3f}")

	return predictions

# function to train the model
def train_model(device, model, optimizer, cross_entropy, train_dataloader):
	print("\nTraining...")
	model.train()

	total_loss, total_accuracy = 0, 0
	# empty list to save model predictions
	total_preds=[]
	total_len = len(train_dataloader)

	start = timer()
	pre_time = start

	# iterate over batches
	for step, batch in enumerate(train_dataloader):
		# progress update after every 50 batches.
		if step % 50 == 0 and not step == 0:
			print(f"  Batch {step:>5,}  of  {total_len:>5,}.  Timer: {timer()-pre_time:.3f}")
			pre_time = timer()

		# push the batch to gpu
		batch = [r.to(device) for r in batch]

		sent_id, mask, labels = batch

		# clear previously calculated gradients 
		model.zero_grad()        

		# get model predictions for the current batch
		preds = model(sent_id, mask)

		# compute the loss between actual and predicted values
		loss = cross_entropy(preds, labels)

		# add on to the total loss
		total_loss = total_loss + loss.item()

		# backward pass to calculate the gradients
		loss.backward()

		# clip the the gradients to 1.0. It helps in preventing the exploding gradient problem
		nn.utils.clip_grad_norm_(model.parameters(), 1.0)

		# update parameters
		optimizer.step()

		# model predictions are stored on GPU. So, push it to CPU
		preds=preds.detach().cpu().numpy()

		# append the model predictions
		total_preds.append(preds)

	print(f"  Batch {total_len:>5,}  of  {total_len:>5,}.  Timer: {timer()-pre_time:.3f}")

	# compute the training loss of the epoch
	avg_loss = total_loss / total_len

	# predictions are in the form of (no. of batches, size of batch, no. of classes).
	# reshape the predictions in form of (number of samples, no. of classes)
	total_preds  = np.concatenate(total_preds, axis=0)

	print(f"Train time: {timer()-start:.3f}")
	#returns the loss and predictions
	return avg_loss, total_preds

# function to evaluate the model
def evaluate_model(device, model, cross_entropy, val_dataloader):
	print("\nEvaluating...")

	# deactivate dropout layers
	model.eval()

	total_loss, total_accuracy = 0, 0
	# empty list to save the model predictions
	total_preds = []
	total_len = len(val_dataloader)

	start = timer()
	pre_time = start

	# iterate over batches
	for step,batch in enumerate(val_dataloader):
		# Progress update every 50 batches.
		if step % 50 == 0 and not step == 0:
			# Report progress.
			print(f"  Batch {step:>5,}  of  {total_len:>5,}.  Timer: {timer()-pre_time:.3f}")
			pre_time = timer()

		# push the batch to gpu
		batch = [t.to(device) for t in batch]

		sent_id, mask, labels = batch

		# deactivate autograd
		with torch.no_grad():
			# model predictions
			preds = model(sent_id, mask)

			# compute the validation loss between actual and predicted values
			loss = cross_entropy(preds,labels)

			total_loss = total_loss + loss.item()

			preds = preds.detach().cpu().numpy()

			total_preds.append(preds)

	print(f"  Batch {total_len:>5,}  of  {total_len:>5,}.  Timer: {timer()-pre_time:.3f}")

	# compute the validation loss of the epoch
	avg_loss = total_loss / total_len

	# reshape the predictions in form of (number of samples, no. of classes)
	total_preds  = np.concatenate(total_preds, axis=0)

	print(f"Evaluate time: {timer()-start:.3f}")
	return avg_loss, total_preds

def train(device, model, optimizer, cross_entropy, epochs, train_dataloader, val_dataloader):
	best_valid_loss = float('inf')
	# empty lists to store training and validation loss of each epoch
	train_losses=[]
	valid_losses=[]

	#for each epoch
	for epoch in range(epochs):

		print(f"\n Epoch {epoch+1} / {epochs}")
		start = timer()
		#train model
		train_loss, _ = train_model(device, model, optimizer, cross_entropy, train_dataloader)

		#evaluate model
		valid_loss, _ = evaluate_model(device, model, cross_entropy, val_dataloader)

		#save the best model
		if valid_loss < best_valid_loss:
			best_valid_loss = valid_loss
		torch.save(model.state_dict(), data_path/weight_file)

		# append training and validation loss
		train_losses.append(train_loss)
		valid_losses.append(valid_loss)

		print(f"\nTraining Loss: {train_loss:.3f}")
		print(f"Validation Loss: {valid_loss:.3f}")
		print(f"Epoch time: {timer()-start:.3f}")

# save preprocess data for train
def save_data(train_dataloader, val_dataloader, test_dataloader, train_labels, test_labels):
	start = timer()
	print(format("Save DataLoader", '18s'), end='...')

	torch.save(train_dataloader, data_path/train_loader_file)
	torch.save(val_dataloader, data_path/val_loader_file)
	torch.save(test_dataloader, data_path/test_loader_file)
	
	test_labels.to_pickle(data_path/test_label_file)
	train_labels.to_pickle(data_path/train_label_file)

	print(f" Elapsed time: {timer()-start:.3f}")

def preprocess(path, max_len, batch_size):
	df = load_data(path)
	train_text, train_labels, val_text, val_labels, test_text, test_labels = split_data(df['body'], df['tag'])
	tokens_train, tokens_val, tokens_test = tokenize(train_text, val_text, test_text, max_len)
	train_dataloader, val_dataloader, test_dataloader = data_loader(tokens_train, train_labels, tokens_val, val_labels, tokens_test, test_labels, batch_size)
	save_data(train_dataloader, val_dataloader, test_dataloader, train_labels, test_labels)
