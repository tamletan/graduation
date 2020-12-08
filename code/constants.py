from pathlib import Path
data_path = Path("./data")
data_path.mkdir(parents=True, exist_ok=True)

test_loader_file = 'test_loader.pt'
val_loader_file = 'val_loader.pt'
train_loader_file = 'train_loader.pt'
train_label_file = 'train_label.pickle'
test_label_file = 'test_label.pickle'
weight_file = 'weights.pt'
model_file = 'model.pt'

bert_pretrain = 'bert-base-uncased'