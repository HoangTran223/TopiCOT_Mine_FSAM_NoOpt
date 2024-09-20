from topmost.data import download_20ng
from topmost.data import download_dataset
from topmost.preprocessing import Preprocessing

# download raw data
# download_20ng.download_save(output_dir='./datasets/20NG')

preprocessing = Preprocessing(vocab_size=5000, stopwords='.\data\stopwords\snowball_stopwords.txt')

rst = preprocessing.preprocess_jsonlist(dataset_dir='./datasets/20NG', label_name="group")

preprocessing.save('./datasets/20NG', **rst)
