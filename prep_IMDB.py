from topmost.data import download_IMDB
from topmost.data import download_dataset
from topmost.preprocessing import Preprocessing

# download raw data
download_IMDB.IMDB(root='./datasets/IMDB/raw', download=True)

preprocessing = Preprocessing(vocab_size=5000, stopwords='.\data\stopwords\snowball_stopwords.txt')

rst = preprocessing.preprocess_jsonlist(dataset_dir='./datasets/IMDB/raw', label_name="sentiment")

preprocessing.save('./datasets/IMDB', **rst)