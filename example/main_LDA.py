from ConfirmatoryLDA import CLDA
import pickle
from sklearn.feature_extraction.text import CountVectorizer

# load data
dir_ = '/Users/shinbo/Desktop/metting/LDA/paper/experiments/car/prepare/preproc.pkl'
data = pickle.load(open(dir_, 'rb'))
stop_words = ['the','and','to','it','be','have','in','for','of','this','that']
for i in range(len(data)):
    data[i] = [w for w in data[i] if w not in stop_words]
# make cv object to pass on LDA model
data_join = [' '.join(doc) for doc in data]
cv = CountVectorizer()
X = cv.fit_transform(data_join).toarray()

# directory to save model
save_dir = '/Users/shinbo/Desktop/metting/LDA/paper/experiments/car/model/DMM_result.pkl'

# Hyperparameters for ordinary lda
K = 4
alpha = 50/K
eta = 0.01
maxIter = 500
maxIterDoc =100
threshold = 0.01
random_state = 42

# training
lda = CLDA.CLDA_VI(alpha=alpha,eta=eta,K=K)
lda.train(X, cv, maxIter, maxIterDoc, threshold, random_state)

pickle.dump(lda, open(save_dir, 'wb'))