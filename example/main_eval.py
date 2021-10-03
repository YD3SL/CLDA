from ConfirmatoryLDA.evaluation import WTC, BTC, TOP_N_WORDS_df

# directory of word2vec model
# /Users/shinbo/Desktop/metting/LDA/paper/word_embedding/GoogleNews-vectors-negative300.bin/GoogleNews-vectors-negative300.bin

# load data
dir_ = '/Users/shinbo/Desktop/metting/LDA/paper/experiments/car/prepare/preproc.pkl'
data = pickle.load(open(dir_, 'rb'))
stop_words = ['the','and','to','it','be','have','in','for','of','this','that']
for i in range(len(data)):
    data[i] = [w for w in data[i] if w not in stop_words]
# make cv object used to train LDA model
data_join = [' '.join(doc) for doc in data]
cv = CountVectorizer()
X = cv.fit_transform(data_join).toarray()

# load LDA model
dir_model = '/Users/shinbo/Desktop/metting/LDA/paper/experiments/hotel/model/CDMM_result.pkl'
DMM_CLDA = pickle.load(open(dir_model,'rb'))
DMM_CLDA_lam =  [DMM_CLDA.components_[k,:] for k in range(4)]
TOP_N_WORDS = 100
DMM_CLDA_Top_words = TOP_N_WORDS_df(DMM_CLDA, TOP_N_WORDS, ['price','food','drink','service'])

# WTC
print(WTC(DMM_CLDA_Top_words,N))

# BTC
print(BTC(DMM_CLDA_Top_words,N))