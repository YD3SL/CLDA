import gensim
from collections import Counter
from itertools import combinations
from scipy.spatial.distance import cosine

w2v_dir = input("Please enter directory of word2vec model \n")
word2vec_model = gensim.models.keyedvectors.KeyedVectors.load_word2vec_format(
    w2v_dir, binary=True
)

def WTC(df,N):
    """Within Topic Coherence Measure.

        [Note]
        It ignores a word which does not have trained word vector.

        Parameters
        ----------
        df : Word-Topic distribution K by V
            where K is number of topics and V is number of words

        N : Number of top N words 

        Returns
        -------
        total : WTC value of each topic (1 * K)

    """
    df = df.iloc[:N,:]
    total = []
    for col in df.columns:
        cos_val = 0
        words = df[col].tolist()
        for c in combinations(words,2):
#             print(c)
            try:
                cos_val += 1-cosine(word2vec_model.get_vector(c[0]),
                               word2vec_model.get_vector(c[1]))
            except:
                pass
#             print(c)
#             print(cosine(word2glove[c[0]], word2glove[c[1]]))
        print(col, cos_val)
        total.append(cos_val)
    return total

def BTC(df,N):
    """Between Topic Coherence Measure.

        [Note]
        It ignores a word which does not have trained word vector.

        Parameters
        ----------
        df : Word-Topic distribution K by V
            where K is number of topics and V is number of words

        N : Number of top N words 

        Returns
        -------
        btc : BTC value of each topic (1 * K)

    """
    df = df.iloc[:N,:]
    cols = list(df.columns)
    result = {}
    for col in cols:
        result[col] = {}
        rest_cols = list(set(cols) - set([col]))
        for word in df[col].tolist():
            # cosine sim within topic
            cos_val_intrinsic = 0
            for intra_word in list( set(df[col].tolist()) - set([word]) ):
                try:
                    cos_val_intrinsic += 1-cosine(word2vec_model.get_vector(word),
                           word2vec_model.get_vector(intra_word))
                except:
                    pass
            # cosin sim between topic
            max_cos_val_between = 1e-10
            for cross_col in rest_cols:
                cos_val_between = 0
                for cross_word in df[cross_col].tolist():
                    try:
                        cos_val_between += 1-cosine(word2vec_model.get_vector(word),
                               word2vec_model.get_vector(cross_word))
                    except:
                        pass
                if max_cos_val_between < cos_val_between:
                    max_cos_val_between = cos_val_between
            # compare within topic vs between topic
            if cos_val_intrinsic < max_cos_val_between:
                result[col][word] = 0
            else:
                result[col][word] = 1
    val = 0
    btc = []
#     print(result)
    for topic in result.keys():
        btc.append(sum(result[topic]))
        for word in result[topic].keys():
            val += result[topic][word]
#     print(val)
    return btc

def TOP_N_WORDS_df(MODEL, TOP_N_WORDS, col_names, cv):
    """TOP N words from word-topic distribution

        Parameters
        ----------
        MODEL : LDA or CLDA model

        TOP_N_WORDS : Number of top N words 

        col_names : names of each topic

        cv: CountVectorizer object used in LDA model

        Returns
        -------
        result : N * K dataframe which shows top N words of each topic

    """
    result = pd.DataFrame()
    for i in range(4):
        temp = pd.DataFrame({'words':cv.get_feature_names(), 'lambda':MODEL.components_[i,:]})
        temp = temp.sort_values(by='lambda', ascending=False).iloc[:TOP_N_WORDS,:]
        result[i] = temp['words'].tolist()
    result.columns = col_names
    return result