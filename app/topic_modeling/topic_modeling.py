# coding: utf-8

"""
Topic modeling using LDA (Latent Dirichlet Allocation)
"""

#verbose         = 'yes'
verbose   = 'no'
multicore = 'yes'

import logging
import numpy as np
import pandas as pd
import codecs
from gensim import corpora, models
from sklearn.feature_extraction.text import CountVectorizer
from scipy.stats import chisquare

PROGRESS_NUM = 25
logging.addLevelName(PROGRESS_NUM, 'PROGRESS')


MS_topic_labels = \
{ 0:'Other reason for chat'\
, 1:'Dropped chat'\
, 2:'Promotion-Loyalty code'\
, 3:'Sparks'\
, 4:'Live Chat'\
, 5:'Placing an order'\
, 6:'Amend My Order - Account'\
, 7:'Delivery Query'\
, 8:'Food'\
, 9:'Password reset'\
,10:'Policy Queries'\
,11:'GM Availability'\
,12:'Store Service and feedback'\
,13:'Dropped Call'\
,14:'GM Quality'\
,15:'Returns  Refunds'\
,16:'Transfer'\
,17:'Website Comments  Feedback'}


class LDATopicClassifier(object):
    def __init__(self,folder = '.', model_filename = \
    'LDAmodel__num_topics30__passes5__min_df5__max_df0.95__max_featuresNone__corpus_id05_fastTextConv.pickle'):

        self.model = load_LDA_model(folder+'/'+ model_filename)
        self.lda_params = self.model['lda_tf'].get_params()
        self.vocabulary = self.model['tf_vectorizer'].vocabulary_
        self.tf_vectorizer = CountVectorizer(vocabulary=self.vocabulary)
        self.n_MS_topics   = len(MS_topic_labels.keys())

        #====================================================================================#
        # LDA-MS topic correspondence

        # by default all LDA topics are assigned to 'Other reason for chat'
        self.MS_from_LDA_topic_id = [0 for i in range(self.lda_params['n_topics'])]

        # by-eye identified LDA topics
        if model_filename == 'LDAmodel__num_topics30__passes5__min_df5__max_df0.95__max_featuresNone.pickle':

            #Topic #0:
            self.MS_from_LDA_topic_id[0]=2
            #order param_number param_price param_name thanks thank check allow discount place number refund account patience help ok couple sorry working yes
            #Topic #1:
            self.MS_from_LDA_topic_id[1]=8
            #chicken louise fresh drink flower bouquet plant title dead water condition tie pot british pie square supply aberdeen plate weight
            #Topic #2:
            #param_number param_name number contact help sorry param_ms request happy team th information hi unable collection phone furniture order customer check
            #Topic #3:
            #param_name param_ms customer just service time like sorry know really make pass team said want store told sure good hear
            #Topic #4:
            self.MS_from_LDA_topic_id[4]=14
            #trousers suit look bought quality fault grey waste broken worn main best washed jumper coat local ago car match seeing
            #Topic #5:
            self.MS_from_LDA_topic_id[5]=15
            #gift card send address param_number param_price days bought date param_email refund great friend param_name just number thank param_smileyface goodwill working
            #Topic #6:
            self.MS_from_LDA_topic_id[6]=12
            #store receipt purchase sorry able bought visit send did param_price unable till hi unfortunately help yes request went hear need
            #Topic #7:
            #param_name chat param_ms thank thanks enjoy rest help day today web welcome contact hi ok param_number want need continue end
            #Topic #8:
            #st event navy street year hospital cancer oxford raffle wasnt avenue prize york area world raise important fully breast donate
            #Topic #9:
            self.MS_from_LDA_topic_id[9]=8
            #birthday cake support hot party chocolate chosen penny group champagne executive trust bikini choosing child butter fast boy birth sponge
            #Topic #10:
            #understand team inconvenience sorry time issue relevant accept really disappointment department concern consideration certainly param_name left concerned future disappointing point
            #Topic #11:
            #staff dress pair jeans set member wear faulty remove buy sound price removed look linen pack tights reduced floral suitable
            #Topic #12:
            self.MS_from_LDA_topic_id[12]=2
            #card gift enter reward credit param_ms pay page use number voucher payment param_number help debit param_name code param_price order digit
            #Topic #13:
            self.MS_from_LDA_topic_id[13]=6
            #account address param_email param_name card thanks new password registered sparks check number help allow thank param_number reset yes link just
            #Topic #14:
            #param_name just thanks thank param_smileyface help know hi problem great oh yes ok look good like sure really sorry let
            #Topic #15:
            #knickers accordingly complaint want stop clothes option param_ms query marketing shorts comes direct assist appropriate response reply send choice clothing
            #Topic #16:
            self.MS_from_LDA_topic_id[16]=2
            #code voucher use discount used offer applied bag promotion sorry trying work valid page conjunction accepted sale tried clothing promotional
            #Topic #17:
            self.MS_from_LDA_topic_id[17]=5
            #address param_number param_name sent number confirm thank param_email thanks received order send param_postcode replacement check mail yes post help confirmation
            #Topic #18:
            self.MS_from_LDA_topic_id[18]=7
            #order delivery param_number param_name number date collect parcel check allow today place sorry tomorrow day home ordered thanks couple thank
            #Topic #19:
            self.MS_from_LDA_topic_id[19]=4
            #soon thanks patience adviser hi chat live param_ms help welcome today param_name shortly fruit hello queue available want need continue
            #Topic #20:
            self.MS_from_LDA_topic_id[20]=11
            #param_number stock size product check item available code param_name sorry store thanks looking colour days shirt hi allow ok black
            #Topic #21:
            #school bath employee salmon kay daily share leicester technology recall accurate vicki train cheltenham mat height fosse dessert luckily accident
            #Topic #22:
            self.MS_from_LDA_topic_id[22]=2
            #bag free add code gift product box item added param_number enter order price need basket total search automatically shopping taken
            #Topic #23:
            self.MS_from_LDA_topic_id[23]=17
            #click lingerie message try site link error screen trying fitting clear book browser assistant dine stuff appointment guide priced july
            #Topic #24:
            self.MS_from_LDA_topic_id[24]=8
            #food product sorry team code like hear thanks use pass feedback hi param_email bought quality touch really param_number look address
            #Topic #25:
            #half survey body hey machine elderly parking tables purse market listening variety weekly ticket major causeway buttons nest speed introduce
            #Topic #26:
            self.MS_from_LDA_topic_id[26]=3
            #sparks card offer account param_name added discount use add activate automatically need yes registered page param_number just coffee param_ms shop
            #Topic #27:
            #international flat loose kirsty turn result vegetarian angry doubt physical mess attachment wed break maximum business gosh blend research constantly
            #Topic #28:
            #single carole man multiple retrieve scratch cheque panel preference silver useless eu woman menu progress care ur magazine spanish pity
            #Topic #29:
            self.MS_from_LDA_topic_id[29]=15
            #return item refund returned store bra exchange wine days post form need order parcel nearest note received print ordered postage

        elif model_filename == 'LDAmodel__num_topics30__passes5__min_df5__max_df0.95__max_featuresNone__corpus_id05_fastTextConv.pickle':

            #Topic #0:
            self.MS_from_LDA_topic_id[0]=12
            #store collect collected pick visit need food able simply nearest
            #Topic #1:
            self.MS_from_LDA_topic_id[1]=6
            #correct mail guest incorrect amend wrong locked resend order confirmation
            #Topic #2:
            self.MS_from_LDA_topic_id[2]=5
            #param_price order discount applied place account apply bag param_number check
            #Topic #3:
            self.MS_from_LDA_topic_id[3]=11
            #stock check size product param_name item available thanks help sorry
            #Topic #4:
            self.MS_from_LDA_topic_id[4]=4
            #chat contact help param_ms param_name voucher hi want continue need
            #Topic #5:
            self.MS_from_LDA_topic_id[5]=2
            #offer sparks param_number discount param_name use lingerie clothing card used
            #Topic #6:
            #param_name param_smileyface thank thanks send great address sorry really problem
            #Topic #7:
            self.MS_from_LDA_topic_id[7]=6
            #address param_email param_name account thanks confirm number param_postcode sent check
            #Topic #8:
            self.MS_from_LDA_topic_id[8]=3
            #sparks account card param_number param_name registered password new thanks number
            #Topic #9:
            self.MS_from_LDA_topic_id[9]=7
            #delivery order date parcel tomorrow place home day deliver param_name
            #Topic #10:
            #member chicken support hot bath midnight chosen meat trust dine
            #Topic #11:
            #birthday send unfortunate friend sent copy mother upset mum daughter
            #Topic #12:
            self.MS_from_LDA_topic_id[12]=8
            #cake wedding fruit survey chocolate possibly group scratch panel silver
            #Topic #13:
            #said assistant share till half queue went particular non lady
            #Topic #14:
            self.MS_from_LDA_topic_id[14]=10
            #store purchase bought param_ms colleague request able policy best goods
            #Topic #15:
            self.MS_from_LDA_topic_id[15]=2
            #card gift param_price param_number number use payment pay arrive junk
            #Topic #16:
            #soon thanks patience adviser hi chat live param_name help param_ms
            #Topic #17:
            self.MS_from_LDA_topic_id[17]=2
            #code used promotion param_number sorry work bar type promotional accepted
            #Topic #18:
            self.MS_from_LDA_topic_id[18]=2
            #code bag add enter free reward gift page box param_number
            #Topic #19:
            self.MS_from_LDA_topic_id[19]=8
            #food param_number team sorry contact param_name param_ms information date touch
            #Topic #20:
            #param_number param_name days school colour th check customer param_postcode week
            #Topic #21:
            self.MS_from_LDA_topic_id[21]=15
            #item return refund returned receipt post parcel need form days
            #Topic #22:
            self.MS_from_LDA_topic_id[22]=14
            #trousers furniture pair fault suit look ordered main leg length
            #Topic #23:
            #param_name understand team sorry time really param_ms pass inconvenience issue
            #Topic #24:
            self.MS_from_LDA_topic_id[24]=12
            #bra site family message fitting interested book appointment employee men
            #Topic #25:
            self.MS_from_LDA_topic_id[25]=15
            #order param_number param_name number thank thanks check allow help refund
            #Topic #26:
            self.MS_from_LDA_topic_id[26]=12
            #store staff customer service manager like param_name range help retail
            #Topic #27:
            #good morning param_name post line look local param_smileyface try bear
            #Topic #28:
            #thank param_name thanks param_ms today welcome web chat day enjoy
            #Topic #29:
            #event opening year able marble arch smell beef local body
            pass
        else:
            raise ValueError('Unknown model_filename '+model_filename)
    #====================================================================================#


    def getLDATopicDistri(self, text):

        chat_tf = self.tf_vectorizer.fit_transform(text)
        #print chat_tf
        lda_topic_distribution = self.model['lda_tf'].transform(chat_tf)[0]

        total = sum(lda_topic_distribution)
        return lda_topic_distribution/total


    def getMSTopicDistri(self, text):

        lda_topic_distribution = self.getLDATopicDistri(text)
        n_lda_topics = len(lda_topic_distribution)

        # aggregation
        MS_topic_distribution = np.zeros(len(MS_topic_labels.keys()))

        for i in range(self.lda_params['n_topics']):
            MS_topic_id = self.MS_from_LDA_topic_id[i]
            MS_topic_distribution[MS_topic_id] += lda_topic_distribution[i]

        return MS_topic_distribution

    def get_predominant_MStopic_id(self,text):
        text_MS_topic_distribution = self.getMSTopicDistri(text)
        return np.argmax(text_MS_topic_distribution)


    #def get_predominant_MStopic_id(self,text):
        #text_MS_topic_distribution = self.getMSTopicDistri(text)
        #homogeneous_distri = np.ones(self.n_MS_topics)/self.n_MS_topics

        ##print homogeneous_distri
        ##print text_MS_topic_distribution
        ##print chisquare(homogeneous_distri, homogeneous_distri)
        ##print chisquare(text_MS_topic_distribution, homogeneous_distri)

        #chi_stat = chisquare(homogeneous_distri,text_MS_topic_distribution)[1]
        #if chi_stat < .05:
            #return np.argmax(text_MS_topic_distribution)
        #else:
            #return -1

    def get_predominant_MStopic_label(self,text):
        predominant_MStopic_id = self.get_predominant_MStopic_id(text)
        if predominant_MStopic_id == -1:
            return 'None'
        else:
            return MS_topic_labels[predominant_MStopic_id]



    def getLDATopicHistogram(self, text, axis):
        import numpy as np
        lda_topic_distribution = self.getLDATopicDistri(text)

        #norm = sum(lda_topic_distribution)
        axis.barh(range(len(lda_topic_distribution)),lda_topic_distribution)
        #axis.invert_yaxis()
        axis.set_xlabel('Topic probability')
        axis.set_ylabel('LDA Topic')
        axis.set_yticks(np.array(range(len(lda_topic_distribution)))+0.5)
        axis.set_yticklabels(range(len(lda_topic_distribution)))


    def getMSTopicHistogram(self, text, axis):
        import numpy as np
        ms_topic_distribution = self.getMSTopicDistri(text)
        labels = self.get_ms_topic_labels()
        #data = {labels[i]: val for i, val in enumerate(ms_topic_distribution)}
        #print(data)
        #df = pd.DataFrame(data = data, x =
        #                   index = range(len(ms_topic_distribution)))
        #df.plot(kind='barh', ax=axis, legend = False)
        axis.barh(range(len(ms_topic_distribution)), ms_topic_distribution)
        #axis.invert_yaxis()
        axis.set_xlabel('Topic probability')
        axis.set_ylabel('M&S Topic')
        axis.set_yticks(np.array(range(len(ms_topic_distribution)))+0.5)
        axis.set_yticklabels(labels)

    def print_topic_top_words(self, n_top_words=10):
        feature_names = self.model['tf_vectorizer'].get_feature_names()
        for topic_idx, topic in enumerate(self.model['lda_tf'].components_):
            print("Topic #%d:" % topic_idx)
            print(" ".join([feature_names[i]
                            for i in topic.argsort()[:-n_top_words - 1:-1]]))

    def show_pyLDAvis_plots(self):
        from pyLDAvis.sklearn import prepare
        prepare(self.model['lda_tf'], self.model['dtm_tf'], self.model['tf_vectorizer'])


    def get_ms_topic_labels(self):
        return MS_topic_labels.values()




def fit_LDA_sklearn(file_path,num_topics=30,passes=5\
,          min_df=5, max_df=0.95,max_features=None,corpus_id='Default'):
    """
    train and return LDA model

    Parameters:
    file_path  : path to the text file containing tfidf-filtered tokenized chats
                  one chat per line, tokens separated by whitespace
    num_topics : # of topics
    passes     : # of passes through the corpus during training
    min_df     : lower limit of DF range considered
    max_df     : upper limit of DF range considered
    """

    #====================================================================================#
    # Configure messages sent to the terminal
    if verbose=='yes': level=logging.INFO
    else:              level=PROGRESS_NUM
    logging.basicConfig(format='%(levelname)s : %(message)s', level=level)
    #====================================================================================#

    chats = text_stream(file_path)
    #chats = text_file(file_path)
    #import codecs
    #chats = codecs.open(file_path,'r','utf-8')

    #====================================================================================#
    logging.log(PROGRESS_NUM,'Convert to document-term matrix')
    from sklearn.feature_extraction.text import CountVectorizer
    tf_vectorizer = CountVectorizer(min_df=min_df, max_df=max_df\
    ,max_features=max_features,stop_words='english')
    dtm_tf = tf_vectorizer.fit_transform(chats)
    print(str(len(tf_vectorizer.vocabulary_))+' features')
    logging.info(str(len(tf_vectorizer.vocabulary_))+' features')
    #====================================================================================#

    #====================================================================================#
    logging.log(PROGRESS_NUM,'Training LDA')
    from sklearn.decomposition import LatentDirichletAllocation
    lda_tf = LatentDirichletAllocation(n_topics=num_topics,
    max_iter=passes, n_jobs=-1,learning_method='online', random_state=0)
    lda_tf.fit(dtm_tf)
    #====================================================================================#

    #====================================================================================#
    logging.log(PROGRESS_NUM,'Save LDA model data')
    id_str = 'num_topics'+str(num_topics)\
    +          '__passes'+str(passes)\
    +          '__min_df'+str(min_df)\
    +          '__max_df'+str(max_df)\
    +          '__max_features'+str(max_features)\
    +          '__corpus_id' + corpus_id
    save_LDA_model(lda_tf, dtm_tf, tf_vectorizer, 'LDAmodel__' + id_str +'.pickle')
    save_LDA_visualization(lda_tf, dtm_tf, tf_vectorizer, 'LDAvis__' + id_str +'.html')
    #====================================================================================#

def save_LDA_model(lda_tf, dtm_tf, tf_vectorizer, model_file):
    model = {'lda_tf':lda_tf,'dtm_tf':dtm_tf,'tf_vectorizer':tf_vectorizer}
    import pickle
    with open(model_file, 'wb') as handle: pickle.dump(model, handle)

def load_LDA_model(model_file):
    import pickle
    with open(model_file,'rb') as handle: model = pickle.load(handle)
    return model

def save_LDA_visualization(lda_tf, dtm_tf, tf_vectorizer, html_file):
    """
    Save LDA visualization as html
    """
    from pyLDAvis.sklearn import prepare
    data = prepare(lda_tf, dtm_tf, tf_vectorizer)
    from pyLDAvis import save_html
    save_html(data,html_file)



def fit_LDA_gensim(file_path,num_topics=10,passes=1,chunksize=2000):
    """
    train and return LDA model

    Parameters:
    file_path   : path to the text file containing tfidf-filtered tokenized chats
                  one chat per line, tokens separated by whitespace
    num_topics  :
    update_every:
    passes      :
    chunksize   :
    """

    #====================================================================================#
    # Configure messages sent to the terminal
    if verbose=='yes': level=logging.INFO
    else:              level=PROGRESS_NUM
    logging.basicConfig(format='%(levelname)s : %(message)s', level=level)
    #====================================================================================#


    #====================================================================================#
    logging.log(PROGRESS_NUM,'create a Gensim dictionary from the texts')

    dictionary = corpora.Dictionary(\
    line.split() for line in codecs.open(file_path,'r','utf-8'))
    #====================================================================================#


    #====================================================================================#
    logging.log(PROGRESS_NUM,'convert chats to a bag of words corpus')
    chats = text_stream(file_path)
    # creates corpus object without loading the whole document in RAM
    corpus = corpus_stream(file_path,dictionary)
    ## creates corpus object loading the whole document in RAM
    #corpus = [dictionary.doc2bow(text.split()) for text in chats]
    #====================================================================================#


    #====================================================================================#
    logging.log(PROGRESS_NUM,'Training LDA')

    if multicore == 'yes':
        lda = models.LdaMulticore(corpus, id2word=dictionary,\
        num_topics=num_topics, passes=passes,chunksize=chunksize)
    else:
        lda = models.LdaModel(corpus, id2word=dictionary, \
        num_topics=num_topics, passes=passes,chunksize=chunksize)


    lda.show_topics()
    if verbose=='yes': lda.print_topics(num_topics)
    #====================================================================================#

    #====================================================================================#
    # creates corpus object loading the whole document in RAM
    # needed to plot with pyLDAvis
    corpus = [dictionary.doc2bow(text.strip().split()) for text in chats]
    #====================================================================================#

    return lda, corpus, dictionary


class corpus_stream(object):
    """
    creates corpus object without loading the whole document in RAM
    #corpus = [dictionary.doc2bow(text.split()) for text in chats]
    """
    def __init__(self,file_path,in_dict):
        self.dictionary = in_dict
        self.file_path  = file_path

    def __iter__(self):
         for line in open(self.file_path):
             # assume there's one document per line, tokens separated by whitespace
             yield self.dictionary.doc2bow(line.strip().split())

class text_stream(object):
    """
    Allows reading a text file without loading it in RAM
    """
    def __init__(self,file_path):
        self.file_path  = file_path

    def __iter__(self):
        import codecs
        for line in codecs.open(self.file_path,'r','utf-8'):
            yield line

def text_file(file_path):
    import codecs
    return codecs.open(file_path,'r','utf-8')

def barchart(data, labels, axis):
    pos = np.arange(len(data)) + 0.5  # the bar centers on the y axis
    axis.barh(pos, data.sort_index(), align='center', height=0.25)
    axis.yticks(pos, labels.sort_index())
