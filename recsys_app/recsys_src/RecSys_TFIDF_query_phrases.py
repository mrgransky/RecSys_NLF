import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import Colormap as cm
#import spacy
from colorama import Back, Fore, Style
import seaborn as sns

import matplotlib
matplotlib.use("Agg")

from utils import *
from collections import Counter

from scipy.sparse import csr_matrix, coo_matrix
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer

import nltk
nltk_modules = ['punkt', 
               'averaged_perceptron_tagger', 
               'stopwords',
               'wordnet',
							 'omw-1.4',
							 ]
nltk.download('all',
              quiet=True, 
              raise_on_error=True,
              )

# Adapt stop words
#STOPWORDS = nltk.corpus.stopwords.words('english')
#print(nltk.corpus.stopwords.words('finnish'))

STOPWORDS = nltk.corpus.stopwords.words(nltk.corpus.stopwords.fileids())
my_custom_stopwords = ['btw', "could've", "n't","'s","—", "i'm", "'m", 
												"i've", "ive", "'d", "i'd", " i'll", "'ll", "'ll", "'re", "'ve", 
												'aldiz', 'baizik', 'bukatzeko', 
												'edota', 'eze', 'ezpabere', 'ezpada', 'ezperen', 'gainera', 
												'gainerontzean', 'guztiz', 'hainbestez', 'horra', 'onların', 'ordea', 
												'osterantzean', 'sha', 'δ', 'δι', 'агар-чи', 'аз-баски', 'афташ', 'бале', 
												'баҳри', 'болои', 'валекин', 'вақте', 'вуҷуди', 'гар', 'гарчанде', 'даме', 'карда', 
												'кошки', 'куя', 'кӣ', 'магар', 'майлаш', 'модоме', 'нияти', 'онан', 'оре', 'рӯи', 
												'сар', 'тразе', 'хом', 'хуб', 'чаро', 'чи', 'чунон', 'ш', 'шарте', 'қадар', 
												'ҳай-ҳай', 'ҳамин', 'ҳатто', 'ҳо', 'ҳой-ҳой', 'ҳол', 'ҳолате', 'ӯим', 'באיזו', 'בו', 'במקום', 
												'בשעה', 'הסיבה', 'לאיזו', 'למקום', 'מאיזו', 'מידה', 'מקום', 'סיבה', 'ש', 'שבגללה', 'שבו', 'תכלית', 'أفعل', 
												'أفعله', 'انفك', 'برح', 'سيما', 'कम', 'से', 'ἀλλ', '’',
												]
STOPWORDS.extend(my_custom_stopwords)
#print(len(STOPWORDS), STOPWORDS)
UNIQUE_STOPWORDS = set(STOPWORDS)
#print(len(UNIQUE_STOPWORDS), UNIQUE_STOPWORDS)

parser = argparse.ArgumentParser(description='National Library of Finland (NLF) RecSys')
parser.add_argument('--inputDF', default=f"{os.path.expanduser('~')}/Datasets/Nationalbiblioteket/dataframes/nikeY.docworks.lib.helsinki.fi_access_log.07_02_2021.log.dump", type=str) # smallest
parser.add_argument('--qusr', default="ip69", type=str)
parser.add_argument('--qtip', default="Kristiinan Sanomat_77 A_1", type=str) # smallest
parser.add_argument('--qphrase', default="ystävä", type=str) # smallest

args = parser.parse_args()

# how to run:
# python RecSys_TFIDF.py --inputDF ~/Datasets/Nationalbiblioteket/dataframes/nikeY.docworks.lib.helsinki.fi_access_log.07_02_2021.log.dump

def clean_text(text):
	text = text.lower().strip()
	
	text = re.sub(r"\r\n", "", text)

	#text = re.sub(r"(\n)\1{2,}", " ", text).strip()

	#text = "".join(i for i in text if ord(i)<128)
	#text = re.sub("[^A-Za-z0-9 ]+", "", text) # does not work with äöå...
	#text = re.sub("[^A-ZÜÖÄa-z0-9 ]+", "", text) # äöüÄÖÜß
	text = re.sub(r"\W+|_"," ", text) # replace special characters with space
	text = re.sub("\s+", " ", text)

	return text

def my_tokenizer(sentence, stopwords=UNIQUE_STOPWORDS, min_words=4, max_words=200, ):	
	wnl = nltk.stem.WordNetLemmatizer()

	sentences = sentence.lower()
	sentences = re.sub(r'[~|^|*][\d]+', '', sentences)

	# works not good: 
	#tokens = [w for w in nltk.tokenize.word_tokenize(sentences)]
	#filtered_tokens = [w for w in tokens if ( w not in stopwords and w not in string.punctuation )]

	tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')	
	tokens = tokenizer.tokenize(sentences)
	filtered_tokens = [w for w in tokens if not w in UNIQUE_STOPWORDS]

	lematized_tokens = [wnl.lemmatize(i,j[0].lower()) if j[0].lower() in ['a','n','v'] else wnl.lemmatize(i) for i,j in nltk.pos_tag(filtered_tokens)]

	return lematized_tokens    

def extract_best_indices(m, topk, mask=None):
	"""
	Use sum of the cosine distance over all tokens.
	m (np.array): cos matrix of shape (nb_in_tokens, nb_dict_tokens)
	topk (int): number of indices to return (from high to lowest in order)
	"""
	# return the sum on all tokens of cosinus for each sentence
	if len(m.shape) > 1:
		cos_sim = np.mean(m, axis=0) 
	else: 
		cos_sim = m
	index = np.argsort(cos_sim)[::-1] # from highest idx to smallest score 
	if mask is not None:
		assert mask.shape == m.shape
		mask = mask[index]
	else:
		mask = np.ones(len(cos_sim))
	mask = np.logical_or(cos_sim[index] != 0, mask) #eliminate 0 cosine distance
	best_index = index[mask][:topk]  
	return best_index

def get_TFIDF_RecSys(dframe, qu_phrase="kirjasto", user_name=args.qusr, nwp_title_issue_page_name=args.qtip, topN=5):
	print(f"{'RecSys (TFIDF)'.center(80, '-')}")
	#print(list(dframe["nwp_content_results"][4].keys()))
	#print(json.dumps(dframe["nwp_content_results"][4], indent=2, ensure_ascii=False))

	print(f">> Cleaning df: {dframe.shape} with {dframe['search_query_phrase'].isna().sum()} NaN >> search_query_phrase << rows..")
	dframe = dframe.dropna(subset=["search_query_phrase"], how='all',).reset_index(drop=True)
	print(f">> Cleaned df: {dframe.shape}")
	dframe['search_query_phrase'] = [','.join(map(str, elem)) for elem in dframe['search_query_phrase']]
	
	#print(f">> Getting list of all OCR content...")
	fst_lst = [d for d in  dframe.loc[:, "search_query_phrase"].values.flatten().tolist()]

	print(fst_lst)

	fprefix = "_".join(args.inputDF.split("/")[-1].split(".")[:-2]) # nikeY_docworks_lib_helsinki_fi_access_log_07_02_2021
	tfidf_vec_fpath = os.path.join(dfs_path, f"{fprefix}_tfidf_vectorizer_qu_phrases.lz4")
	tfidf_rf_matrix_fpath = os.path.join(dfs_path, f"{fprefix}_tfidf_matrix_RF_qu_phrases.lz4")

	if not os.path.exists(tfidf_rf_matrix_fpath):
		print(f"TFIDF for {len(fst_lst)} query phrases, might take a while...".center(110, " "))
		st_t = time.time()

		# Fit TFIDF # not time consuming...
		tfidf_vec = TfidfVectorizer(#min_df=5,
															#ngram_range=(1, 2),
															tokenizer=my_tokenizer,
															stop_words=UNIQUE_STOPWORDS,
															)

		tfidf_matrix_rf = tfidf_vec.fit_transform(raw_documents=fst_lst)
		#tfidf_matrix_rf = np.random.choice(10_000, 10_000)

		save_tfidf_vec(tfidf_vec, fname=tfidf_vec_fpath)
		save_tfidf_matrix(tfidf_matrix_rf, fname=tfidf_rf_matrix_fpath)

		print(f"\t\tElapsed_t: {time.time()-st_t:.2f} s")
	else:
		tfidf_vec = load_tfidf_vec(fpath=tfidf_vec_fpath)
		tfidf_matrix_rf = load_tfidf_matrix(fpath=tfidf_rf_matrix_fpath)
	#return

	feat_names = tfidf_vec.get_feature_names_out()
	print(f"1st 100 features:\n{feat_names[:60]}\n")
	
	vocabs = tfidf_vec.vocabulary_
	print(len(feat_names), len(vocabs))
	#print(json.dumps(vocabs, indent=2, ensure_ascii=False))
	with open(os.path.join(dfs_path, f"{fprefix}_vocabs_qu_phrase.json"), "w") as fw:
		json.dump(vocabs, fw, indent=4, ensure_ascii=False)
	
	# Embed qu_phrase
	qu_tokens = my_tokenizer(qu_phrase)
	print(f">> tokenize query word: '{qu_phrase}'\t {len(qu_tokens)} {qu_tokens}")
	tfidf_matrix_qu = tfidf_vec.transform(qu_tokens)

	print(f"REFERENCE_TFIDF_MATRIX: {tfidf_matrix_rf.shape}\tQUERY_TFIDF_MATRIX: {tfidf_matrix_qu.shape}")# (n_sample, n_vocab)
	
	#print(tfidf_matrix_rf.toarray()[1])
	#print(tfidf_matrix_qu.toarray())

	# Create list with similarity between query and dataset
	kernel_matrix = cosine_similarity(tfidf_matrix_qu, tfidf_matrix_rf)
	print(f">> Cosine Similarity Matrix: {kernel_matrix.shape}")

	"""
	# Best cosine distance for each token independantly
	best_index = extract_best_indices(kernel_matrix, topk=topN)
	print(best_index)
	print(f"> You searched for {qu_phrase}\tTop-{topN} Recommendations:")
	
	with pd.option_context('display.max_rows', 300, 'display.max_colwidth', 1500):
		print(dframe[["nwp_content_results",
									#"referer",
									]
								].iloc[best_index]
					)
	df_rec = dframe.iloc[best_index]
	print("<>"*100)
	print(list(df_rec.columns))
	print("<>"*100)
	print(df_rec)
	#return best_index
	return df_rec
	"""

def get_users_history(dframe, users_list, TopN=5):
	print(f">> Getting history of users: {users_list}")
	searched_phrases = list()
	retrieved_users = list()
	dframe = dframe.fillna({#'snippet_highlighted_words': 0, 
															#'content_highlighted_words': 0,
															'search_query_phrase': '',
														}
													)
	dframe['search_query_phrase'] = [','.join(map(str, elem)) for elem in dframe['search_query_phrase']]

	for usr_i, usr_v in enumerate(users_list):
		print(f"{'QUERY Phrases'.center(50,'-')}")
		print(usr_i, usr_v)
		#print(dframe[ ( dframe["user_ip"] == usr_v ) & ( dframe["search_query_phrase"].notnull() ) ][["search_query_phrase"]].values)
		qu_phs = ["".join(elem) for elem in dframe[(dframe["user_ip"] == usr_v) & ( dframe["search_query_phrase"].notnull() )][["search_query_phrase"]].values.tolist()]
		print(f"All qu phrases: {len(qu_phs)}")
		unq_qu_phs_dict = Counter(qu_phs)
		print(json.dumps(unq_qu_phs_dict, indent=2, ensure_ascii=False))
		unq_qu_phs_dict.pop("", None)

		if len(unq_qu_phs_dict) > 0:
			#print(max(unq_qu_phs_dict, key=unq_qu_phs_dict.get))
			searched_phrases.append(max(unq_qu_phs_dict, key=unq_qu_phs_dict.get))
			retrieved_users.append(usr_v)
		else:
			print(f"\t<!> Useless user!! Trying next user...")
		print()
		"""
		if len(searched_phrases) >= TopN:
			print(f"We found Top-{TopN} results => leave for loop..")
			break
		"""
	for sim_usr, usr_hist in zip(retrieved_users, searched_phrases):
		print(f"\t{sim_usr} : (Top Searched Query Phrase: {usr_hist})")
	print()
	print(f">> search hist: {searched_phrases}")

	return searched_phrases
	
def run_RecSys(df):
	print_df_detail(df, fname=__file__)
	df_RecSys = get_TFIDF_RecSys(qu_phrase=args.qphrase, dframe=df)
	"""
	#sim_users_hist_phrases = get_users_history(dframe=df, users_list=df_RecSys["user_ip"].values.tolist())
	
	print(f"TFIDF Recommendation Result".center(100,'-'))
	print(f"Since you searched for query phrase(s): {Fore.BLUE+Back.YELLOW}{args.qphrase}{Style.RESET_ALL}\n"
				f"you might also be interested in Phrases:\n{Fore.GREEN}{sim_users_hist_phrases}{Style.RESET_ALL}")
	print(f"TFIDF Recommendation Result".center(100,'-'))
	"""

def main():
	df = load_df(infile=args.inputDF)
	run_RecSys(df)
	#return

if __name__ == '__main__':
	os.system("clear")
	main()
	print()