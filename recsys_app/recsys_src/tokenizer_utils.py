import os
import re
from functools import cache
import contextlib
import logging
import time

with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
	import nltk
	nltk_modules = ['punkt',
								'stopwords',
								'wordnet',
								'averaged_perceptron_tagger', 
								'omw-1.4',
								]
	nltk.download(#'all',
								nltk_modules,
								quiet=True, 
								raise_on_error=True,
								)	
	import stanza
	from stanza.pipeline.multilingual import MultilingualPipeline
	from stanza.pipeline.core import DownloadMethod

	lang_id_config={
		"langid_lang_subset": [
			'fi', 
			'sv', 
			'en', 
			'da', 
			'ru', 
			'de', 
			# 'fr',
		]
	}

	lang_configs = {
		"en": {"processors":"tokenize,lemma,pos", "package":'lines',"tokenize_no_ssplit":True},
		"sv": {"processors":"tokenize,lemma,pos","tokenize_no_ssplit":True},
		"da": {"processors":"tokenize,lemma,pos","tokenize_no_ssplit":True},
		"ru": {"processors":"tokenize,lemma,pos","tokenize_no_ssplit":True},
		"fi": {"processors":"tokenize,lemma,pos,mwt", "package":'tdt',"tokenize_no_ssplit":True},
		"et": {"processors":"tokenize,lemma,pos", "package":'edt',"tokenize_no_ssplit":True},
		"de": {"processors":"tokenize,lemma,pos", "package":'hdt',"tokenize_no_ssplit":True},
		# "fr": {"processors":"tokenize,lemma,pos,mwt", "package":'sequoia',"tokenize_no_ssplit":True},
	}
	print(f"Creating Stanza[{stanza.__version__}] < MultilingualPipeline >", end=" ")
	tt = time.time()
	# Create the MultilingualPipeline object
	smp = MultilingualPipeline( 
		lang_id_config=lang_id_config,
		lang_configs=lang_configs,
		download_method=DownloadMethod.REUSE_RESOURCES,
	)
	print(f"Elapsed_t: {time.time()-tt:.3f} sec")
		useless_upos_tags = [
		"PUNCT",
		"CCONJ",
		"SCONJ",
		"SYM",
		"AUX",
		"NUM",
		"DET",
		"ADP",
		"PRON",
		"PART",
		"ADV",
		"INTJ",
		# "X", # foriegn words will be excluded,
	]
	
	STOPWORDS = nltk.corpus.stopwords.words(nltk.corpus.stopwords.fileids())
	with open('recsys_app/recsys_src/meaningless_lemmas.txt', 'r') as file_:
		my_custom_stopwords=[line.strip() for line in file_]
	STOPWORDS.extend(my_custom_stopwords)
	UNQ_STW = list(set(STOPWORDS))

logging.getLogger("stanza").setLevel(logging.WARNING) # disable stanza log messages with severity levels of WARNING and higher (ERROR, CRITICAL)

def nltk_lemmatizer(docs):
	return None

def trankit_lemmatizer(docs):
	return None

def clean_(docs: str="This is a <NORMAL> string!!"):
	print(f"<>"*50)
	print(f'Raw input >>{docs}<<')
	# print(f"{f'Inp. word(s): { len( docs.split() ) }':<20}", end="")
	# st_t = time.time()
	if not docs or len(docs) == 0 or docs == "":
		return
	docs = docs.lower()
	# treat all as document
	# docs = re.sub(r'\"|\'|<[^>]+>|[~*^][\d]+', ' ', docs).strip() # "kuuslammi janakkala"^5 or # "karl urnberg"~1
	docs = re.sub(r'[\{\}@®¤†±©§½✓%,+;,=&\'\-$€£¥#*"°^~?!❁—.•()˶“”„:/।|‘’<>»«□™♦_■►▼▲❖★☆¶…\\\[\]]+', ' ', docs )#.strip()
	# docs = " ".join(map(str, [w for w in docs.split() if len(w)>2]))
	# docs = " ".join([w for w in docs.split() if len(w)>2])
	docs = re.sub(r'\b(?:\w*(\w)(\1{2,})\w*)\b|\d+', " ", docs)#.strip()
	# docs = re.sub(r'\s{2,}', " ", re.sub(r'\b\w{,2}\b', ' ', docs).strip() ) # rm words with len() < 3 ex) ö v or l m and extra spaces
	docs = re.sub(r'\s{2,}', 
								" ", 
								# re.sub(r'\b\w{,2}\b', ' ', docs).strip() 
								re.sub(r'\b\w{,2}\b', ' ', docs)#.strip() 
				).strip() # rm words with len() < 3 ex) ö v or l m and extra spaces

	print(f'Cleaned input >>{docs}<<')
	print(f"<>"*50)

	# # print(f"{f'Preprocessed: { len( docs.split() ) } words':<30}{str(docs.split()[:3]):<65}", end="")
	if not docs or len(docs) == 0 or docs == "":
		return
	return docs

@cache
def stanza_lemmatizer(docs):
	try:
		print(f'Stanza[{stanza.__version__}] Raw Input: {docs}')
		# print(f"{f'nW: { len( docs.split() ) }':<10}{str(docs.split()[:7]):<150}", end="")
		st_t = time.time()
		all_ = smp(docs)
		# list comprehension: slow but functional alternative
		# print(f"{f'{ len(all_.sentences) } sent.: { [ len(vsnt.words) for _, vsnt in enumerate(all_.sentences) ] } words':<40}", end="")
		# lemmas_list = [ re.sub(r'"|#|_|\-','', wlm.lower()) for _, vsnt in enumerate(all_.sentences) for _, vw in enumerate(vsnt.words) if ( (wlm:=vw.lemma) and len(wlm)>=3 and len(wlm)<=40 and not re.search(r"\b(?:\w*(\w)(\1{2,})\w*)\b|<eos>|<EOS>|<sos>|<SOS>|<UNK>|<unk>|\s+", wlm) and vw.upos not in useless_upos_tags and wlm not in UNQ_STW ) ]
		lemmas_list = [ 
			re.sub(r'["#_\-]', '', wlm.lower())
			for _, vsnt in enumerate(all_.sentences) 
			for _, vw in enumerate(vsnt.words) 
			if ( 
					(wlm:=vw.lemma)
					and 3 <= len(wlm) <= 40
					and not re.search(r'\b(?:\w*(\w)(\1{2,})\w*)\b|<eos>|<EOS>|<sos>|<SOS>|<UNK>|"|#|<unk>|\s+', wlm) 
					and vw.upos not in useless_upos_tags 
					and wlm not in UNQ_STW
			)
		]
		end_t = time.time()
	except Exception as e:
		print(f"<!> Stanza Error: {e}")
		return
	# print( lemmas_list )
	print(f"Lemmatized to {len(lemmas_list)} lemma(s) {lemmas_list} Elapsed_t: {end_t-st_t:.2f} s")
	print("-"*100)
	return lemmas_list