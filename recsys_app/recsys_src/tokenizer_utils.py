import os
import re
from functools import cache, lru_cache
import contextlib
import logging
import time

with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
	import nltk
	nltk_modules = [
		'punkt',
		'stopwords',
		'wordnet',
		'averaged_perceptron_tagger', 
		'omw-1.4',
	]
	nltk.download(
		#'all',
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
			# 'da',
			# 'nb', 
			# 'ru',
			# 'et',
			# 'de',
			# 'fr',
		]
	}

	lang_configs = {
		"en": {"processors":"tokenize,lemma,pos", "package":'lines',"tokenize_no_ssplit":True},
		"sv": {"processors":"tokenize,lemma,pos","tokenize_no_ssplit":True},
		# "sv": {"processors":"tokenize,lemma,pos", "package":'lines',"tokenize_no_ssplit":True}, # errors!!!
		# "da": {"processors":"tokenize,lemma,pos","tokenize_no_ssplit":True},
		# "nb": {"processors":"tokenize,lemma,pos","tokenize_no_ssplit":True},
		# "ru": {"processors":"tokenize,lemma,pos","tokenize_no_ssplit":True},
		# "fi": {"processors":"tokenize,lemma,pos,mwt", "package":'tdt',"tokenize_no_ssplit":True}, # TDT
		"fi": {"processors":"tokenize,lemma,pos,mwt", "package":'ftb',"tokenize_no_ssplit":True}, # FTB
		# "et": {"processors":"tokenize,lemma,pos", "package":'edt',"tokenize_no_ssplit":True},
		# "de": {"processors":"tokenize,lemma,pos", "package":'hdt',"tokenize_no_ssplit":True},
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
	# UNQ_STW = list(set(STOPWORDS))
	UNQ_STW = set(STOPWORDS)

# logging.getLogger("stanza").setLevel(logging.WARNING) # disable stanza log messages with severity levels of WARNING and higher (ERROR, CRITICAL)

def nltk_lemmatizer(docs):
	return None

def trankit_lemmatizer(docs):
	return None

@cache
def clean_(docs: str="This is a <NORMAL> string!!"):
	print(f'Raw Input:\n>>{docs}<<')
	if not docs or len(docs) == 0 or docs == "":
		return
	t0 = time.time()
	docs = re.sub(
		r'[\{\}@®¤†±©§½✓%,+–;,=&\'\-$€£¥#*"°^~?!❁—.•()˶“”„:/।|‘’<>»«□™♦_■►▼▲❖★☆¶…\\\[\]]+',
		' ',
		docs,
	)
	docs = re.sub(
		r'\b(?:\w*(\w)(\1{2,})\w*)\b|\d+',
		" ",
		docs,
	)
	docs = re.sub(
		r'\s{2,}', 
		" ", 
		# re.sub(r'\b\w{,2}\b', ' ', docs).strip() 
		re.sub(r'\b\w{,2}\b', ' ', docs)#.strip() 
	).strip()
	##########################################################################################
	# TODO: library checking?!!!! Enchant? only for Pouta!
	docs = docs.lower()
	##########################################################################################
	print(f'Cleaned Input [elasped_t: {time.time()-t0:.5f} s]:\n{docs}')
	print(f"<>"*40)
	if not docs or len(docs) == 0 or docs == "":
		return
	return docs

@cache
def stanza_lemmatizer(docs: str="This is a <NORMAL> sentence in document."):
	try:
		print(f'Stanza[{stanza.__version__}] Raw Input: {docs}')
		st_t = time.time()
		all_ = smp(docs)
		lemmas_list = [
			re.sub(r'[";=&#<>_\-\+\^\.\$\[\]]', '', wlm.lower())
			for _, vsnt in enumerate(all_.sentences) 
			for _, vw in enumerate(vsnt.words) 
			if ( 
					(wlm:=vw.lemma)
					and 5 <= len(wlm) <= 43
					and not re.search(r'\b(?=\d|\w)(?:\w*(?<!\b)(\w)(\1{2,})\w*|\d+\w*|\w*\d\w*)\b|<ros>|<eos>|/|<EOS>|<sos>|<SOS>|<UNK>|<unk>|\?|\^|\s+', wlm) # excludes words containing digits!
					and vw.upos not in useless_upos_tags 
					and wlm not in UNQ_STW
			)
		]
		end_t = time.time()
	except Exception as e:
		print(f"<!> Stanza Error: {e}")
		return
	print( lemmas_list )
	print(f"Found {len(lemmas_list)} lemma(s) in {end_t-st_t:.2f} s".center(130, "-") )
	return lemmas_list