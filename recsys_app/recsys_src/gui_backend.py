import os
import re
import tarfile
import json
import glob
import time
import gzip
import dill
import numpy as np
import pandas as pd
# import urllib.parse
import urllib
import requests
from typing import List, Set, Dict, Tuple
from recsys_app.recsys_src.tokenizer_utils import *
lemmatizer_methods = {
	"nltk": nltk_lemmatizer,
	"trankit": trankit_lemmatizer,
	"stanza": stanza_lemmatizer,
}
###########################################################################################
HOME: str = os.getenv('HOME') # echo $HOME
USER: str = os.getenv('USER') # echo $USER
Files_DIR: str = "/media/volume" if USER == "ubuntu" else HOME
lmMethod: str="stanza"
nSPMs: int = 2 #if USER == "ubuntu" else 50 # dynamic changing of nSPMs due to Rahti CPU memory issues!
DATASET_DIR: str = f"Nationalbiblioteket/compressed_concatenated_SPMs" if USER == "ubuntu" else f"datasets/compressed_concatenated_SPMs"
compressed_spm_file = os.path.join(Files_DIR, DATASET_DIR, f"concat_x{nSPMs}.tar.gz")
spm_files_dir = os.path.join(Files_DIR, DATASET_DIR, f"concat_x{nSPMs}")
fprefix: str = f"concatinated_{nSPMs}_SPMs"
BASE_DIGI_URL: str = "https://digi.kansalliskirjasto.fi/search?requireAllKeywords=true&query="
###########################################################################################

def get_num_results(URL: str="www.example.com"):
	print(f"{URL:<200}")
	st_t = time.time()

	parsed_url = urllib.parse.urlparse(URL)
	# print(parsed_url)
	parameters = urllib.parse.parse_qs( parsed_url.query, keep_blank_values=True)
	# print(parameters)
	# print(f"<>"*50)
	offset_pg=( int( re.search(r'page=(\d+)', URL).group(1) )-1)*20 if re.search(r'page=(\d+)', URL) else 0
	search_pg_api = f"{parsed_url.scheme}://{parsed_url.netloc}/rest/binding-search/search/binding?offset={offset_pg}&count=20"
	# print(offset_pg, search_pg_api)
	payload = {
		"authors": parameters.get('author') if parameters.get('author') else [],
		"collections": parameters.get('collection') if parameters.get('collection') else [],
		"districts": [], # TODO: must be investigated!!
		"endDate": parameters.get('endDate')[0] if parameters.get('endDate') else None,
		"exactCollectionMaterialType": "false", # TODO: must be investigated!!
		"formats": parameters.get('formats') if parameters.get('formats') else [],
		"fuzzy": parameters.get('fuzzy')[0] if parameters.get('fuzzy') else "false",
		"hasIllustrations": parameters.get('hasIllustrations')[0] if parameters.get('hasIllustrations') else "false",
		"importStartDate": parameters.get('importStartDate')[0] if parameters.get('importStartDate') else None,
		"importTime": parameters.get('importTime')[0] if parameters.get('importStartDate') else "ANY",
		"includeUnauthorizedResults": parameters.get('showUnauthorizedResults')[0] if parameters.get('showUnauthorizedResults') else "false",
		"languages": parameters.get('lang') if parameters.get('lang') else [],
		"orderBy": parameters.get('orderBy')[0] if parameters.get('orderBy') else "IMPORT_DATE",
		"pages": parameters.get('pages')[0]  if parameters.get('pages') else "",
		"publicationPlaces": parameters.get('publicationPlace') if parameters.get('publicationPlace') else [],
		"publications": parameters.get('title') if parameters.get('title') else [],
		"publishers": parameters.get('publisher') if parameters.get('publisher') else [],
		"query": parameters.get('query')[0] if parameters.get('query') else "",
		"queryTargetsMetadata": parameters.get('qMeta')[0] if parameters.get('qMeta') else "false",
		"queryTargetsOcrText": parameters.get('qOcr')[0] if parameters.get('qOcr') else "true",
		"requireAllKeywords": parameters.get('requireAllKeywords')[0]  if parameters.get('requireAllKeywords') else "false",
		"searchForBindings": parameters.get('searchForBindings')[0]  if parameters.get('searchForBindings') else "false",
		"showLastPage": parameters.get('showLastPage')[0]  if parameters.get('showLastPage') else "false",
		"startDate": parameters.get('startDate')[0] if parameters.get('startDate') else None,
		"tags": parameters.get('tag') if parameters.get('tag') else [],
	}
	
	headers = {
		'Content-type': 'application/json',
		'Accept': 'application/json; text/plain; */*', 
		'Cache-Control': 'no-cache',
		'Connection': 'keep-alive',
		'Pragma': 'no-cache',
	}

	try:
		r = requests.post(
			url=search_pg_api, 
			json=payload, 
			headers=headers,
		)
		res = r.json()
		# a list of up to 20 results, each of which contains: 
		#print(res.keys()): ['bindingId', 'bindingTitle', 'publicationId', 'generalType', 'authorized', 'authors', 'pageNumber', 'language', 'publisher', 'issue', 'importDate', 'dateAccuracy', 'placeOfPublication', 'textHighlights', 'terms', 'score', 'url', 'thumbnailUrl', 'date']
		SEARCH_RESULTS = res.get("rows")
		
		print(f"Got {len(SEARCH_RESULTS)} NLF baseline result(s)\t{time.time()-st_t:.3f} sec")
		numb = len(SEARCH_RESULTS)
		# print(json.dumps(SEARCH_RESULTS, indent=2, ensure_ascii=False))
		# print("#"*120)
	# except Exception as e:
	# 	print(f"<!> {e}")
	# 	return
	except (
		requests.exceptions.Timeout,
		requests.exceptions.ConnectionError, 
		requests.exceptions.RequestException, 
		requests.exceptions.TooManyRedirects,
		requests.exceptions.InvalidSchema,
		ValueError, 
		TypeError, 
		IndexError,
		EOFError, 
		RuntimeError,
		json.JSONDecodeError,
		json.decoder.JSONDecodeError,
		Exception, 
	) as e:
		print(f"{type(e).__name__} line {e.__traceback__.tb_lineno} in {__file__}: {e.args}")
		return

	return numb

def get_lemmatized_sqp(qu_list, lm: str="stanza"):
	# qu_list = ['some word in this format with always length 1']
	#print(len(qu_list), qu_list)
	assert len(qu_list) == 1, f"query list length MUST be len(qu_list)==1, Now: {len(qu_list)}!!"
	return lemmatizer_methods.get(lm)( clean_(docs=qu_list[0]) )

def get_query_vec(mat, mat_row, mat_col, tokenized_qu_phrases=["åbo", "akademi"]):
	query_vector=np.zeros((1, mat.shape[1]), dtype=np.float32)
	query_vector[0, list(np.in1d(mat_col, tokenized_qu_phrases).nonzero()[0])]=1
	# print(query_vector.shape, query_vector.dtype, np.count_nonzero(query_vector), np.where(query_vector.flatten()!=0)[0])
	#print(np.argsort(tempquery.flatten())[-len(query_words):])
	# print(np.where(query_vector.flatten()!=0)[0])
	return query_vector

def get_optimized_cs(spMtx, query_vec, idf_vec, spMtx_norm, exponent: float=1.0):
	print(f"Optimized Cosine Similarity (1 x nUsers={spMtx.shape[0]})".center(130, "-"))
	print(f"<spMtx> {type(spMtx)} {spMtx.shape} {spMtx.dtype}")
	print(f"<quVec> {type(query_vec)} {query_vec.shape} {query_vec.dtype}")
	print(f"<IDF> {type(idf_vec)} {idf_vec.shape} {idf_vec.dtype}")
	st_t=time.time()
	nUsers, _ = spMtx.shape
	quInterest=np.squeeze(query_vec)*np.squeeze(np.asarray(idf_vec))#(nTokens,)x(nTokens,)
	quInterestNorm=np.linalg.norm(quInterest)#.astype("float32") # float	
	idx_nonzeros=np.nonzero(quInterest)#[1]
	cs=np.zeros(nUsers, dtype=np.float32) # (nUsers,)
	idf_squeezed=np.squeeze(np.asarray(idf_vec))
	quInterest_nonZeros=quInterest[idx_nonzeros]*(1/quInterestNorm)
	# for ui,_ in enumerate(spMtx): # slightly faster than for ui in np.arange(nUsers, dtype=np.int32)
	for ui in np.arange(nUsers, dtype=np.int32):
		usrInterest=np.squeeze(spMtx[ui, idx_nonzeros].toarray())*idf_squeezed[idx_nonzeros] # 1 x len(idx[1])
		usrInterestNorm=spMtx_norm[ui]+1e-18

		# usrInterest_noNorms=usrInterest # added Nov 10th
		# temp_cs_multiplier=np.sum(usrInterest_noNorms*quInterest_nonZeros) # added Nov 10th

		usrInterest=(usrInterest*(1/usrInterestNorm))#**0.1 # seems faster
		# usrInterest=numba_exp(array=(usrInterest*(1/usrInterestNorm)), exponent=0.1)#~0.35s 1cpu=>~0.07s 8cpu

		usrInterest=(usrInterest**exponent) # added Nov 30th

		cs[ui]=np.sum(usrInterest*quInterest_nonZeros)
		# cs[ui]*=temp_cs_multiplier # added Nov 10th
	print(f"Elapsed_t: {time.time()-st_t:.1f} s {type(cs)} {cs.dtype} {cs.shape}".center(130, "-"))
	return cs # (nUsers,)

def get_avg_rec(spMtx, cosine_sim, idf_vec, spMtx_norm):
	nUsers, nTokens= spMtx.shape
	avg_rec=np.zeros(nTokens, dtype=np.float32)# (nTokens,)
	idf_squeezed=np.squeeze(np.asarray(idf_vec))
	non_zero_cosines = np.nonzero(cosine_sim)[0]
	print(
		f"avgRecSys nTKs={spMtx.shape[1]}\n"
		f"spMtx {type(spMtx)} {spMtx.shape} {spMtx.dtype}\n"
		f"CS {type(cosine_sim)} {cosine_sim.shape} {cosine_sim.dtype} NonZero(s): {non_zero_cosines.shape[0]}\n"
		f"IDF {type(idf_vec)} {idf_vec.shape} {idf_vec.dtype}"
	)
	st_t = time.time()
	for nonzero_idx_CCS in non_zero_cosines: # only for those users with NON-Zero Cosine:
		nonzero_idxs=np.nonzero(spMtx[nonzero_idx_CCS, :])[1] # necessary!
		userInterest=np.squeeze(spMtx[nonzero_idx_CCS, nonzero_idxs].toarray())*idf_squeezed[nonzero_idxs] #(nTokens,)x(nTokens,)
		userInterestNorm=spMtx_norm[nonzero_idx_CCS]+1e-18
		userInterest*=(1/userInterestNorm) # (nTokens,)
		update_vec=cosine_sim[nonzero_idx_CCS]*userInterest # (nTokens,)
		avg_rec[nonzero_idxs]+=update_vec # (nTokens,) + (len(idx_nonzeros),)
	avg_rec*=(1/np.sum(cosine_sim))# (nTokens,)
	print(f"Elapsed_t: {time.time()-st_t:.2f} s {type(avg_rec)} {avg_rec.dtype} {avg_rec.shape}".center(130, "-"))	
	return avg_rec #(nTokens,) #(nTokens_shrinked,) # smaller matrix

def get_topK_tokens(mat_cols, avgrec, tok_query: List[str], meaningless_lemmas_list: List[str], raw_query: str="Raw Query Phrase!", K: int=50):
	print(
		f"topK={K} token(s)\n"
		f"Query [raw]: {raw_query}\n"
		f"Query [tokenized]: {raw_query.lower().split()} | tk: {tok_query}"
	)
	st_t = time.time()
	# return [mat_cols[iTK] for iTK in avgrec.argsort()[-K:]][::-1] # n
	# return [mat_cols[iTK] for iTK in avgrec.argsort()[-K:] if mat_cols[iTK] not in tok_query][::-1] # 
	# raw_query.lower().split() in case we have false lemma: ex) tiedusteluorganisaatio puolustusvoimat
	# topK_tokens_list = [mat_cols[iTK] for iTK in avgrec.argsort()[-K:] if ( mat_cols[iTK] not in tok_query and mat_cols[iTK] not in raw_query.lower().split() )][::-1] #

	topK_tokens_list = [
		mat_cols[iTK] 
		for iTK in avgrec.argsort()[-K:] 
		if ( 
			mat_cols[iTK] not in tok_query
			and mat_cols[iTK] not in meaningless_lemmas_list
			and mat_cols[iTK] not in raw_query.lower().split()
			and get_num_results(URL=f"{BASE_DIGI_URL}" + urllib.parse.quote_plus(raw_query + " " + mat_cols[iTK]))>0
		)
	][::-1]

	# TODO: send it to url_scapring for checking len of results:
	# BASE_DIGI_URL: str = "https://digi.kansalliskirjasto.fi/search?query="
	
	# topK_tokens_urls = [scrap_search_page(URL=f"{BASE_DIGI_URL}" + urllib.parse.quote_plus(raw_query + " " + tk)) for tk in topK_tokens_list]
	# topK_tokens_urls = [f"{BASE_DIGI_URL}" + urllib.parse.quote_plus(raw_query + " " + tk) for tk in topK_tokens_list]
	# print(topK_tokens_urls)
	
	# for sq_url in topK_tokens_urls:
	# 	# print(sq_url, type(sq_url))
	# 	n = get_num_results(URL=sq_url)
	# 	print(n)

	print(f"Found {len(topK_tokens_list)} Recommendation results in {time.time()-st_t:.2f} sec".center(130, "-"))
	return topK_tokens_list

def load_pickle(fpath:str="unknown",):
	print(f"Checking for existence? {fpath}")
	st_t = time.time()
	try:
		with gzip.open(fpath, mode='rb') as f:
			pkl=dill.load(f)
	except gzip.BadGzipFile as ee:
		print(f"<!> {ee} | gzip.open NOT functional!")
		with open(fpath, mode='rb') as f:
			pkl=dill.load(f)
	except Exception as e:
		print(f"<<!>> {e} loading for pandas read_pkl...")
		pkl = pd.read_pickle(fpath)
	elpt = time.time()-st_t
	fsize = os.stat( fpath ).st_size / 1e9 # in GB
	print(f"Loaded in: {elpt:.3f} s {type(pkl)} {fsize:.2f} GB".center(150, " "))
	return pkl

def extract_tar(fname):
	output_folder = fname.split(".")[0]
	if not os.path.isdir(output_folder):
		print(f'extracting {nSPMs} nSPMs: {fname}')
		print(f"{output_folder} does not exist, creating...")
		with tarfile.open(fname, 'r:gz') as tfile:
			tfile.extractall(output_folder)

@cache
def get_recsys_results(query_phrase: str="This is a sample query phrase!", nTokens: int=5):
	tokenized_query_phrase = get_lemmatized_sqp(qu_list=[query_phrase], lm=lmMethod)
	print(f"tokenized_query_phrase: {tokenized_query_phrase}")
	if not tokenized_query_phrase:
		print(f"tokenized_query_phrase none=> return!!!!")
		return 
	query_vector=get_query_vec(
		mat=concat_spm_U_x_T,
		mat_row=concat_spm_usrNames,
		mat_col=concat_spm_tokNames,
		tokenized_qu_phrases=tokenized_query_phrase,
	)
	print(
		f"quVec {type(query_vector)} {query_vector.dtype} {query_vector.shape} Allzero? {np.all(query_vector==0.0)}\n"
		f"|NonZeros|: {np.count_nonzero(query_vector)} "
		f"@ idx(s): {np.where(query_vector.flatten()!=0)[0]}\n"
		f"{[f'idx[{qidx}]: {concat_spm_tokNames[qidx]}' for _, qidx in enumerate(np.where(query_vector.flatten()!=0)[0])]}"
	)
	if np.count_nonzero(query_vector) == 0:
		print(f"Sorry! >> {query_phrase} << Not Found in our database! Search something else...")
		return
	ccs=get_optimized_cs(
		spMtx=concat_spm_U_x_T,
		query_vec=query_vector, 
		idf_vec=idf_vec,
		spMtx_norm=usrNorms, # must be adjusted, accordingly!
	)
	avgRecSys=get_avg_rec(
		spMtx=concat_spm_U_x_T,
		cosine_sim=ccs**5,
		idf_vec=idf_vec,
		spMtx_norm=usrNorms,
	)
	topKtokens=get_topK_tokens(
		mat_cols=concat_spm_tokNames,
		avgrec=avgRecSys,
		raw_query=query_phrase,
		tok_query=tokenized_query_phrase,
		meaningless_lemmas_list=UNQ_STW,
		K=25,
	)
	# print(f">>> Found {len(topKtokens)} Recommendations...")
	return topKtokens

extract_tar(fname=compressed_spm_file)

print(f"USER: >>{USER}<< using {nSPMs} nSPMs")
concat_spm_U_x_T=load_pickle(fpath=glob.glob( spm_files_dir+'/'+f'{fprefix}'+'_shrinked_spMtx_USERs_vs_TOKENs_*_nUSRs_x_*_nTOKs.gz')[0])
concat_spm_usrNames=load_pickle(fpath=glob.glob( spm_files_dir+'/'+f'{fprefix}'+'_shrinked_spMtx_rows_*_nUSRs.gz')[0])
concat_spm_tokNames=load_pickle(fpath=glob.glob( spm_files_dir+'/'+f'{fprefix}'+'_shrinked_spMtx_cols_*_nTOKs.gz')[0])
idf_vec=load_pickle(fpath=glob.glob( spm_files_dir+'/'+f'{fprefix}'+'_shrinked_idf_vec_1_x_*_nTOKs.gz')[0])
usrNorms=load_pickle(fpath=glob.glob( spm_files_dir+'/'+f'{fprefix}'+'_shrinked_users_norm_1_x_*_nUSRs.gz')[0])