import os
import re
import tarfile
import json
import glob
import time
import gzip
import dill
import random
import numpy as np
import pandas as pd
import urllib
import aiohttp
import asyncio
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from typing import List, Set, Dict, Tuple
from celery import shared_task
from recsys_app.recsys_src.tokenizer_utils import *
lemmatizer_methods = {
	"nltk": nltk_lemmatizer,
	"trankit": trankit_lemmatizer,
	"stanza": stanza_lemmatizer,
}

session = requests.Session()
retries = Retry(total=5, backoff_factor=0.1, status_forcelist=[500, 502, 503, 504])
session.mount('https://', HTTPAdapter(max_retries=retries))

#######################################################################################################################
HOME: str = os.getenv('HOME') # echo $HOME
USER: str = os.getenv('USER') # echo $USER
Files_DIR: str = "/media/volume" if USER == "ubuntu" else HOME
lmMethod: str="stanza"
nSPMs: int = 732 if USER == "ubuntu" else 2 # dynamic changing of nSPMs due to Rahti CPU memory issues!
DATASET_DIR: str = f"Nationalbiblioteket/compressed_concatenated_SPMs" if USER == "ubuntu" else f"datasets/compressed_concatenated_SPMs"
fprefix: str = f"concatinated_{nSPMs}_SPMs_lm_{lmMethod}"
compressed_spm_file = os.path.join(Files_DIR, DATASET_DIR, f"concat_x{nSPMs}_lm_{lmMethod}.tar.gz")
spm_files_dir = os.path.join(Files_DIR, DATASET_DIR, f"concat_x{nSPMs}_lm_{lmMethod}")

SEARCH_QUERY_DIGI_URL: str = "https://digi.kansalliskirjasto.fi/search?requireAllKeywords=true&query="
DIGI_HOME_PAGE_URL : str = "https://digi.kansalliskirjasto.fi"
headers = {
	'Content-type': 'application/json',
	'Accept': 'application/json; text/plain; */*', 
	'Cache-Control': 'no-cache',
	'Connection': 'keep-alive',
	'Pragma': 'no-cache',
}
payload = {
	"authors": [],
	"collections": [],
	"districts": [],
	"endDate": None,
	"exactCollectionMaterialType": "false",
	"formats": [],
	"fuzzy": "false",
	"hasIllustrations": "false",
	"importStartDate": None,
	"importTime": "ANY",
	"includeUnauthorizedResults": "false",
	"languages": [],
	"orderBy": "RELEVANCE",
	"pages": "",
	"publicationPlaces": [],
	"publications": [],
	"publishers": [],
	"queryTargetsMetadata": "false",
	"queryTargetsOcrText": "true",
	"searchForBindings": "false",
	"showLastPage": "false",
	"startDate": None,
	"tags": [],	
}
#######################################################################################################################

@cache
def get_nlf_pages(INPUT_QUERY: str="global warming"):
	st_t = time.time()
	URL = f"{SEARCH_QUERY_DIGI_URL}" + urllib.parse.quote_plus(INPUT_QUERY)
	print(f"{URL:<150}", end=" ")
	parsed_url = urllib.parse.urlparse(URL)
	parameters = urllib.parse.parse_qs( parsed_url.query, keep_blank_values=True)
	offset_pg = ( int( re.search(r'page=(\d+)', URL).group(1) )-1)*20 if re.search(r'page=(\d+)', URL) else 0
	search_page_request_url = f"{DIGI_HOME_PAGE_URL}/rest/binding-search/search/binding?offset={offset_pg}&count=20"
	payload["query"] = parameters.get('query')[0] if parameters.get('query') else ""
	payload["requireAllKeywords"] = parameters.get('requireAllKeywords')[0] if parameters.get('requireAllKeywords') else "false"
	try:
		r = session.post(
			url=search_page_request_url, 
			json=payload, 
			headers=headers,
		)
		r.raise_for_status()  # Raise HTTPError for bad status codes
		res = r.json()
		TOTAL_NUM_NLF_RESULTs = res.get("totalResults")
		print(f"Found NLF tot_page(s): {TOTAL_NUM_NLF_RESULTs:<10} in {time.time()-st_t:.1f} sec")
	except requests.exceptions.RequestException as e:
		print(f"<!> Error: {e}")
		return
	return TOTAL_NUM_NLF_RESULTs

def get_lemmatized_sqp(qu_phrase, lm: str="stanza"):
	return lemmatizer_methods.get(lm)( clean_(docs=qu_phrase) )

def get_query_vec(mat, mat_row, mat_col, tokenized_qu_phrases=["åbo", "akademi"]):
	query_vector=np.zeros((1, mat.shape[1]), dtype=np.float32)
	query_vector[0, list(np.in1d(mat_col, tokenized_qu_phrases).nonzero()[0])]=1
	# print(query_vector.shape, query_vector.dtype, np.count_nonzero(query_vector), np.where(query_vector.flatten()!=0)[0])
	#print(np.argsort(tempquery.flatten())[-len(query_words):])
	# print(np.where(query_vector.flatten()!=0)[0])
	return query_vector

def get_customized_cosine_similarity(spMtx, query_vec, idf_vec, spMtx_norm, exponent: float=1.0):
	print(f"Customized Cosine Similarity (1 x nUsers={spMtx.shape[0]})".center(130, "-"))
	print(
		f"Query: {query_vec.shape} {type(query_vec)} {query_vec.dtype}\n"
		f"spMtx {type(spMtx)} {spMtx.shape} {spMtx.dtype}\n"
		f"spMtxNorm: {type(spMtx_norm)} {spMtx_norm.shape} {spMtx_norm.dtype}\n"
		f"IDF {type(idf_vec)} {idf_vec.shape} {idf_vec.dtype}"
	)
	st_t=time.time()
	# nUsers, _ = spMtx.shape
	# quInterest=np.squeeze(query_vec)*np.squeeze(np.asarray(idf_vec))#(nTokens,)x(nTokens,)
	# quInterestNorm=np.linalg.norm(quInterest)#.astype("float32") # float	
	# idx_nonzeros=np.nonzero(quInterest)#[1]
	# cs=np.zeros(nUsers, dtype=np.float32) # (nUsers,)
	# idf_squeezed=np.squeeze(np.asarray(idf_vec))
	# quInterest_nonZeros=quInterest[idx_nonzeros]*(1/quInterestNorm)
	# # for ui,_ in enumerate(spMtx): # slightly faster than for ui in np.arange(nUsers, dtype=np.int32)
	# for ui in np.arange(nUsers, dtype=np.int32):
	# 	usrInterest=np.squeeze(spMtx[ui, idx_nonzeros].toarray())*idf_squeezed[idx_nonzeros] # 1 x len(idx[1])
	# 	usrInterestNorm=spMtx_norm[ui]+1e-18

	# 	# usrInterest_noNorms=usrInterest # added Nov 10th
	# 	# temp_cs_multiplier=np.sum(usrInterest_noNorms*quInterest_nonZeros) # added Nov 10th

	# 	usrInterest=(usrInterest*(1/usrInterestNorm))#**0.1 # seems faster
	# 	# usrInterest=numba_exp(array=(usrInterest*(1/usrInterestNorm)), exponent=0.1)#~0.35s 1cpu=>~0.07s 8cpu

	# 	usrInterest=(usrInterest**exponent) # added Nov 30th

	# 	cs[ui]=np.sum(usrInterest*quInterest_nonZeros)
	# 	# cs[ui]*=temp_cs_multiplier # added Nov 10th
	####################################################################################################
	# nUsers, _ = spMtx.shape
	# idf_squeezed = idf_vec.ravel() # faster than np.squeeze(np.asarray(idf_vec))
	# query_vec_squeezed = query_vec.ravel()
	# quInterest = query_vec_squeezed * idf_squeezed #(nTokens,)x(nTokens,)
	# quInterestNorm = np.linalg.norm(quInterest)
	# idx_nonzeros = np.nonzero(quInterest)#[1]
	# cs=np.zeros(nUsers, dtype=np.float32) # (nUsers,)
	# quInterest_nonZeros=quInterest[idx_nonzeros] * (1/quInterestNorm)
	# usrInterestNorm=spMtx_norm+np.float32(1e-18)    
	# for ui in np.arange(nUsers, dtype=np.int32): # ip1, ip2, ..., ipN
	# 	usrInterest = spMtx[ui, idx_nonzeros].toarray().ravel() * idf_squeezed[idx_nonzeros]
	# 	usrInterest=(usrInterest*(1/usrInterestNorm[ui]))#**0.1 # faster?!        
	# 	usrInterest=(usrInterest**exponent)
	# 	cs[ui]=np.sum(usrInterest*quInterest_nonZeros)
	# return cs # (nUsers,)
	####################################################################################################

	################################### Vectorized Implementation ##########################################
	idf_squeezed = idf_vec.ravel()
	query_vec_squeezed = query_vec.ravel()
	quInterest = query_vec_squeezed * idf_squeezed
	quInterestNorm = np.linalg.norm(quInterest)
	
	idx_nonzeros = np.nonzero(quInterest)[0] # Get the indices of non-zero elements in quInterest
	quInterest_nonZeros = quInterest[idx_nonzeros] / quInterestNorm
	usrInterestNorm = spMtx_norm + np.float32(1e-18)
	
	# Extract only the necessary columns from the sparse matrix
	spMtx_nonZeros = spMtx[:, idx_nonzeros].tocsc()  # Converting to CSC for faster column slicing
	
	# Calculate user interest by element-wise multiplication with IDF
	spMtx_nonZeros = spMtx_nonZeros.multiply(idf_squeezed[idx_nonzeros])
	
	# Normalize user interests
	spMtx_nonZeros = spMtx_nonZeros.multiply(1 / usrInterestNorm[:, None])
	
	# Apply exponent if necessary
	if exponent != 1.0:
		spMtx_nonZeros.data **= exponent
	
	cs = spMtx_nonZeros.dot(quInterest_nonZeros) # Compute the cosine similarity scores
	
	print(f"Elapsed_t: {time.time()-st_t:.2f} s {type(cs)} {cs.dtype} {cs.shape}".center(130, " "))
	return cs
	################################### Vectorized Implementation ##########################################

def get_customized_recsys_avg_vec(spMtx, cosine_sim, idf_vec, spMtx_norm):
	print(f"avgRecSys (1 x nTKs={spMtx.shape[1]})".center(130, "-"))
	st_t = time.time()

	####################################################################################################################################
	# using Only USERs with nonZero Cosine Similarity: (less efficient)
	# nUsers, nTokens= spMtx.shape
	# avg_rec=np.zeros(nTokens, dtype=np.float32)# (nTokens,)
	# idf_squeezed=np.squeeze(np.asarray(idf_vec))
	# non_zero_cosines = np.nonzero(cosine_sim)[0]
	# print(
	# 	f"spMtx {type(spMtx)} {spMtx.shape} {spMtx.dtype}\n"
	# 	f"spMtxNorm: {type(spMtx_norm)} {spMtx_norm.shape} {spMtx_norm.dtype}\n"
	# 	f"CS {type(cosine_sim)} {cosine_sim.shape} {cosine_sim.dtype} NonZero(s): {non_zero_cosines.shape[0]}\n"
	# 	f"IDF {type(idf_vec)} {idf_vec.shape} {idf_vec.dtype}"
	# )
	# st_t = time.time()
	# for nonzero_idx_CCS in non_zero_cosines: # only for those users with NON-Zero Cosine:
	# 	nonzero_idxs=np.nonzero(spMtx[nonzero_idx_CCS, :])[1] # necessary!
	# 	userInterest=np.squeeze(spMtx[nonzero_idx_CCS, nonzero_idxs].toarray())*idf_squeezed[nonzero_idxs] #(nTokens,)x(nTokens,)
	# 	userInterestNorm=spMtx_norm[nonzero_idx_CCS]+1e-18
	# 	userInterest*=(1/userInterestNorm) # (nTokens,)
	# 	update_vec=cosine_sim[nonzero_idx_CCS]*userInterest # (nTokens,)
	# 	avg_rec[nonzero_idxs]+=update_vec # (nTokens,) + (len(idx_nonzeros),)
	# avg_rec*=(1/np.sum(cosine_sim))# (nTokens,)
	# print(f"Elapsed_t: {time.time()-st_t:.2f} s {type(avg_rec)} {avg_rec.dtype} {avg_rec.shape}".center(130, " "))	
	# return avg_rec #(nTokens,) #(nTokens_shrinked,) # smaller matrix
	####################################################################################################################################
	
	# ####################################################################################################################################
	# # # Optimized implementation using for loop [15.05.2024] ~120s for 732 log files
	# nUsers, nTokens= spMtx.shape
	# avg_rec=np.zeros(nTokens, dtype=np.float32) #(nTokens,)
	# idf_squeezed = idf_vec.ravel() # faster than np.squeeze(np.asarray(idf_vec))
	# non_zero_cosines = np.nonzero(cosine_sim)[0]
	# userInterestNorm=spMtx_norm + np.float32(1e-18)# avoid zero division
	# print(
	# 	f"spMtx {type(spMtx)} {spMtx.shape} {spMtx.dtype}\n"
	# 	f"spMtxNorm: {type(spMtx_norm)} {spMtx_norm.shape} {spMtx_norm.dtype}\n"
	# 	f"CS {type(cosine_sim)} {cosine_sim.shape} {cosine_sim.dtype} NonZero(s): {non_zero_cosines.shape[0]}\n"
	# 	f"IDF {type(idf_vec)} {idf_vec.shape} {idf_vec.dtype}"
	# )
	# spMtx_csr = spMtx#.tocsr() # Convert to CSR for efficient row operations
	# for nonzero_idx_CCS in non_zero_cosines: # only for those users with NON-Zero Cosine Similarity
	# 	userInterest = spMtx_csr[nonzero_idx_CCS].multiply(idf_squeezed) #(nTokens,)x(nTokens,)        
	# 	userInterest.data /= userInterestNorm[nonzero_idx_CCS] #(nTokens,)
	# 	update_vec = cosine_sim[nonzero_idx_CCS]*userInterest.data #(nTokens,)
	# 	avg_rec[userInterest.nonzero()[1]] += update_vec # (nTokens,) + (len(idx_nonzeros),)
	# avg_rec /= np.sum(cosine_sim) # (nTokens,)
	# print(f"Elapsed_t: {time.time()-st_t:.2f} s {type(avg_rec)} {avg_rec.dtype} {avg_rec.shape}".center(130, " "))	
	# return avg_rec #(nTokens,) #(nTokens_shrinked,) # smaller matrix
	# ####################################################################################################################################

	#################################################Vectorized Version#################################################
	nUsers, nTokens = spMtx.shape
	avg_rec = np.zeros(nTokens, dtype=np.float32)
	idf_squeezed = idf_vec.ravel()
	non_zero_cosines = np.nonzero(cosine_sim)[0]
	non_zero_values = cosine_sim[non_zero_cosines]
	userInterestNorm = spMtx_norm + np.float32(1e-18)# avoid zero division
	print(
		f"spMtx {type(spMtx)} {spMtx.shape} {spMtx.dtype}\n"
		f"spMtxNorm: {type(spMtx_norm)} {spMtx_norm.shape} {spMtx_norm.dtype}\n"
		f"CS {type(cosine_sim)} {cosine_sim.shape} {cosine_sim.dtype} NonZero(s): {non_zero_cosines.shape[0]}\n"
		f"IDF {type(idf_vec)} {idf_vec.shape} {idf_vec.dtype}"
	)
	# Process only rows with non-zero cosine similarities
	spMtx_non_zero = spMtx[non_zero_cosines]
	
	# Element-wise multiplication with IDF vector
	userInterest = spMtx_non_zero.multiply(idf_squeezed).tocsr()
	
	# Normalize user interest vectors
	norm_factors = np.repeat(
		userInterestNorm[non_zero_cosines], 
		np.diff(userInterest.indptr)
	)
	userInterest.data /= norm_factors
	
	# Multiply by cosine similarities
	cosine_factors = np.repeat(
		non_zero_values, 
		np.diff(userInterest.indptr)
	)
	userInterest.data *= cosine_factors
	
	# Sum the weighted user interest vectors
	avg_rec = userInterest.sum(axis=0).A1
	
	# Normalize the result
	avg_rec /= np.sum(non_zero_values)
	
	print(f"Elapsed_t: {time.time()-st_t:.2f} s {type(avg_rec)} {avg_rec.dtype} {avg_rec.shape}".center(130, " "))	
	return avg_rec

async def get_recommendation_num_NLF_pages_async(session, INPUT_QUERY: str="global warming", REC_TK: str="pollution"):
	URL = f"{SEARCH_QUERY_DIGI_URL}" + urllib.parse.quote_plus(INPUT_QUERY + " " + REC_TK)
	# print(f"{URL:<150}", end=" ")
	parsed_url = urllib.parse.urlparse(URL)
	parameters = urllib.parse.parse_qs(parsed_url.query, keep_blank_values=True)
	offset_pg = (int(re.search(r'page=(\d+)', URL).group(1)) - 1) * 20 if re.search(r'page=(\d+)', URL) else 0
	search_page_request_url = f"{DIGI_HOME_PAGE_URL}/rest/binding-search/search/binding?offset={offset_pg}&count=20"
	payload["query"] = parameters.get('query')[0] if parameters.get('query') else ""
	payload["requireAllKeywords"] = parameters.get('requireAllKeywords')[0] if parameters.get('requireAllKeywords') else "false"
	try:
		async with session.post(
			url=search_page_request_url,
			json=payload,
			headers=headers,
		) as response:
				response.raise_for_status()
				res = await response.json()
				TOTAL_NUM_NLF_RESULTs = res.get("totalResults")
				# print(f"Found NLF tot_page(s): {TOTAL_NUM_NLF_RESULTs:<10} in {time.time() - st_t:.1f} sec")
				return TOTAL_NUM_NLF_RESULTs
	except aiohttp.ClientError as e:
		print(f"<!> Error: {e}")
		return None

async def get_num_NLF_pages_asynchronous_run(qu: str="global warming", TOKENs_list: List[str]=["tk1", "tk2"]):
	async with aiohttp.ClientSession() as session:
		tasks = [
			NUMBER_OF_PAGES
			for tk in TOKENs_list
			if (
				(NUMBER_OF_PAGES:=get_recommendation_num_NLF_pages_async(session, INPUT_QUERY=qu, REC_TK=tk))
			)
		]
		num_NLF_pages = await asyncio.gather(*tasks)
		# # Filter out None and 0 values
		# num_NLF_pages = [pages for pages in num_NLF_pages if pages not in [None, 0]]
		return num_NLF_pages

def get_topK_tokens(mat_cols, avgrec, tok_query: List[str], meaningless_lemmas_list: List[str], raw_query: str="Raw Query Phrase!", K: int=50):
	print(
		f"Looking for < topK={K} > token(s)...\n"
		f"Query [raw]: {raw_query}\n"
		f"Query [tokenized]: {raw_query.lower().split()} | tk: {tok_query}"
	)
	st_t = time.time()
	topK_tokens_list = []
	# tot_nlf_res_list = []
	for iTK in avgrec.argsort()[-K:]:
		recommended_token = mat_cols[iTK]
		if (
			recommended_token not in tok_query
			and recommended_token not in meaningless_lemmas_list
			and recommended_token not in raw_query.lower().split()
		):
			# tot_nlf_res = get_num_NLF_pages(INPUT_QUERY=raw_query, REC_TK=recommended_token)
			# if tot_nlf_res > 0:
			# 	tot_nlf_res_list.append(tot_nlf_res)
				# tot_nlf_res_list = [random.randint(1, 9000) for i, v in enumerate(topK_tokens_list)]
				topK_tokens_list.append(recommended_token)
	tot_nlf_res_list = asyncio.run(get_num_NLF_pages_asynchronous_run(qu=raw_query, TOKENs_list=topK_tokens_list))

	# remove zeros:
	tot_nlf_res_list_tmp = tot_nlf_res_list
	topK_tokens_list_tmp = topK_tokens_list
	tot_nlf_res_list = [num for num, word in zip(tot_nlf_res_list_tmp, topK_tokens_list_tmp) if (num and num != 0) ]
	topK_tokens_list = [word for num, word in zip(tot_nlf_res_list_tmp, topK_tokens_list_tmp) if (num and num != 0) ]

	# sort descending:
	tot_nlf_res_list = tot_nlf_res_list[::-1]
	topK_tokens_list = topK_tokens_list[::-1]

	print(
		f"Found {len(topK_tokens_list)} Recommendation Results "
		f"(with {len(tot_nlf_res_list)} NLF pages) "
		f"in {time.time()-st_t:.2f} sec"
		.center(130, "-")
	)
	return topK_tokens_list, tot_nlf_res_list

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
	tokenized_query_phrase = get_lemmatized_sqp(qu_phrase=query_phrase, lm=lmMethod)
	print(f"Search Query Prompt: {query_phrase} [lemma(s)]: {tokenized_query_phrase}")
	if not tokenized_query_phrase:
		return None, 0

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
	if not np.any(query_vector):
		print(f"Sorry! >> {query_phrase} << Not Found in our database! Search something else...")
		return None, 0
	ccs=get_customized_cosine_similarity(
		spMtx=concat_spm_U_x_T,
		query_vec=query_vector, 
		idf_vec=idf_vec,
		spMtx_norm=usrNorms, # must be adjusted, accordingly!
	)
	avgRecSys = get_customized_recsys_avg_vec(
		spMtx=concat_spm_U_x_T,
		cosine_sim=ccs**5,
		idf_vec=idf_vec,
		spMtx_norm=usrNorms,
	)
	topK_TKs, topK_TKs_nlf_num_pages = get_topK_tokens(
		mat_cols=concat_spm_tokNames,
		avgrec=avgRecSys,
		raw_query=query_phrase,
		tok_query=tokenized_query_phrase,
		meaningless_lemmas_list=UNQ_STW,
		K=nTokens,
	)
	return topK_TKs, topK_TKs_nlf_num_pages

extract_tar(fname=compressed_spm_file)

print(f"USER: >>{USER}<< using {nSPMs} nSPMs")
# print(fprefix)
# print(spm_files_dir+'/'+f'{fprefix}'+'_shrinked_spMtx_USERs_vs_TOKENs_*_nUSRs_x_*_nTOKs.gz')
# print(glob.glob( spm_files_dir+'/'+f'{fprefix}'+'_shrinked_spMtx_USERs_vs_TOKENs_*_nUSRs_x_*_nTOKs.gz' ))

concat_spm_U_x_T = load_pickle(
	fpath=glob.glob( spm_files_dir+'/'+f'{fprefix}'+'_shrinked_spMtx_USERs_vs_TOKENs_*_nUSRs_x_*_nTOKs.gz' )[0]
)
concat_spm_usrNames = load_pickle(
	fpath=glob.glob( spm_files_dir+'/'+f'{fprefix}'+'_shrinked_spMtx_rows_*_nUSRs.gz')[0]
)
concat_spm_tokNames = load_pickle(
	fpath=glob.glob( spm_files_dir+'/'+f'{fprefix}'+'_shrinked_spMtx_cols_*_nTOKs.gz')[0]
)
idf_vec = load_pickle(
	fpath=glob.glob( spm_files_dir+'/'+f'{fprefix}'+'_shrinked_idf_vec_1_x_*_nTOKs.gz')[0]
)
usrNorms=load_pickle(
	fpath=glob.glob( spm_files_dir+'/'+f'{fprefix}'+'_shrinked_users_norm_1_x_*_nUSRs.gz')[0]
)