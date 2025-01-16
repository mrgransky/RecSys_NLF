import os
import re
import tarfile
import json
import glob
import time
import gzip
import dill
import random
import torch
import urllib
import aiohttp
import asyncio
import requests

from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from typing import List, Set, Dict, Tuple
from celery import shared_task
from recsys_app.recsys_src.tokenizer_utils import *

import numpy as np
import pandas as pd
import cupy as cp

#######################################################################################################################
lemmatizer_methods = {
	"nltk": nltk_lemmatizer,
	"trankit": trankit_lemmatizer,
	"stanza": stanza_lemmatizer,
}
session = requests.Session()
retries = Retry(total=5, backoff_factor=0.1, status_forcelist=[500, 502, 503, 504])
session.mount('https://', HTTPAdapter(max_retries=retries))

HOME: str = os.getenv('HOME') # echo $HOME
USER: str = os.getenv('USER') # echo $USER
Files_DIR: str = "/media/volume" if USER == "ubuntu" else HOME
lmMethod: str="stanza"
nSPMs: int = 732 if USER == "ubuntu" else 2 # dynamic changing of nSPMs due to Rahti CPU memory issues!
DATASET_DIR: str = f"Nationalbiblioteket/compressed_concatenated_SPMs" if USER == "ubuntu" else f"datasets/compressed_concatenated_SPMs"
compressed_spm_file = os.path.join(Files_DIR, DATASET_DIR, f"concat_x{nSPMs}_lm_{lmMethod}.tar.gz")
fprefix: str = f"concatinated_{nSPMs}_SPMs_lm_{lmMethod}"
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

def get_device_with_most_free_memory():
	if torch.cuda.is_available():
		print(f"Available GPU(s) = {torch.cuda.device_count()}")
		max_free_memory = 0
		selected_device = 0
		for i in range(torch.cuda.device_count()):
			torch.cuda.set_device(i)
			free_memory = torch.cuda.mem_get_info()[0]
			if free_memory > max_free_memory:
				max_free_memory = free_memory
				selected_device = i
		device = torch.device(f"cuda:{selected_device}")
		print(f"Selected GPU: cuda:{selected_device} with {max_free_memory / 1024**3:.2f} GB free memory")
	else:
		device = torch.device("cpu")
		print("No GPU available ==>> using CPU")
	return device
device = get_device_with_most_free_memory()
print(f"USER: >>{USER}<< using {nSPMs} nSPMs | Device: {device}")

#######################################################################################################################
def load_pickle(fpath: str="path/to/file.pkl",):
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

def extract_tar(fname: str="file_name"):
	output_folder = fname.split(".")[0]
	if not os.path.isdir(output_folder):
		print(f'extracting {nSPMs} nSPMs: {fname}')
		print(f"< {output_folder} > NOT found, Creating...")
		with tarfile.open(fname, 'r:gz') as tfile:
			tfile.extractall(output_folder)

def is_substring(A: str="evankelis luterilainen kirkko", B: str="evankelisluterilainen") -> bool:
	words_in_A = A.lower().split()
	# print(f"Q: {words_in_A} | Recommended: {B}")
	for word in words_in_A:
		# print(word)
		if word in B or B in word:
			print(f"\t>> Q: {words_in_A} | < {word} > removing Recommeded: {B}")
			return False
	# print("True")
	return True

@cache
def get_nlf_pages(INPUT_QUERY: str="global warming"):
	print(f"Checking NLF for existence of < {INPUT_QUERY} > ...")
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
	cleaned_phrase = clean_(docs=qu_phrase) 
	return lemmatizer_methods.get(lm)( docs=cleaned_phrase, device=device )

def get_query_vec(mat, mat_row, mat_col, tokenized_qu_phrases=["åbo", "akademi"]):
	query_vector=np.zeros((1, mat.shape[1]), dtype=np.float32)
	query_vector[0, list(np.in1d(mat_col, tokenized_qu_phrases).nonzero()[0])]=1
	# print(query_vector.shape, query_vector.dtype, np.count_nonzero(query_vector), np.where(query_vector.flatten()!=0)[0])
	#print(np.argsort(tempquery.flatten())[-len(query_words):])
	# print(np.where(query_vector.flatten()!=0)[0])
	return query_vector

def get_customized_cosine_similarity_gpu(spMtx, query_vec, idf_vec, spMtx_norm, exponent:float=1.0, batch_size:int=1024):
		print(f"[GPU Optimized] Customized Cosine Similarity (1 x nUsers={spMtx.shape[0]}) batch_size={batch_size}".center(130, "-"))
		print(
				f"Query: {query_vec.shape} {type(query_vec)} {query_vec.dtype} non_zeros={np.count_nonzero(query_vec)} non_zero_ratio={np.count_nonzero(query_vec) / query_vec.size:.2f}\n"
				f"spMtx {type(spMtx)} {spMtx.shape} {spMtx.dtype}\n"
				f"spMtxNorm: {type(spMtx_norm)} {spMtx_norm.shape} {spMtx_norm.dtype}\n"
				f"IDF {type(idf_vec)} {idf_vec.shape} {idf_vec.dtype}"
		)
		st_t = time.time()

		# Convert inputs to CuPy arrays (float32)
		query_vec_squeezed = cp.asarray(query_vec.ravel(), dtype=cp.float32)
		idf_squeezed = cp.asarray(idf_vec.ravel(), dtype=cp.float32)
		spMtx_norm = cp.asarray(spMtx_norm, dtype=cp.float32)

		# Convert sparse matrix to CuPy CSR format
		spMtx_csr = spMtx.tocsr()
		spMtx_gpu = cp.sparse.csr_matrix(
				(cp.asarray(spMtx_csr.data, dtype=cp.float32), cp.asarray(spMtx_csr.indices), cp.asarray(spMtx_csr.indptr)),
				shape=spMtx_csr.shape
		)

		# Compute quInterest and its norm
		quInterest = query_vec_squeezed * idf_squeezed
		quInterestNorm = cp.linalg.norm(quInterest)

		# Get indices of non-zero elements in quInterest
		idx_nonzeros = cp.nonzero(quInterest)[0]
		quInterest_nonZeros = quInterest[idx_nonzeros] / quInterestNorm

		# Normalize user interests
		usrInterestNorm = spMtx_norm + cp.float32(1e-4)

		# Initialize result array
		cs = cp.zeros(spMtx_gpu.shape[0], dtype=cp.float32)

		# Process in batches to avoid memory overflow
		for i in range(0, spMtx_gpu.shape[0], batch_size):
				# Define batch range
				start_idx = i
				end_idx = min(i + batch_size, spMtx_gpu.shape[0])

				# Extract batch from sparse matrix
				spMtx_batch = spMtx_gpu[start_idx:end_idx, :]

				# Extract only the necessary columns from the batch
				spMtx_nonZeros = spMtx_batch[:, idx_nonzeros]

				# Apply IDF and normalize
				spMtx_nonZeros = spMtx_nonZeros.multiply(idf_squeezed[idx_nonzeros])
				spMtx_nonZeros = spMtx_nonZeros.multiply(1 / usrInterestNorm[start_idx:end_idx, None])

				# Apply exponent if necessary
				if exponent != 1.0:
						spMtx_nonZeros.data **= exponent

				# Compute cosine similarity scores for the batch
				cs_batch = spMtx_nonZeros.dot(quInterest_nonZeros)

				# Store batch results
				cs[start_idx:end_idx] = cs_batch

				# Free memory for the batch
				del spMtx_batch, spMtx_nonZeros, cs_batch
				cp.get_default_memory_pool().free_all_blocks()

		print(f"Elapsed_t: {time.time() - st_t:.2f} s {type(cs)} {cs.dtype} {cs.shape}".center(130, " "))
		return cp.asnumpy(cs)  # Convert result back to NumPy for compatibility

def get_customized_cosine_similarity(spMtx, query_vec, idf_vec, spMtx_norm, exponent: float=1.0):
	print(f"Customized Cosine Similarity (1 x nUsers={spMtx.shape[0]})".center(130, "-"))
	print(
		f"Query: {query_vec.shape} {type(query_vec)} {query_vec.dtype}\n"
		f"spMtx {type(spMtx)} {spMtx.shape} {spMtx.dtype}\n"
		f"spMtxNorm: {type(spMtx_norm)} {spMtx_norm.shape} {spMtx_norm.dtype}\n"
		f"IDF {type(idf_vec)} {idf_vec.shape} {idf_vec.dtype}"
	)
	st_t=time.time()
	################################### Vectorized Implementation ##########################################
	idf_squeezed = idf_vec.ravel()
	query_vec_squeezed = query_vec.ravel()
	quInterest = query_vec_squeezed * idf_squeezed # Element-wise multiplication
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

def count_years_by_range(yr_vs_nPGs: Dict[str, int], ts_1st: int=1899, ts_2nd=np.arange(1900, 1919+1, 1), ts_3rd=np.arange(1920, 1945+1, 1), ts_end: int=1946):
	first_range = 0
	second_range = 0
	third_range = 0
	forth_range = 0
	if not yr_vs_nPGs:
		return [0, 0, 0, 0]	
	for year, count in yr_vs_nPGs.items():
		year_int = int(year)
		if year_int <= ts_1st: # green
			first_range += count
		elif year_int in ts_2nd: # WWI: 28 july 1914 – 11 nov. 1918 # pink
			second_range += count
		elif year_int in ts_3rd: # WWII: 1 sep. 1939 – 2 sep. 1945 # blue
			third_range += count
		elif year_int >= ts_end: # red
			forth_range += count
	yearly_nlf_pages = [first_range, second_range, third_range, forth_range]
	return yearly_nlf_pages

async def get_recommendation_num_NLF_pages_async(session, INPUT_QUERY: str="global warming", REC_TK: str="pollution", ts_1st: int=1899, ts_2nd=np.arange(1900, 1919+1, 1), ts_3rd=np.arange(1920, 1945+1, 1), ts_end: int=1946):
	URL = f"{SEARCH_QUERY_DIGI_URL}" + urllib.parse.quote_plus(INPUT_QUERY + " " + REC_TK)
	# print(f"{URL:<150}", end=" ")
	parsed_url = urllib.parse.urlparse(URL)
	parameters = urllib.parse.parse_qs(parsed_url.query, keep_blank_values=True)
	offset_pg = (int(re.search(r'page=(\d+)', URL).group(1)) - 1) * 20 if re.search(r'page=(\d+)', URL) else 0
	search_page_request_url = f"{DIGI_HOME_PAGE_URL}/rest/binding-search/search/binding?offset={offset_pg}&count=20"
	payload["query"] = parameters.get('query')[0] if parameters.get('query') else ""
	payload["requireAllKeywords"] = parameters.get('requireAllKeywords')[0] if parameters.get('requireAllKeywords') else "false"
	st_t = time.time()
	try:
		async with session.post(
			url=search_page_request_url,
			json=payload,
			headers=headers,
		) as response:
				response.raise_for_status()
				res = await response.json()
				TOTAL_NUM_NLF_RESULTs = res.get("totalResults") # <class 'int'>
				NLF_pages_by_year_dict = res.get("hitsByYear") # <class 'dict'>: 'year': num_pgs EX) '1939':10
				NLF_pages_by_year_list = count_years_by_range(
					yr_vs_nPGs=NLF_pages_by_year_dict,
					ts_1st=ts_1st,
					ts_2nd=ts_2nd,
					ts_3rd=ts_3rd,
					ts_end=ts_end,
				)
				# print(type(NLF_pages_by_year_dict), NLF_pages_by_year_dict)
				# print(type(NLF_pages_by_year_list), NLF_pages_by_year_list)
				# print(f"Found {type(TOTAL_NUM_NLF_RESULTs)} NLF tot_page(s): {TOTAL_NUM_NLF_RESULTs:<10} in {time.time() - st_t:.1f} sec")
				# print()
				return TOTAL_NUM_NLF_RESULTs, NLF_pages_by_year_list
	except (
		aiohttp.ClientError,
		asyncio.TimeoutError,
	) as e:
		print(f"<!> Error: {e}")
		return None, None

async def get_num_NLF_pages_asynchronous_run(qu: str="global warming", TOKENs_list: List[str]=["tk1", "tk2"], ts_1st: int=1899, ts_2nd=np.arange(1900, 1919+1, 1), ts_3rd=np.arange(1920, 1945+1, 1), ts_end: int=1946):
	# async with aiohttp.ClientSession() as session:
	# 		tasks = [
	# 			NLF_TOT_NUM_PGs
	# 			for tk in TOKENs_list
	# 			if (
	# 				(NLF_TOT_NUM_PGs:=get_recommendation_num_NLF_pages_async(session, INPUT_QUERY=qu, REC_TK=tk))
	# 			)
	# 		]
	# 		num_NLF_pages = await asyncio.gather(*tasks)
	# 		return num_NLF_pages
	async with aiohttp.ClientSession() as session:
		tasks = [
			get_recommendation_num_NLF_pages_async(
				session, 
				INPUT_QUERY=qu, 
				REC_TK=tk,
				ts_1st=ts_1st,
				ts_2nd=ts_2nd,
				ts_3rd=ts_3rd,
				ts_end=ts_end,
			)
			for tk in TOKENs_list
		]
		results = await asyncio.gather(*tasks)

		# Separating total NLF results and pages by year from the gathered results
		num_NLF_pages = [result[0] for result in results if result[0] is not None]
		NLF_pages_by_year_list = [result[1] for result in results if result[1] is not None]
		
		return num_NLF_pages, NLF_pages_by_year_list

def get_topK_tokens(
		mat_cols, 
		avgrec, 
		tok_query: List[str], 
		meaningless_lemmas_list: List[str], 
		raw_query: str="MonarKisti", 
		K: int=50, 
		ts_1st: int=1899, 
		ts_2nd=np.arange(1900, 1919+1, 1), 
		ts_3rd=np.arange(1920, 1945+1, 1), 
		ts_end: int=1946,
	):
	print(
		f"Looking for < topK={K} > token(s) from NLF REST API...\n"
		f"Query [raw]: {raw_query}\n"
		f"Query [tokenized]: {raw_query.lower().split()} | tk: {tok_query}"
	)
	st_t = time.time()
	topK_tokens_list = []
	for iTK in avgrec.argsort()[-K:]:
		recommended_token = mat_cols[iTK]
		if (
			recommended_token not in tok_query
			and recommended_token not in meaningless_lemmas_list
			and recommended_token not in raw_query.lower().split()
			and is_substring(A=raw_query, B=recommended_token) # evankelis luterilainen kirkko vs evankelisluterilainen
			# and recommended_token not in raw_query.lower() # reklamkampanj vs reklam | keskustapuolue vs puolue 
			# and raw_query.lower() not in recommended_token # tehdas vs rautatehdas
		):
			topK_tokens_list.append(recommended_token)
	tot_nlf_res_list, nlf_pages_by_year_list = asyncio.run(
		get_num_NLF_pages_asynchronous_run(
			qu=raw_query, 
			TOKENs_list=topK_tokens_list,
			ts_1st=ts_1st,
			ts_2nd=ts_2nd,
			ts_3rd=ts_3rd,
			ts_end=ts_end,
		)
	)
	###################################################################################################################
	# remove zeros: not time consuming...
	# print(f"Done=> removing zero(s)...")
	# rm_t = time.time()

	tot_nlf_res_list_tmp = tot_nlf_res_list
	topK_tokens_list_tmp = topK_tokens_list
	nlf_pages_by_year_list_tmp = nlf_pages_by_year_list

	tot_nlf_res_list = [num for num, word in zip(tot_nlf_res_list_tmp, topK_tokens_list_tmp) if (num and num != 0) ]
	topK_tokens_list = [word for num, word in zip(tot_nlf_res_list_tmp, topK_tokens_list_tmp) if (num and num != 0) ]
	nlf_pages_by_year_list = [yearly_pages for yearly_pages, tot_pages, tk in zip(nlf_pages_by_year_list_tmp, tot_nlf_res_list_tmp, topK_tokens_list_tmp) if (tot_pages and tot_pages != 0)]

	# print(len(topK_tokens_list), topK_tokens_list)
	# print(len(tot_nlf_res_list), tot_nlf_res_list)
	# print(len(nlf_pages_by_year_list), nlf_pages_by_year_list)
	# print(f"elp: {time.time()-rm_t:.5f} sec")
	###################################################################################################################

	###################################################################################################################
	# sort descending: not time consuming...
	sort_t = time.time()
	# print(f"=> sorting...")
	tot_nlf_res_list = tot_nlf_res_list[::-1]
	topK_tokens_list = topK_tokens_list[::-1]
	nlf_pages_by_year_list = nlf_pages_by_year_list[::-1]
	
	# print(len(topK_tokens_list), topK_tokens_list)
	# print(len(tot_nlf_res_list), tot_nlf_res_list)
	# print(len(nlf_pages_by_year_list), nlf_pages_by_year_list)

	# print(f"elp: {time.time()-sort_t:.5f} sec => DONE!!!")
	###################################################################################################################
	print(
		f"Found {len(topK_tokens_list)} Recommendation Results "
		f"(with {len(tot_nlf_res_list)} NLF pages) and separated: {len(nlf_pages_by_year_list)} "
		f"Elapsed: {time.time()-st_t:.2f} sec"
		.center(160, "-")
	)
	return topK_tokens_list, tot_nlf_res_list, nlf_pages_by_year_list

@cache
def get_recsys_results(query_phrase: str="A Sample query phrase!", nTokens: int=5, ts_1st: int=1899, ts_2nd=np.arange(1900, 1919+1, 1), ts_3rd=np.arange(1920, 1945+1, 1), ts_end: int=1946):
	tokenized_query_phrase = get_lemmatized_sqp(qu_phrase=query_phrase, lm=lmMethod)
	print(f"Search Query Prompt: {query_phrase} [lemma(s)]: {tokenized_query_phrase}")
	if not tokenized_query_phrase:
		return None, 0, 0

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
		return None, 0, 0
	ccs=get_customized_cosine_similarity_gpu(
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
	topK_TKs, topK_TKs_nlf_num_pages, topK_TKs_nlf_pages_by_year = get_topK_tokens(
		mat_cols=concat_spm_tokNames,
		avgrec=avgRecSys,
		raw_query=query_phrase,
		tok_query=tokenized_query_phrase,
		meaningless_lemmas_list=UNQ_STW,
		K=nTokens,
		ts_1st=ts_1st,
		ts_2nd=ts_2nd,
		ts_3rd=ts_3rd,
		ts_end=ts_end,
	)	
	return topK_TKs, topK_TKs_nlf_num_pages, topK_TKs_nlf_pages_by_year  # Returning the additional value
#######################################################################################################################

extract_tar(fname=compressed_spm_file)

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