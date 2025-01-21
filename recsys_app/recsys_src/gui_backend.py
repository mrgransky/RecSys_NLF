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
import warnings
warnings.filterwarnings("ignore")

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

def get_device():
	if torch.cuda.is_available():
		cuda_id = 3 if USER == "ubuntu" else 0
		device = torch.device(f"cuda:{cuda_id}")
	else:
		device = torch.device("cpu")
	print(f"Using device: {device}")
	return device, cuda_id

def get_dynamic_batch_size(spMtx, memory_fraction=0.9, gpu_id=0):
	"""
	Calculate optimal batch size based on available GPU memory and matrix size.
	
	Args:
		spMtx: The sparse matrix to be processed
		memory_fraction: Fraction of available GPU memory to use (default: 0.9)
		gpu_id: ID of the GPU cuda device to use (default: 0)
	
	Returns:
		int: Calculated batch size
	"""
	try:
		# Get GPU memory information
		device = cp.cuda.Device(gpu_id)
		free_memory = device.mem_info[0]  # Free memory in bytes
		total_memory = device.mem_info[1]  # Total memory in bytes
		# Calculate memory available for processing
		available_memory = free_memory * memory_fraction
		# Estimate memory requirements per row
		# Assuming float32 (4 bytes) for all numerical values
		bytes_per_float = 4
		avg_nonzeros_per_row = spMtx.nnz / spMtx.shape[0]		
		# Estimate memory needed per row:
		# 1. Sparse matrix row (indices + data)
		# 2. Dense array allocations
		# 3. Intermediate calculations
		memory_per_row = (
			(avg_nonzeros_per_row * 2 * bytes_per_float) +  # Sparse matrix data + indices
			(spMtx.shape[1] * bytes_per_float) +            # Dense array allocations
			(avg_nonzeros_per_row * bytes_per_float * 3)    # Intermediate calculations buffer
		)
		# Calculate batch size
		batch_size = int(available_memory / memory_per_row)
		# Set minimum and maximum batch sizes
		min_batch_size = 256
		max_batch_size = 1024
		batch_size = max(min_batch_size, min(batch_size, max_batch_size))
		# Round to nearest power of 2 for better memory alignment
		batch_size = 2 ** int(np.log2(batch_size))		
		# print(f"Calculated batch size: {batch_size}")
		# print(f"Available GPU memory: {free_memory / 1024**3:.2f} GB")
		# print(f"Memory per row estimate: {memory_per_row / 1024**2:.2f} MB")
		return batch_size
	except Exception as e:
		print(f"Error calculating batch size: {str(e)}")
		return 128 # Return a conservative default batch size

def get_device_with_most_free_memory():
	if torch.cuda.is_available():
		# print(f"Available GPU(s)| torch = {torch.cuda.device_count()} | CuPy: {cp.cuda.runtime.getDeviceCount()}")
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
		selected_device = None
		print("No GPU available ==>> using CPU")
	return device, selected_device

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

def get_lemmatized_sqp(qu_phrase, lm: str="stanza", device:str="cuda:0"):
	cleaned_phrase = clean_(docs=qu_phrase) 
	return lemmatizer_methods.get(lm)( docs=cleaned_phrase, device=device )

def get_query_vec(mat, mat_row, mat_col, tokenized_qu_phrases=["åbo", "akademi"]):
	query_vector=np.zeros((1, mat.shape[1]), dtype=np.float32)
	query_vector[0, list(np.in1d(mat_col, tokenized_qu_phrases).nonzero()[0])]=1
	# print(query_vector.shape, query_vector.dtype, np.count_nonzero(query_vector), np.where(query_vector.flatten()!=0)[0])
	#print(np.argsort(tempquery.flatten())[-len(query_words):])
	# print(np.where(query_vector.flatten()!=0)[0])
	return query_vector

def get_customized_cosine_similarity(spMtx, query_vec, idf_vec, spMtx_norm, exponent:float=1.0,):
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

def get_customized_recsys_avg_vec(spMtx, cosine_sim, idf_vec, spMtx_norm,):
	print(f"avgRecSys (1 x nTKs={spMtx.shape[1]})".center(130, "-"))
	st_t = time.time()
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

def get_customized_cosine_similarity_gpu(spMtx, query_vec, idf_vec, spMtx_norm, exponent:float=1.0,):
	device, gpu_id = get_device_with_most_free_memory()
	batch_size = get_dynamic_batch_size(spMtx=spMtx, gpu_id=gpu_id)
	try:
		print(f"[GPU Optimized] Customized Cosine Similarity (1 x nUsers={spMtx.shape[0]})".center(150, "-"))
		cp.get_default_memory_pool().free_all_blocks()
		torch.cuda.empty_cache()
		device = cp.cuda.Device(gpu_id)
		device.synchronize()
		device_id = device.id
		device_name = cp.cuda.runtime.getDeviceProperties(device_id)['name'].decode('utf-8')
		print(
			f"GPU [cuda:{device_id}]: {device_name} "
			f"Memory [Free/Total]: ({device.mem_info[0] / 1024 ** 3:.3f} / {device.mem_info[1] / 1024 ** 3:.3f}) GB"
		)
		print(
			f"Query: {query_vec.shape} {type(query_vec)} {query_vec.dtype} non_zeros={np.count_nonzero(query_vec)} (ratio={np.count_nonzero(query_vec) / query_vec.size})\n"
			f"spMtx {type(spMtx)} {spMtx.shape} {spMtx.dtype}\n"
			f"spMtxNorm: {type(spMtx_norm)} {spMtx_norm.shape} {spMtx_norm.dtype}\n"
			f"IDF {type(idf_vec)} {idf_vec.shape} {idf_vec.dtype}\n"
			f"[Dynamic] batch size: {batch_size}"
		)
		st_t = time.time()
		# Convert inputs to GPU with memory management
		with device:
			query_vec_squeezed = cp.asarray(query_vec.ravel(), dtype=cp.float32)
			idf_squeezed = cp.asarray(idf_vec.ravel(), dtype=cp.float32)
			spMtx_norm_gpu = cp.asarray(spMtx_norm, dtype=cp.float32)
			# Convert sparse matrix efficiently
			spMtx_csr = spMtx.tocsr()
			spMtx_gpu = cp.sparse.csr_matrix(
				(
					cp.asarray(spMtx_csr.data, dtype=cp.float32),
				 	cp.asarray(spMtx_csr.indices),
				 	cp.asarray(spMtx_csr.indptr)
				),
				shape=spMtx_csr.shape
			)
			del spMtx_csr # # Free CPU memory
			# Compute interest and normalization
			quInterest = query_vec_squeezed * idf_squeezed
			quInterestNorm = cp.linalg.norm(quInterest)
			idx_nonzeros = cp.nonzero(quInterest)[0]
			quInterest_nonZeros = quInterest[idx_nonzeros] / quInterestNorm
			usrInterestNorm = spMtx_norm_gpu + cp.float32(1e-4)
			del query_vec_squeezed, quInterest # Free unnecessary arrays
			cs = cp.zeros(spMtx_gpu.shape[0], dtype=cp.float32) # initialize result array
			for i in range(0, spMtx_gpu.shape[0], batch_size):
					start_idx = i
					end_idx = min(i + batch_size, spMtx_gpu.shape[0])
					# Process batch
					spMtx_batch = spMtx_gpu[start_idx:end_idx, idx_nonzeros]
					spMtx_batch = spMtx_batch.multiply(idf_squeezed[idx_nonzeros])
					spMtx_batch = spMtx_batch.multiply(1 / usrInterestNorm[start_idx:end_idx, None])
					if exponent != 1.0:
							spMtx_batch.data **= exponent
					cs[start_idx:end_idx] = spMtx_batch.dot(quInterest_nonZeros)
					# Clean up batch memory
					del spMtx_batch
					cp.get_default_memory_pool().free_all_blocks()
					torch.cuda.empty_cache()
			# Get result and clean up
			result = cp.asnumpy(cs)
			# Final cleanup
			del cs, spMtx_gpu, idf_squeezed, spMtx_norm_gpu, quInterest_nonZeros, usrInterestNorm
			cp.get_default_memory_pool().free_all_blocks()
			torch.cuda.empty_cache()
		print(f"Elapsed_t: {time.time() - st_t:.2f} s".center(150, " "))
		return result
	except Exception as e:
		print(f"Error in cosine similarity calculation: {str(e)}")
		raise
	finally:
		# Ensure memory cleanup on exit
		cp.get_default_memory_pool().free_all_blocks()
		torch.cuda.empty_cache()

def get_customized_recsys_avg_vec_gpu(spMtx, cosine_sim, idf_vec, spMtx_norm,):
	try:
		print(f"[GPU optimized] avgRecSys (1 x nTKs={spMtx.shape[1]})".center(150, "-"))
		device, gpu_id = get_device_with_most_free_memory()
		batch_size = get_dynamic_batch_size(spMtx=spMtx, gpu_id=gpu_id)
		st_t = time.time()
		torch.cuda.empty_cache()
		cp.get_default_memory_pool().free_all_blocks()
		device = cp.cuda.Device(gpu_id)
		device.synchronize()
		device_id = device.id
		device_name = cp.cuda.runtime.getDeviceProperties(device_id)['name'].decode('utf-8')
		print(
			f"GPU [cuda:{device_id}]: {device_name} "
			f"Memory [Free/Total]: ({device.mem_info[0] / 1024 ** 3:.3f} / {device.mem_info[1] / 1024 ** 3:.3f}) GB"
		)
		with device:
			idf_squeezed = cp.asarray(idf_vec.ravel(), dtype=cp.float32)
			cosine_sim_gpu = cp.asarray(cosine_sim, dtype=cp.float32)
			spMtx_norm_gpu = cp.asarray(spMtx_norm, dtype=cp.float32)
			non_zero_cosines = cp.nonzero(cosine_sim_gpu)[0]
			non_zero_values = cosine_sim_gpu[non_zero_cosines]
			print(
				f"spMtx {type(spMtx)} {spMtx.shape} {spMtx.dtype}\n"
				f"spMtxNorm: {type(spMtx_norm)} {spMtx_norm.shape} {spMtx_norm.dtype}\n"
				f"CS {type(cosine_sim)} {cosine_sim.shape} {cosine_sim.dtype} NonZero(s): {non_zero_cosines.shape[0]}\n"
				f"IDF {type(idf_vec)} {idf_vec.shape} {idf_vec.dtype}\n"
				f"[Dynamic] batch size: {batch_size}"
			)
			del cosine_sim_gpu
			spMtx_csr = spMtx.tocsr()
			spMtx_gpu = cp.sparse.csr_matrix(
				(
					cp.asarray(spMtx_csr.data, dtype=cp.float32),
				 	cp.asarray(spMtx_csr.indices),
				 	cp.asarray(spMtx_csr.indptr)
				),
				shape=spMtx_csr.shape
			)
			del spMtx_csr
			avg_rec = cp.zeros(spMtx.shape[1], dtype=cp.float32)
			for i in range(0, len(non_zero_cosines), batch_size):
				batch_indices = non_zero_cosines[i:i + batch_size]
				batch_values = non_zero_values[i:i + batch_size]
				spMtx_batch = spMtx_gpu[batch_indices]
				batch_result = spMtx_batch.multiply(idf_squeezed)
				
				norm_factors = spMtx_norm_gpu[batch_indices] + cp.float32(1e-18)
				batch_result = batch_result.multiply(1.0 / norm_factors[:, None])
				batch_result = batch_result.multiply(batch_values[:, None])
				
				avg_rec += batch_result.sum(axis=0).ravel()
				# Clean up batch memory
				del spMtx_batch, batch_result
				cp.get_default_memory_pool().free_all_blocks()
			
			# Normalize the result
			sum_non_zero_values = cp.sum(non_zero_values)
			avg_rec /= sum_non_zero_values
			# Convert back to CPU and clean up GPU memory
			result = cp.asnumpy(avg_rec)
			# Final cleanup
			del avg_rec, spMtx_gpu, idf_squeezed, spMtx_norm_gpu
			del non_zero_cosines, non_zero_values
			cp.get_default_memory_pool().free_all_blocks()
			torch.cuda.empty_cache()
		print(f"Elapsed_t: {time.time()-st_t:.2f} s {type(result)} {result.dtype} {result.shape}".center(150, " "))
		return result
	except Exception as e:
		print(f"Error in average recommendation calculation: {str(e)}")
		raise
	finally:
		# Ensure memory cleanup on exit
		cp.get_default_memory_pool().free_all_blocks()
		torch.cuda.empty_cache()

# def count_years_by_range(yr_vs_nPGs: Dict[str, int], ts_1st: int=1899, ts_2nd=np.arange(1900, 1919+1, 1), ts_3rd=np.arange(1920, 1945+1, 1), ts_end: int=1946):
# 	first_range = 0
# 	second_range = 0
# 	third_range = 0
# 	forth_range = 0
# 	if not yr_vs_nPGs:
# 		return [0, 0, 0, 0]	
# 	for year, count in yr_vs_nPGs.items():
# 		year_int = int(year)
# 		if year_int <= ts_1st: # green
# 			first_range += count
# 		elif year_int in ts_2nd: # WWI: 28 july 1914 – 11 nov. 1918 # pink
# 			second_range += count
# 		elif year_int in ts_3rd: # WWII: 1 sep. 1939 – 2 sep. 1945 # blue
# 			third_range += count
# 		elif year_int >= ts_end: # red
# 			forth_range += count
# 	yearly_nlf_pages = [first_range, second_range, third_range, forth_range]
# 	return yearly_nlf_pages

def count_years_by_range(yr_vs_nPGs: Dict[str, int], ts_1st: int = 1899, ts_2nd=None, ts_3rd=None, ts_end: int = 1946):
		if not yr_vs_nPGs:
				return [0, 0, 0, 0]

		# Convert data to NumPy arrays for vectorized operations
		years = np.array(list(map(int, yr_vs_nPGs.keys())))
		counts = np.array(list(yr_vs_nPGs.values()))

		# Calculate ranges using boolean masks
		first_range = counts[years <= ts_1st].sum()
		second_range = counts[np.isin(years, ts_2nd)].sum() if ts_2nd is not None else 0
		third_range = counts[np.isin(years, ts_3rd)].sum() if ts_3rd is not None else 0
		forth_range = counts[years >= ts_end].sum()

		return [first_range, second_range, third_range, forth_range]

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

# def get_topK_tokens(
# 		mat_cols, 
# 		avgrec, 
# 		tok_query: List[str], 
# 		meaningless_lemmas_list: List[str], 
# 		raw_query: str="MonarKisti", 
# 		K: int=50,
# 		ts_1st: int=1899, 
# 		ts_2nd=np.arange(1900, 1919+1, 1), 
# 		ts_3rd=np.arange(1920, 1945+1, 1), 
# 		ts_end: int=1946,
# 	):
# 	print(
# 		f"Searching top-{K} token(s) in NLF REST API...\n"
# 		f"Query\n"
# 		f"\t«raw»: {raw_query}\n"
# 		f"\t«split listed»: {raw_query.lower().split()}\n"
# 		f"\t«tokenized»: {tok_query}"
# 	)
# 	st_t = time.time()
# 	topK_tokens_list = []
# 	for iTK in avgrec.argsort()[-K:]:
# 		recommended_token = mat_cols[iTK]
# 		if (
# 			recommended_token not in tok_query
# 			and recommended_token not in meaningless_lemmas_list
# 			and recommended_token not in raw_query.lower().split() # 
# 			and is_substring(A=raw_query, B=recommended_token)
# 			# and recommended_token not in raw_query.lower() # reklamkampanj vs reklam | keskustapuolue vs puolue 
# 			# and raw_query.lower() not in recommended_token # tehdas vs rautatehdas
# 		):
# 			topK_tokens_list.append(recommended_token)
	
# 	tot_nlf_res_list, nlf_pages_by_year_list = asyncio.run(
# 		get_num_NLF_pages_asynchronous_run(
# 			qu=raw_query, 
# 			TOKENs_list=topK_tokens_list,
# 			ts_1st=ts_1st,
# 			ts_2nd=ts_2nd,
# 			ts_3rd=ts_3rd,
# 			ts_end=ts_end,
# 		)
# 	)
# 	###################################################################################################################
# 	# remove zeros: not time consuming...
# 	# print(f"Done=> removing zero(s)...")
# 	# rm_t = time.time()

# 	tot_nlf_res_list_tmp = tot_nlf_res_list
# 	topK_tokens_list_tmp = topK_tokens_list
# 	nlf_pages_by_year_list_tmp = nlf_pages_by_year_list

# 	tot_nlf_res_list = [num for num, word in zip(tot_nlf_res_list_tmp, topK_tokens_list_tmp) if (num and num != 0) ]
# 	topK_tokens_list = [word for num, word in zip(tot_nlf_res_list_tmp, topK_tokens_list_tmp) if (num and num != 0) ]
# 	nlf_pages_by_year_list = [yearly_pages for yearly_pages, tot_pages, tk in zip(nlf_pages_by_year_list_tmp, tot_nlf_res_list_tmp, topK_tokens_list_tmp) if (tot_pages and tot_pages != 0)]

# 	# print(len(topK_tokens_list), topK_tokens_list)
# 	# print(len(tot_nlf_res_list), tot_nlf_res_list)
# 	# print(len(nlf_pages_by_year_list), nlf_pages_by_year_list)
# 	# print(f"elp: {time.time()-rm_t:.5f} sec")
# 	###################################################################################################################

# 	###################################################################################################################
# 	# sort descending: not time consuming...
# 	# sort_t = time.time()
# 	# print(f"=> sorting...")
# 	tot_nlf_res_list = tot_nlf_res_list[::-1]
# 	topK_tokens_list = topK_tokens_list[::-1]
# 	nlf_pages_by_year_list = nlf_pages_by_year_list[::-1]
	
# 	# print(len(topK_tokens_list), topK_tokens_list)
# 	# print(len(tot_nlf_res_list), tot_nlf_res_list)
# 	# print(len(nlf_pages_by_year_list), nlf_pages_by_year_list)

# 	# print(f"elp: {time.time()-sort_t:.5f} sec => DONE!!!")
# 	###################################################################################################################
# 	print(
# 		f"Found {len(topK_tokens_list)} Recommendation Results "
# 		f"(with {len(tot_nlf_res_list)} NLF pages) and separated: {len(nlf_pages_by_year_list)} "
# 		f"Elapsed: {time.time()-st_t:.2f} sec"
# 		.center(160, "-")
# 	)
# 	return topK_tokens_list, tot_nlf_res_list, nlf_pages_by_year_list

def get_topK_tokens(
		mat_cols,
		avgrec,
		tok_query: List[str],
		meaningless_lemmas_list: List[str],
		raw_query: str = "MonarKisti",
		K: int = 50,
		ts_1st: int = 1899,
		ts_2nd=np.arange(1900, 1919 + 1, 1),
		ts_3rd=np.arange(1920, 1945 + 1, 1),
		ts_end: int = 1946,
):
	print(
		f"Searching top-{K} token(s) in NLF REST API...\n"
		f"Query\n"
		f"\t«raw»: {raw_query}\n"
		f"\t«split listed»: {raw_query.lower().split()}\n"
		f"\t«tokenized»: {tok_query}"
	)
	st_t = time.time()
	# Filter tokens efficiently
	topK_tokens_list = [
		mat_cols[iTK]
		for iTK in avgrec.argsort()[-K:]
		if (
			mat_cols[iTK] not in tok_query
			and mat_cols[iTK] not in meaningless_lemmas_list
			and mat_cols[iTK] not in raw_query.lower().split()
			and is_substring(A=raw_query, B=mat_cols[iTK])  # evankelis luterilainen kirkko vs evankelisluterilainen
			# and mat_cols[iTK] not in raw_query.lower() # reklamkampanj vs reklam | keskustapuolue vs puolue 
			# and raw_query.lower() not in mat_cols[iTK] # tehdas vs rautatehdas
		)
	]
	# Fetch NLF pages in parallel
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
	# Filter results where NLF pages exist
	filtered_results = [
		(tk, nlf, yearly_pages)
		for tk, nlf, yearly_pages in zip(topK_tokens_list, tot_nlf_res_list, nlf_pages_by_year_list)
		if nlf and nlf != 0
	]
	# Unpack filtered results
	topK_tokens_list, tot_nlf_res_list, nlf_pages_by_year_list = zip(*filtered_results) if filtered_results else ([], [], [])
	# Reverse results for descending order
	topK_tokens_list = list(reversed(topK_tokens_list))
	tot_nlf_res_list = list(reversed(tot_nlf_res_list))
	nlf_pages_by_year_list = list(reversed(nlf_pages_by_year_list))
	print(
		f"Found {len(topK_tokens_list)} Recommendation Results "
		f"(with {len(tot_nlf_res_list)} NLF pages) and separated: {len(nlf_pages_by_year_list)} "
		f"Elapsed: {time.time()-st_t:.2f} sec"
		.center(160, "-")
	)
	return topK_tokens_list, tot_nlf_res_list, nlf_pages_by_year_list

@cache
def get_recsys_results(
	query_phrase: str="A Sample query phrase!",
	nTokens:int=5,
	ts_1st:int=1899,
	ts_2nd=np.arange(1900, 1919+1, 1),
	ts_3rd=np.arange(1920, 1945+1, 1),
	ts_end: int=1946,
):
	device, _ = get_device_with_most_free_memory()
	tokenized_query_phrase = get_lemmatized_sqp(qu_phrase=query_phrase, lm=lmMethod, device=device)
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

	avgRecSys = get_customized_recsys_avg_vec_gpu(
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