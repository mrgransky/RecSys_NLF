from url_scraping import *
from utils import *

# how to run in background:
# nohup python -u nationalbiblioteket_logs.py --query 0 --saveDF True > logNEW_q0.out &

parser = argparse.ArgumentParser(description='National Library of Finland (NLF)')
parser.add_argument('-qlf', '--queryLogFile', required=True, type=str, help="Query log file") # smallest
parser.add_argument('--dsPath', required=True, type=str, help='Save DataFrame in directory | Def: None')

args = parser.parse_args()

def single_query(file_="", ts=None, browser_show=False):
	print(f">> Analyzing a single query of {file_}")
	st_t = time.time()
	#df = get_df_no_ip_logs(infile=file_, TIMESTAMP=ts)
	df = get_df_pseudonymized_logs(infile=file_, TIMESTAMP=ts)

	elapsed_t = time.time() - st_t
	print(f">> Elapsed_t: {elapsed_t:.2f} sec\tINITIAL df: {df.shape}\tavg search/s: {df.shape[0]/(24*60*60):.3f}")
	print("-"*100)
	
	#print(df.head(30))
	#print("-"*150)
	#print(df.tail(30))

	print(f">> Generating a sample query...")
	#qlink = int(np.random.randint(0, high=df.shape[0]+1, size=1))
	#qlink = np.random.choice(df.shape[0]+1)

	#qlink = 74399 # with quey word!
	qlink = 2500 # with quey word!
	########### file Q=6 ########### 
	#qlink = 1368231 # does not query words but results are given!
	#qlink = 91218 # does not query words
	#qlink = 54420 # going directly to OCR
	#qlink = 72120 # parsing elapsed time: 11.73 sec varies stochastically!

	########### file Q=0 ########### 
	#qlink = 52 # no "page="" in query
	#qlink = 1402 # google.fi # no "page="" in query
	#qlink = 5882 # ERROR due to login credintial
	#qlink = 277939 # MARC is not available for this page.
	#qlink = 231761 # 5151 # shows text content with ocr although txt_ocr returns False
	#qlink = 104106 # 55901 # indirect search from google => no "page="" in query
	#qlink = 227199 # 349904 # 6462 #  # with txt ocr available & "page="" in query
	#qlink = 103372 # broken link
	#qlink = 158 # short link
	#qlink = 21888 # broken connection
	#qlink = 30219 #30219 # 15033 #  # "" nothing in url
	#qlink = 96624 # both referer and user_agent pd.NA
	#qlink = 10340 # read timeout error in puhti when updating with session & head ?
	
	#print(df.iloc[qlink])
	s = time.time()
	
	single_url = df["referer"][qlink]
	print(f">> Q: {qlink} : {single_url}")

	r = checking_(single_url)
	if r is None:
		return

	single_url = r.url

	if browser_show:
		#print(f"\topening in browser... ")
		webbrowser.open(single_url, new=2)
	
	print(f"\n>> Parsing cleanedup URL: {single_url}")

	parsed_url, parameters = get_parsed_url_parameters(single_url)
	print(f"\n>> Parsed url:")
	print(parsed_url)

	print(f"\n>> Explore parsed url parameters:")
	print(parameters)

	# check for queries -> REST API:
	if parameters.get("query"):
		print(f">> QU: {parameters.get('query')}")
		my_query_word = ",".join( parameters.get("query") )
		#print(my_query_word)
		df.loc[qlink, "query_word"] = my_query_word

		#run_bash_script(param=parameters)

		#print("#"*65)
		#print(f"\tEXECUTE BASH REST API for {my_query_word}")
		#print("#"*65)
		#df.loc[qlink, "search_results"] = get_all_search_details(single_url)

	# term(ONLY IF OCR page):
	if parameters.get("term") and parameters.get("page"):
		print(parameters.get("term"))
		my_ocr_term = ",".join(parameters.get("term"))
		print(f"Saving OCR terms: {my_ocr_term}")
		df.loc[qlink, "ocr_term"] = my_ocr_term
		df.loc[qlink, "ocr_page"] = ",".join(parameters.get("page"))

		txt_pg_url = f"{parsed_url.scheme}://{parsed_url.netloc}{parsed_url.path}/page-{parameters.get('page')[0]}.txt"
		print(f">> page-X.txt available?\t{txt_pg_url}\t")

		text_response = checking_(txt_pg_url)
		if text_response is not None:
			print(f"\t\t\tYES >> loading...\n")
			df.loc[qlink, "ocr_text"] = text_response.text
	
	print(f"\n\n>> Parsing Completed!\tElapsed time: {time.time()-s:.2f} s\tFINAL df: {df.shape}")
	print(list(df.columns))
	print("-"*80)
	print(df.isna().sum())
	print("-"*80)
	print(f"Memory usage of each column in bytes (total column(s)={len(list(df.columns))})")
	print(df.memory_usage(deep=True, index=False, ))

	#print(df.head(30))
	#print("-"*150)
	#print(df.tail(30))
	#save_(df, infile=f"SINGLEQuery_timestamp_{ts}_{file_}")

def all_queries(file_: str=args.queryLogFile, ts: List[str]=None):
	print(f"Query Log File: {file_}")
	# check if queryLogFile.dump already exist: return
	log_file_name = file_[file_.rfind("/")+1:] # nike6.docworks.lib.helsinki.fi_access_log.2021-10-13.log
	if os.path.isfile(os.path.join(args.dsPath, f"{log_file_name}.dump")):
		print(f"{os.path.join(args.dsPath, log_file_name+'.dump')} already exist, exiting...")
		return

	st_t = time.time()
	df = get_df_pseudonymized_logs(infile=file_, TIMESTAMP=ts)
	print(f"DF_init Loaded in: {time.time()-st_t:.3f} sec | {df.shape}".center(100, " "))

	if df.shape[0] == 0:
		print(f"<!> Empty DF_init: {df.shape}, Nothing to retrieve, Stop Executing...")
		return

	print(f"{f'Page Analysis'.center(150, ' ')}\n"
				f"search pages: {df.referer.str.count('/search').sum()}, "
				f"collection pages: {df.referer.str.count('/collections').sum()}, "
				f"serial publication pages: {df.referer.str.count('/serial-publications').sum()}, "
				f"paper-for-day pages: {df.referer.str.count('/papers-for-day').sum()}, "
				f"clippings pages: {df.referer.str.count('/clippings').sum()}, "
				f"newspaper content pages: {df.referer.str.count('term=').sum()}, "
				#f"unknown pages: {df.referer.str.count('/collections').sum()}."
				)
	print("*"*150)
	
	parsing_t = time.time()
	print(f">> Scraping Newspaper Content Pages...")
	st_nwp_content_t = time.time()
	# df["nwp_content_referer"] = df[df.referer.str.contains('term=')]["referer"]
	df["nwp_content_referer"] = df.referer.map(lambda x: x if re.search(r'\/binding\/(\d+)', x) else np.nan, na_action="ignore",)
	df["nwp_content_results"] = df["nwp_content_referer"].map(scrap_newspaper_content_page, na_action='ignore')
	print(f"{f'Total Elapsed_t [Newspaper Content Pages]: {time.time()-st_nwp_content_t:.2f} s'.center(120, ' ')}")

	print(f">> Scraping Query Search Pages...")
	st_search_t = time.time()
	df["search_referer"] = df[df.referer.str.contains('/search')]["referer"]
	df["search_query_phrase"] = df["search_referer"].map(get_query_phrase, na_action='ignore')
	df["search_results"] = df["search_referer"].map(scrap_search_page, na_action='ignore')
	print(f"{f'Total Elapsed_t [Query Search Pages]: {time.time()-st_search_t:.2f} s'.center(120, ' ')}")

	print(f">> Scraping Collection Pages...")
	st_collection_t = time.time()
	df["collection_referer"] = df[df.referer.str.contains('/collections')]["referer"]
	df["collection_query_phrase"] = df["collection_referer"].map(get_query_phrase, na_action='ignore')
	df["collection_results"] = df["collection_referer"].map(scrap_collection_page, na_action='ignore')
	print(f"{f'Total Elapsed_t [Collection Pages]: {time.time()-st_collection_t:.2f} s'.center(120, ' ')}")

	print(f">> Scraping Clipping Pages...")
	st_clipping_t = time.time()
	df["clipping_referer"] = df[df.referer.str.contains('/clippings')]["referer"]
	df["clipping_query_phrase"] = df["clipping_referer"].map(get_query_phrase, na_action='ignore')
	df["clipping_results"] = df["clipping_referer"].map(scrap_clipping_page, na_action='ignore')
	print(f"{f'Total Elapsed_t [Clipping Pages]: {time.time()-st_clipping_t:.2f} s'.center(120, ' ')}")

	print("#"*100)
	print(f"Total Parsing Elapsed_t: {time.time()-parsing_t:.2f} s | DF: {df.shape}")
	print("<>"*50)

	print(f"Memory usage of each column in bytes (total column(s)={len(list(df.columns))})")
	print(df.memory_usage(deep=True, index=False, ))
	print("-"*100)

	print(df.info(verbose=True, memory_usage="deep", show_counts=True, ))
	
	save_pickle( pkl=df, fname=os.path.join( args.dsPath, f'{log_file_name}.dump') )

def run():
	make_folder(folder_name=args.dsPath)
	# all_log_files = [lf[ lf.rfind("/")+1: ] for lf in natsorted( glob.glob( os.path.join(dpath, "*.log") ) )]
	"""	
	# run single log file	
	single_query(file_=args.queryLogFile,
							#browser_show=True, 
							#ts=["23:52:00", "23:59:59"],
							)
	"""		
	# run all log files using array in batch
	all_queries(
		file_=args.queryLogFile,
		#ts=["14:30:00", "14:56:59"],
	)

if __name__ == '__main__':
	# os.system('clear')
	print(f"Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}".center(120, " "))
	run()
	print(f"Finished: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}".center(120, " "))
