from utils import *
from collections import Counter

parser = argparse.ArgumentParser(description='National Library of Finland (NLF) RecSys')
parser.add_argument('--inputDF', default=f"{os.path.expanduser('~')}/Datasets/Nationalbiblioteket/dataframes/nikeY.docworks.lib.helsinki.fi_access_log.07_02_2021.log.dump", type=str) # smallest
parser.add_argument('--qusr', default="ip69", type=str)
parser.add_argument('--qtip', default="Kristiinan Sanomat_77 A_1", type=str) # smallest
parser.add_argument('--qphrase', default="ystävä", type=str) # smallest

args = parser.parse_args()

# how to run:
# python RecSys.py --inputDF ~/Datasets/Nationalbiblioteket/dataframes/nikeY.docworks.lib.helsinki.fi_access_log.07_02_2021.log.dump

def topN_nwp_title_issue_page(nwp_tip, sim_df, N=10):
		if nwp_tip not in sim_df.index:
				print(f"Error: Newspaper `{nwp_tip}` not Found!")
				return
		print(f"Top-{N} Newspaper similar to `{nwp_tip}`:")
		sim_df = sim_df.drop(nwp_tip)
		similar_newspapers = list(sim_df.sort_values(by=nwp_tip, ascending=False).index[1: N+1])
		similarity_values = list(sim_df.sort_values(by=nwp_tip, ascending=False).loc[:, nwp_tip])[1:N+1]
		for sim_nwp, sim_val in zip(similar_newspapers, similarity_values):
			print(f"\t{sim_nwp} : {sim_val:.3f}")

def topN_users(usr, sim_df, dframe, N=5):
	if usr not in sim_df.index:
		print(f"User `{usr}` not Found!\tTry another user_ip next time...")
		return
	
	print(f"{'Query USER Search History Phrases'.center(100,'-')}")
	qu_usr_search_history = get_query_user_details(usr_q=usr, dframe=dframe)
	if qu_usr_search_history is None:
		print(f"You have not searched for any specific words/phrases yet... "
					f"=> No recommendation is available at the moment! "
					f"Please continue searching...")
		return
	print(len(qu_usr_search_history), qu_usr_search_history)
	
	print(f"Top-{N} similar users to `{usr}`:")

	print(sim_df.head(20))
	print(f'<>'*50)

	similar_users = list(sim_df.sort_values(by=usr, ascending=False).index[1:]) # excluding usr
	similarity_values = list(sim_df.sort_values(by=usr, ascending=False).loc[:, usr])[1:] # excluding usr
	#similar_users = list(sim_df.sort_values(by=usr, ascending=False).index[1: N+1])
	#similarity_values = list(sim_df.sort_values(by=usr, ascending=False).loc[:, usr])[1: N+1]

	#print(sim_df.sort_values(by=usr, ascending=False).head(20))

	print(len(similar_users), similar_users[:20])
	print(len(similarity_values), similarity_values[:20])
	print("#"*100)
	
	print(f"{f'Similar USER(s) to {usr} | Detail'.center(100,'-')}")
	similar_users_search_phrases_history = get_similar_users_details(similar_users, similarity_values, dframe=dframe, TopN=N)

	print(f"Recommendation Result (Implicit Feedback)".center(100,'-'))
	print(f"Since you {Fore.RED}\033[1m{usr}\033[0m{Style.RESET_ALL} "
				f"searched for {len(qu_usr_search_history)} Query Phrase(s):\n"
				f"{Fore.BLUE}{qu_usr_search_history}{Style.RESET_ALL}\n"
				f"you might also be interested in Phrases:\n{Fore.GREEN}{similar_users_search_phrases_history}{Style.RESET_ALL}")
	print(f"Recommendation Result (Implicit Feedback)".center(100,'-'))

def get_similar_users_details(sim_users_list, similarity_vals, dframe, TopN=5):
	searched_phrases = list()
	retrieved_users = list()
	retrieved_similarity_values = list()

	for usr_i, usr_v in enumerate(sim_users_list):
		print(f"{'QUERY Phrases'.center(50,'-')}")
		print(usr_i, usr_v)
		qu_phs = ["".join(elem) for elem in dframe[(dframe["user_ip"] == usr_v)][["search_query_phrase"]].values.tolist()]
		print(f"All qu phrases: {len(qu_phs)}")
		unq_qu_phs_dict = Counter(qu_phs)
		print(json.dumps(unq_qu_phs_dict, indent=2, ensure_ascii=False))
		unq_qu_phs_dict.pop("", None)

		if len(unq_qu_phs_dict) > 0:
			#print(max(unq_qu_phs_dict, key=unq_qu_phs_dict.get))
			searched_phrases.append(max(unq_qu_phs_dict, key=unq_qu_phs_dict.get))
			retrieved_users.append(usr_v)
			retrieved_similarity_values.append(similarity_vals[usr_i])
		else:
			print(f"\t<!> Useless user!! Trying next user...")
		print()
		if len(searched_phrases) >= TopN:
			print(f"We found Top-{TopN} results => leave for loop..")
			break
	print(f">> search hist: {searched_phrases}")
	for sim_usr, sim_val, usr_hist in zip(retrieved_users, retrieved_similarity_values, searched_phrases):
		print(f"\t{sim_usr} : {sim_val:.4f}\t(Top Searched Query Phrase: {usr_hist})")
	print()

	return searched_phrases

def get_query_user_details(usr_q, dframe):
	qu_phs = ["".join(elem) for elem in dframe[(dframe["user_ip"] == usr_q)][["search_query_phrase"]].values.tolist()]
	print(f"|phrases|: {len(qu_phs)}")
	
	unq_qu_phs_dict = Counter(qu_phs)

	print(json.dumps(unq_qu_phs_dict, indent=2, ensure_ascii=False))
	unq_qu_phs_dict.pop("", None)
	if len(unq_qu_phs_dict) > 0:
		#print(max(unq_qu_phs_dict, key=unq_qu_phs_dict.get))
		#return max(unq_qu_phs_dict, key=unq_qu_phs_dict.get) # only the one with max occurance!
		return list(unq_qu_phs_dict.keys()) # all searched words/phrases!
	else:
		return

def get_basic_RecSys(df, user_name=args.qusr, nwp_title_issue_page_name=args.qtip):
	#print(df.head(30))
	print(f"Search".center(80, "-"))
	df_search = pd.DataFrame()
	df_search["user_ip"] = df.loc[ df['search_results'].notnull(), ['user_ip'] ]

	df_search["search_query_phrase"] = df.loc[ df['search_results'].notnull(), ['search_query_phrase'] ]#.fillna("")
	#df_search['search_query_phrase'] = [','.join(map(str, elem)) if elem else '' for elem in df_search['search_query_phrase']]
	
	df_search["title_issue_page"] = df["search_results"].map(get_search_title_issue_page, na_action='ignore')
	df_search["snippet_highlighted_words"] = df["search_results"].map(get_snippet_hw_counts, na_action='ignore')
	df_search["referer"] = df.loc[ df['search_results'].notnull(), ['referer'] ]

	df_search = df_search.explode(["title_issue_page", "snippet_highlighted_words"],
																ignore_index=True,
																)
	df_search['snippet_highlighted_words'] = df_search['snippet_highlighted_words'].apply(pd.to_numeric)

	with pd.option_context('display.max_rows', 300, 'display.max_colwidth', 1500):
		print(df_search.head(50))

	print("<>"*50)
	print(f"<Nones> tip: {df_search['title_issue_page'].isna().sum()} | "
				f"qu_phrases: {df_search['search_query_phrase'].isna().sum()} "
				f"total DF: {df_search.shape}")
	print(df_search.info(verbose=True, memory_usage="deep"))
	#return

	print(f"Newspaper Content".center(80, "-"))
	df_content = pd.DataFrame()
	df_content["user_ip"] = df.loc[ df['nwp_content_results'].notnull(), ['user_ip'] ]
	df_content["title_issue_page"] = df["nwp_content_results"].map(get_content_title_issue_page, na_action='ignore')
	df_content["content_highlighted_words"] = df["nwp_content_results"].map(get_content_hw_counts, na_action='ignore')
	df_content['content_highlighted_words'] = df_content['content_highlighted_words'].apply(pd.to_numeric)
	#df_content["referer"] = df.loc[ df['nwp_content_results'].notnull(), ['referer'] ]
	df_content = df_content.reset_index(drop=True)
	print(df_content.head(15))
	print(df_content.info(verbose=True, memory_usage="deep"))
	
	"""
	print(f"Merging".center(80, "-"))
	df_merged = pd.merge(df_search, # left
										df_content, # right
										how='outer', 
										#on=['user_ip','title_issue_page'],
										#on=['user_ip',],
										on=['title_issue_page',],
										suffixes=['_l', '_r'],
										)

	df_merged = df_merged.fillna({'snippet_highlighted_words': 0, 
													'content_highlighted_words': 0,
													}
												)

	df_merged["implicit_feedback"] = (0.5 * df_merged["snippet_highlighted_words"]) + df_merged["content_highlighted_words"]

	print(df_merged.shape)
	print(df_merged["title_issue_page"].isna().sum())
	print(df_merged.head(20))
	print(df_merged.info(verbose=True, memory_usage="deep"))
	print("<>"*100)
	
	#print(df_merged[df_merged['implicit_feedback'].notnull()].tail(60))
	print(f"< unique > tip: {len(df_merged['title_issue_page'].unique())}")
	print(f"< unique > user_ip_l: {len(df_merged['user_ip_l'].unique())}")
	print(f"< unique > user_ip_r: {len(df_merged['user_ip_r'].unique())}")
	"""

	print(f"Concatinating".center(80, "-"))
	df_concat = pd.concat([df_search, df_content],)

	df_concat = df_concat.fillna({'snippet_highlighted_words': 0, 
																'content_highlighted_words': 0,
																'search_query_phrase': '',
																}
															)
	df_concat['search_query_phrase'] = [','.join(map(str, elem)) for elem in df_concat['search_query_phrase']]

	df_concat["implicit_feedback"] = (0.5 * df_concat["snippet_highlighted_words"]) + df_concat["content_highlighted_words"]
	df_concat["implicit_feedback"] = df_concat["implicit_feedback"].astype(np.float32)

	df_concat["nwp_tip_index"] = df_concat["title_issue_page"].fillna('UNAVAILABLE').astype("category").cat.codes
	df_concat["user_index"] = df_concat["user_ip"].astype("category").cat.codes


	print(df_concat.shape)
	print(df_concat["title_issue_page"].isna().sum())
	print(df_concat.head(20))
	print(df_concat.info(verbose=True, memory_usage="deep"))
	print("<>"*100)

	print(f"< unique > users: {len(df_concat['user_index'].unique())} | " 
				f"title_issue_page: {len(df_concat['nwp_tip_index'].unique())} "
				f"=> sparse matrix: {len(df_concat['user_index'].unique()) * len(df_concat['nwp_tip_index'].unique())}"
				)

	imp_fb_sparse_matrix = get_sparse_mtx(df_concat)
	print(f">> implicit feedback sparse matrix: {imp_fb_sparse_matrix.shape}")

	st_t = time.time()
	usr_similarity_df = get_similarity_df(df_concat, 
																				imp_fb_sparse_matrix, 
																				method="user-based", 
																				result_dir=make_result_dir(infile=args.inputDF),
																				)

	print(f"<<>> User-based Similarity DF: {usr_similarity_df.shape}\tElapsed_t: {time.time()-st_t:.2f} s")

	topN_users(usr=user_name, sim_df=usr_similarity_df, dframe=df_concat)
	print("<>"*50)

	#return
	st_t = time.time()
	itm_similarity_df = get_similarity_df(df_concat, 
																				imp_fb_sparse_matrix.T, 
																				method="item-based",
																				result_dir=make_result_dir(infile=args.inputDF),
																				)
	print(f"<<>> Item-based Similarity DF: {itm_similarity_df.shape}\tElapsed_t: {time.time()-st_t:.2f} s")

	#topN_nwp_title_issue_page("Karjalatar_135_2", itm_similarity_df)
	topN_nwp_title_issue_page(nwp_tip=nwp_title_issue_page_name, sim_df=itm_similarity_df)
	print("-"*70)

def run_RecSys(df):
	print_df_detail(df, fname=__file__)
	get_basic_RecSys(df)

def main():
	df = load_df(infile=args.inputDF)
	"""
	print(f"DF: {df.shape}")
	print("%"*140)
	cols = list(df.columns)
	print(len(cols), cols)
	print("#"*150)

	print(df.head(10))
	print("-"*150)
	print(df.tail(10))

	print(df.isna().sum())
	print("-"*150)
	print(df[df.select_dtypes(include=[object]).columns].describe().T)
	"""

	run_RecSys(df)
	#return

if __name__ == '__main__':
	os.system("clear")
	main()