"""
import re
import string
import os
import sys
import joblib
import time
import argparse
import glob
import numpy as np
import pandas as pd
"""

from collections import Counter
from natsort import natsorted


from utils import *
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable

import matplotlib
matplotlib.use("Agg")

parser = argparse.ArgumentParser(description='National Library of Finland (NLF) Data Visualization & Analysis')

parser.add_argument('--inputDF', default=os.path.join(dfs_path, "nikeY.docworks.lib.helsinki.fi_access_log.07_02_2021.log.dump"), type=str) # smallest

args = parser.parse_args()

# how to run:
# python nlf_analysis.py --inputDF /home/xenial/Datasets/Nationalbiblioteket/dataframes/nikeY.docworks.lib.helsinki.fi_access_log.07_02_2021.log.dump

sz=12
params = {
	'figure.figsize':	(sz*2.0, sz*1.1),  # W, H
	'figure.dpi':		350,
	'figure.autolayout': True,
	#'figure.constrained_layout.use': True,
	'legend.fontsize':	sz*0.8,
	'axes.labelsize':	sz*1.0,
	'axes.titlesize':	sz*1.0,
	'xtick.labelsize':	sz*1.7,
	'ytick.labelsize':	sz*1.7,
	'lines.linewidth' :	sz*0.1,
	'lines.markersize':	sz*0.8,
	'font.size':		sz*1.5,
	'font.family':		"serif",
}
pylab.rcParams.update(params)

sns.set(font_scale=1.3, 
				style="white", 
				palette='deep', 
				font="serif", 
				color_codes=True,
				)

my_cols = [ "#ffd100", '#16b3ff', '#0ecd19', '#ff9999',]

def clean_(text_list):
	#print(text_list)
	assert len(text_list) == 1
	cleaned_text = text_list[0].lower()
	cleaned_text = re.sub(r'"|<.*?>|[~|*|^][\d]+', '', cleaned_text)
	cleaned_text = re.sub(r"\W+|_"," ", cleaned_text) # replace special characters with space
	cleaned_text = re.sub("\s+", " ", cleaned_text)

	return cleaned_text

def get_query_dataframe(QUERY=0):
	#print(f"\nGiven log file index: {QUERY}")
	all_dump_file_paths = natsorted( glob.glob( os.path.join(dfs_path, "*.dump") ) )
	#print(all_dump_file_paths)

	all_files = [dfile[dfile.rfind("/")+1:dfile.rfind(".dump")] for dfile in all_dump_file_paths]
	#print(all_files)

	query_dataframe_file = all_files[QUERY] 
	result_directory = os.path.join(rpath, query_dataframe_file)
	make_folder(folder_name=result_directory)
	print(f"Input Query: {QUERY}")

	return query_dataframe_file

def get_result_directory(QUERY=0):
	#print(f"\nGiven log file index: {QUERY}")
	all_dump_file_paths = natsorted( glob.glob( os.path.join(dfs_path, "*.dump") ) )
	#print(all_dump_file_paths)

	all_files = [dfile[dfile.rfind("/")+1:dfile.rfind(".dump")] for dfile in all_dump_file_paths]
	#print(all_files)

	query_dataframe_file = all_files[QUERY] 
	res_dir = os.path.join(rpath, query_dataframe_file)
	return res_dir

def plot_missing_features(df, RES_DIR):
	print(f">> Visualizing missing data of  ...")

	print(f">>>>> Barplot >>>>>")
	g = sns.displot(
			data=df.isna().melt(value_name="Missing"),
			y="variable",
			hue="Missing",
			multiple="stack",
			height=15,
			#kde=True,
			aspect=1.8,
	)
	g.set_axis_labels("Samples", "Features")
	for axb in g.axes.ravel():
		# add annotations
		for c in axb.containers:
			# custom label calculates percent and add an empty string so 0 value bars don't have a number
			labels = [f"{(v.get_width()/df.shape[0]*100):.1f} %" if v.get_width() > 0 else "" for v in c]
			axb.bar_label(c,
										labels=labels,
										label_type='edge',
										#fontsize=13,
										rotation=0,
										padding=5,
										)
			break; # only annotate the first!
		axb.margins(y=0.3)
	plt.savefig(os.path.join( RES_DIR, f"missing_barplot.png" ), )
	plt.clf()
	plt.close()

	print(f"{'Heatmap'.center(80, '-')}")
	f, ax = plt.subplots()
	ax = sns.heatmap(
			df.isna(),
			cmap=sns.color_palette("Greys"),
			cbar_kws={'label': 'Missing Data (NaN)', 
								'ticks': [0.0, 1.0],
								'pad': 0.02,
								'shrink': 0.5,
								'orientation': 'horizontal',
								}
							)
	ax.set_ylabel(f"Samples\n\n{df.shape[0]}$\longleftarrow${0}")
	ax.set_yticks([])
	ax.xaxis.tick_top()
	ax.tick_params(axis='x', labelrotation=90)
	plt.suptitle(f"Data Availability: {df.shape}")
	print(os.path.join( RES_DIR, f'missing_heatmap.png' ))
	plt.savefig(os.path.join( RES_DIR, f"missing_cols_heatmap.png" ), bbox_inches='tight')
	plt.clf()
	plt.close(f)

def plot_language(df, RES_DIR, N=20):
	df_cleaned = df.dropna(axis=0, how="any", subset=["language"]).reset_index()

	df_unq = df_cleaned.assign(language=df_cleaned['language'].str.split(',')).explode('language')
	
	print(df_unq["language"].value_counts())
	print("#"*150)

	fig = plt.figure(figsize=(10,4))
	axs = fig.add_subplot(121)
	patches, _ = axs.pie(df_cleaned["language"].value_counts(),
											colors=clrs,
											wedgeprops=dict(width=0.8,
																			edgecolor="#e503",
																			linewidth=0.5,
																			),
											)
	
	axs.axis('equal')
	axs.set_title(f"{len(df_cleaned['language'].value_counts())} Raw NLF Languages\n{df_cleaned['language'].shape[0]}/{df['language'].shape[0]}")
	
	ax2 = fig.add_subplot(122)
	ax2.axis("off")

	ax2.legend(patches,
						[ f"{l} {v*100:.2f} %" for l, v in zip(	df_cleaned["language"].value_counts(normalize=True).index, 
																						df_cleaned["language"].value_counts(normalize=True).values,
																					)
						],
						loc="center",
						frameon=False,
						fontsize=8,
						)
	plt.savefig(os.path.join( RES_DIR, f"pie_chart_RAW_language.png" ), 
							bbox_inches="tight",
							)
	plt.clf()
	plt.close(fig)

	fig = plt.figure(figsize=(10,4))
	axs = fig.add_subplot(121)
	patches, _ = axs.pie(df_unq["language"].value_counts(),
											colors=clrs,
											wedgeprops=dict(width=0.8,
																			edgecolor="#e856",
																			linewidth=0.5,
																			),
											)
	
	axs.axis('equal')
	axs.set_title(f"{len(df_unq['language'].value_counts())} Unique NLF Language\n{df_unq['language'].shape[0]}/{df['language'].shape[0]}")
	
	ax2 = fig.add_subplot(122)
	ax2.axis("off")

	ax2.legend(patches,
						[ f"{l} {v*100:.2f} %" for l, v in zip(	df_unq["language"].value_counts(normalize=True).index, 
																						df_unq["language"].value_counts(normalize=True).values,
																					)
						],
						loc="center",
						frameon=False,
						fontsize=9,
						)
	plt.savefig(os.path.join( RES_DIR, f"pie_chart_unique_language.png" ), 
							bbox_inches="tight",
							)
	plt.clf()
	plt.close(fig)

def plot_query_phrases(df,RES_DIR, Nq=50, Nu=5):
	print(f">> Ploting query phrases | df: {df.shape}")

	#df_cleaned = df.dropna(axis=0, how="any", subset=["query_word"]).reset_index(drop=True)
	df_cleaned = df.copy()
	"""
	print(df_cleaned["query_word"].value_counts())
	print(list(zip(df_cleaned["query_word"].value_counts().index, df_cleaned["query_word"].value_counts().values)))
	print("/"*150)
	"""
	
	#df_cleaned["search_query_phrase"] = df_cleaned['search_query_phrase'].str.replace(r'[^\w\s]+|\d+', '', regex=True).str.replace(r"\s+", " ", regex=True).str.strip().str.lower()#.str.replace(r'(?<=\D)(?=\d)', ' ', regex=True)
	df_cleaned["search_query_phrase"] = df_cleaned['search_query_phrase'].map(clean_, na_action="ignore")

	"""
	print(df_cleaned["query_word"].value_counts())
	print(list(zip(df_cleaned["query_word"].value_counts().index, df_cleaned["query_word"].value_counts().values)))
	print("*"*150)
	#return
	"""

	wordcloud = WordCloud(width=1400, 
												height=550, 
												background_color="black",
												colormap="RdYlGn",
												max_font_size=100,
												stopwords=None,
												collocations=False,
												).generate_from_frequencies(dict(zip(	df_cleaned["search_query_phrase"].value_counts().index, 
																															df_cleaned["search_query_phrase"].value_counts().values,
																															)
																												)
																										)

	plt.figure(figsize=(14, 8))
	plt.imshow(wordcloud, interpolation='bilinear')
	plt.axis("off")
	plt.title(f"Cloud Distribution\n{len(df_cleaned['search_query_phrase'].value_counts())}\n" 
						f"Unique Query Phrases (total: {df_cleaned['search_query_phrase'].notnull().sum()})", color="k")
	plt.margins(x=0, y=0)
	plt.tight_layout(pad=0) 
	plt.savefig(os.path.join( RES_DIR, f"query_phrases_cloud.png" ), )
	plt.clf()
	plt.close()

	#plt.subplots()
	plt.figure()
	p = sns.barplot(x=df_cleaned["search_query_phrase"].value_counts()[:Nq].index,
									y=df_cleaned["search_query_phrase"].value_counts()[:Nq].values,
									palette=clrs, 
									saturation=1,
									edgecolor="#450f30",
									linewidth=1,
									#orient = "h",
									)

	p.set_xticklabels(df_cleaned["search_query_phrase"].value_counts()[:Nq].index, size=10)
	p.axes.set_title(	f"Top-{Nq} Query Phrases (unique: {len(df_cleaned['search_query_phrase'].value_counts())})")
	plt.ylabel("Counts", )
	plt.xlabel("Query Phrase",)
	plt.xticks(rotation=90, fontsize=17)
	for container in p.containers:
			p.bar_label(container,
									label_type="center",
									padding=3.0,
									size=14,
									color="black",
									rotation=90,
									bbox={"boxstyle": "round", 
												"pad": 0.4,
												"facecolor": "orange", 
												"edgecolor": "black", 
												"alpha": 1.0,
												}
									)

	sns.despine(left=True, bottom=True)
	plt.savefig(os.path.join( RES_DIR, f"top_{Nq}_query_phrases.png" ), bbox_inches="tight",)
	plt.clf()
	plt.close()

	MY_DICT = {}

	for usr in df_cleaned["user_ip"].value_counts()[:Nu].index:
		lst = []
		for qu in df_cleaned["search_query_phrase"].value_counts()[:Nq].index:
			#print(usr, qu)
			#c = df[(df["query_word"] == qu) & (df["user_ip"] == usr) ].user_ip.count()
			c = df_cleaned[(df_cleaned["search_query_phrase"] == qu) & (df_cleaned["user_ip"] == usr) ].user_ip.count()
			#print(c)		
			lst.append(c)
		MY_DICT[usr] = lst

	#print(MY_DICT)

	WIDTH = 0.4
	BOTTOM = 0

	fig, axs = plt.subplots()
	for k, v in MY_DICT.items():
			#print(k, v)
			axs.bar(x=df_cleaned["search_query_phrase"].value_counts()[:Nq].index,
								height=v, 
								width=WIDTH,
								bottom=BOTTOM, 
								color=clrs[list(MY_DICT.keys()).index(k)],
								label=k,
								edgecolor="#450f30",
								linewidth=0.5,
								)
			BOTTOM += np.array(v)
			axs.margins(x=0.01)
	axs.legend(	loc="upper right",
							frameon=False,
							title=f"Top-{Nu} Users",
							#ncol=len(MY_DICT),
							fontsize=8.0,
							)
	plt.suptitle(f"Top-{Nq} Query Phrases Searched by Top-{Nu} Users")
	axs.set_ylabel('Counts')
	axs.set_xlabel('\nQuery Phrases')
	axs.tick_params(axis='x', rotation=90, labelsize=10.0)
	axs.spines[['top', 'right']].set_visible(False)
	plt.savefig(os.path.join( RES_DIR, f"top_{Nu}_USR_searching_top_{Nq}_query_phrases.png" ), )
	plt.clf()
	plt.close(fig)

def plot_nwp_content_parsed_term(df,RES_DIR, N=20):
	print(f">> Ploting newspaper content parsed terms | df: {df.shape}")
	#df_cleaned = df.copy()
	df_cleaned = df.dropna(axis=0, how="any", subset=["nwp_content_parsed_term"]).reset_index()
	"""
	with pd.option_context('display.max_rows', 300, 'display.max_colwidth', 500):
		print(df_cleaned[["nwp_content_parsed_term", "referer"]].head(88))
	"""
	nwp_parsed_terms_lst = list()

	def extract_terms(trms_list):
		nwp_parsed_terms_lst.extend([w.lower() for w in trms_list]) # # len: 3539 unq: 669
		#nwp_parsed_terms_lst.extend(trms_list) # len: 3539 unq: 761

	df_cleaned["nwp_content_parsed_term"].apply(extract_terms)

	print(len(nwp_parsed_terms_lst), len(set(nwp_parsed_terms_lst)),nwp_parsed_terms_lst[:20])
	#print(Counter(nwp_parsed_terms_lst)) # dict: {k: words: v: counts}
	wordcloud = WordCloud(width=1800, 
												height=550,
												background_color="black",
												colormap="RdYlGn",
												max_font_size=80,
												stopwords=None,
												collocations=False,
												).generate_from_frequencies(Counter(nwp_parsed_terms_lst))

	fig = plt.figure(figsize=(16, 8))
	plt.imshow(wordcloud)
	plt.axis("off")
	plt.title(f"Word Cloud Distribution\n{len(set(nwp_parsed_terms_lst))} Unique "
						f"Newspaper Content Page Parsed Terms (total: {len(nwp_parsed_terms_lst)})", color="k")
	plt.margins(x=0, y=0)
	plt.tight_layout(pad=0) 

	plt.savefig(os.path.join( RES_DIR, f"nwp_content_parsed_{len(nwp_parsed_terms_lst)}_terms_cloud.png" ), )
	plt.clf()
	plt.close(fig)

def plot_publication_places(df,RES_DIR):
	df_cleaned = df.dropna(axis=0, how="any", subset=["publication_place"]).reset_index()

	df_unq = df_cleaned.assign(publication_place=df_cleaned['publication_place'].str.split(',')).explode('publication_place')
	df_unq["publication_place"] = df_unq["publication_place"].str.title()

	fig = plt.figure(figsize=(10,4))
	axs = fig.add_subplot(121)
	patches, _ = axs.pie(df_cleaned["publication_place"].value_counts(),
											colors=clrs,
											wedgeprops=dict(width=0.8, 
																			edgecolor="#2ef3",
																			linewidth=0.2,
																			),
											)
	
	axs.axis('equal')
	axs.set_title(f"{len(df_cleaned['publication_place'].value_counts())} Raw NLF Publication Places\n{df_cleaned['publication_place'].shape[0]}/{df['publication_place'].shape[0]}")
	
	ax2 = fig.add_subplot(122)
	ax2.axis("off")

	ax2.legend(patches,
						[ f"{l} {v*100:.2f} %" for l, v in zip(	df_cleaned["publication_place"].value_counts(normalize=True).index, 
																						df_cleaned["publication_place"].value_counts(normalize=True).values,
																					)
						],
						loc="center",
						frameon=False,
						fontsize=5,
						)
	plt.savefig(os.path.join( RES_DIR, f"pie_chart_RAW_pub_places.png" ), 
							bbox_inches="tight",
							)
	plt.clf()
	plt.close(fig)

	fig = plt.figure(figsize=(10,4))
	axs = fig.add_subplot(121)
	patches, _ = axs.pie(df_unq["publication_place"].value_counts(),
											colors=clrs,
											wedgeprops=dict(width=0.8, 
																			edgecolor="#ee7f3000",
																			linewidth=0.5,
																			),
											)
	
	axs.axis('equal')

	axs.set_title(f"{len(df_unq['publication_place'].value_counts())} Unique NLF Publication Places\n{df_unq['publication_place'].shape[0]}/{df['publication_place'].shape[0]}")
	
	ax2 = fig.add_subplot(122)
	ax2.axis("off")

	ax2.legend(patches,
						[ f"{l} {v*100:.2f} %" for l, v in zip(	df_unq["publication_place"].value_counts(normalize=True).index, 
																						df_unq["publication_place"].value_counts(normalize=True).values,
																					)
						],
						loc="center",
						frameon=False,
						fontsize=9,
						)
	plt.savefig(os.path.join( RES_DIR, f"pie_chart_unique_pub_places.png" ), 
							bbox_inches="tight",
							)
	plt.clf()
	plt.close(fig)

	wordcloud = WordCloud(width=800, 
												height=250, 
												background_color="black",
												colormap="RdYlGn",
												max_font_size=100,
												stopwords=None,
												collocations=False,
												).generate(df["publication_place"].str.cat(sep=",")) #Concatenate strings in the Series/Index with given separator.

	fig = plt.figure(figsize=(10, 4))
	plt.imshow(wordcloud)
	plt.axis("off")
	plt.title(f"{len(df_unq['publication_place'].value_counts())} Unique Publication Places Cloud", color="k")
	plt.margins(x=0, y=0)
	plt.tight_layout(pad=0) 

	plt.savefig(os.path.join( RES_DIR, f"pub_places_cloud.png" ), )
	plt.clf()
	plt.close(fig)

	df_count_dt = df_unq.groupby(df_unq["publication_place"])[["user_ip", "query_word", "nwp_content_parsed_term",]].count().reset_index()
	print(df_count_dt)

	print("#"*150)
	
	fig, axs = plt.subplots()
	df_count_dt.set_index("publication_place").plot( rot=90,
								ax=axs,
								kind='bar',
								xlabel="Unique Publication Places", 
								ylabel="Count",
								title=f"{df_count_dt.shape[0]} Unique Publication Places | USERS | QUERY WORDS | OCR TERMS",
								color=clrs,
								alpha=0.6,
								)
	for container in axs.containers:
		axs.bar_label(container, rotation=45, fontsize=9,)
	plt.legend(	loc="upper right", 
							frameon=False,
							#title=f"Top-{Nu} Users",
							ncol=df_count_dt.shape[0], 
							)

	plt.savefig(os.path.join( RES_DIR, f"unq_pub_places_usr_qu_ocr.png" ), )
	plt.clf()
	plt.close(fig)

def plot_doc_type(df,RES_DIR):
	df_cleaned = df.dropna(axis=0, how="any", subset=["document_type"]).reset_index()

	df_unq = df_cleaned.assign(document_type=df_cleaned['document_type'].str.split(',')).explode('document_type')
	df_unq["document_type"] = df_unq["document_type"].str.title()

	fig = plt.figure(figsize=(10,4))
	axs = fig.add_subplot(121)
	patches, _ = axs.pie(df_cleaned["document_type"].value_counts(),
											colors=clrs,
											wedgeprops=dict(width=0.8, 
																			edgecolor="#ee7f3000",
																			linewidth=0.3,
																			),
											)
	
	axs.axis('equal')
	axs.set_title(f"{len(df_cleaned['document_type'].value_counts())} Raw NLF document type\n{df_cleaned['document_type'].shape[0]}/{df['document_type'].shape[0]}")
	
	ax2 = fig.add_subplot(122)
	ax2.axis("off")

	ax2.legend(patches,
						[ f"{l} {v*100:.2f} %" for l, v in zip(	df_cleaned["document_type"].value_counts(normalize=True).index, 
																						df_cleaned["document_type"].value_counts(normalize=True).values,
																					)
						],
						loc="center",
						frameon=False,
						fontsize=5,
						)
	plt.savefig(os.path.join( RES_DIR, f"pie_chart_RAW_document_type.png" ), 
							bbox_inches="tight",
							)
	plt.clf()
	plt.close(fig)

	fig = plt.figure(figsize=(10,4))
	axs = fig.add_subplot(121)
	patches, _ = axs.pie(df_unq["document_type"].value_counts(),
											colors=clrs,
											wedgeprops=dict(width=0.8, 
																			edgecolor="#ee7f3000",
																			linewidth=0.5,
																			),
											)
	
	axs.axis('equal')

	axs.set_title(f"{len(df_unq['document_type'].value_counts())} Unique NLF document type\n{df_unq['document_type'].shape[0]}/{df['document_type'].shape[0]}")
	
	ax2 = fig.add_subplot(122)
	ax2.axis("off")

	ax2.legend(patches,
						[ f"{l} {v*100:.2f} %" for l, v in zip(	df_unq["document_type"].value_counts(normalize=True).index, 
																						df_unq["document_type"].value_counts(normalize=True).values,
																					)
						],
						loc="center",
						frameon=False,
						fontsize=9,
						)
	plt.savefig(os.path.join( RES_DIR, f"pie_chart_unique_document_type.png" ), 
							bbox_inches="tight",
							)
	plt.clf()

	wordcloud = WordCloud(width=800, 
												height=250, 
												background_color="black",
												colormap="RdYlGn",
												max_font_size=100,
												stopwords=None,
												collocations=False,
												).generate(df["document_type"].str.cat(sep=",")) #Concatenate strings in the Series/Index with given separator.

	fig = plt.figure(figsize=(10, 4))
	plt.imshow(wordcloud)
	plt.axis("off")
	plt.title(f"{len(df_unq['document_type'].value_counts())} Unique Document Types Cloud", color="k")
	plt.margins(x=0, y=0)
	plt.tight_layout(pad=0) 

	plt.savefig(os.path.join( RES_DIR, f"doc_type_cloud.png" ), )
	plt.clf()
	plt.close(fig)

	df_count_dt = df_unq.groupby(df_unq["document_type"])[["user_ip", "query_word", "nwp_content_parsed_term",]].count().reset_index()
	print(df_count_dt)

	print("#"*150)
	
	fig, axs = plt.subplots()
	df_count_dt.set_index("document_type").plot( rot=0,
								ax=axs,
								kind='bar',
								xlabel="Unique Document Type", 
								ylabel="Count",
								title=f"{df_count_dt.shape[0]} Unique Document Type | USERS | QUERY WORDS | OCR TERMS",
								color=clrs,
								alpha=0.6,
								)
	for container in axs.containers:
		axs.bar_label(container, rotation=45, )
	plt.legend(	loc="upper left", 
							frameon=False,
							#title=f"Top-{Nu} Users",
							ncol=df_count_dt.shape[0], 
							)

	plt.savefig(os.path.join( RES_DIR, f"unq_doc_type.png" ), )
	plt.clf()
	plt.close(fig)

def plot_user(df,RES_DIR, N=50):
	#print(df["user_ip"].value_counts())

	df_tmp = df.dropna(axis=0, how="any", subset=["user_ip"]).reset_index(drop=True)
	print(f">> df: {df.shape} | df_tmp: {df_tmp.shape}")

	df_tmp["query_word"] = df_tmp['query_word'].str.replace(r'[^\w\s]+|\d+', '', regex=True).str.replace(r"\s+", " ", regex=True).str.strip().str.lower()#.str.replace(r'(?<=\D)(?=\d)', ' ', regex=True)
	#df_tmp["nwp_content_parsed_term"] = df_tmp['nwp_content_parsed_term'].str.replace(r'[^\w\s]+|\d+', '', regex=True).str.replace(r"\s+", " ", regex=True).str.strip().str.lower()#.str.replace(r'(?<=\D)(?=\d)', ' ', regex=True)

	MY_DICT = {}
	
	lst_q, lst_ocr, lst_nan = [], [], []
	for usr in df_tmp["user_ip"].value_counts()[:N].index:
		cq = df_tmp[ (df_tmp["user_ip"] == usr) ].query_word.count()
		c_ocr = df_tmp[ (df_tmp["user_ip"] == usr) ].nwp_content_parsed_term.count()
		c_usr = df_tmp[ (df_tmp["user_ip"] == usr) ].user_ip.count()
		print(f"\n{usr}:\tQU: {cq} | OCR: {c_ocr}:\t({cq+c_ocr} / {c_usr})")
		
		lst_q.append(cq)
		lst_ocr.append(c_ocr)
		lst_nan.append(abs(c_usr - (cq+c_ocr)))

		#print(df_tmp[ (df_tmp["user_ip"] == usr) ].query_word.value_counts())

		########################################### QUERY PHRASES ###########################################
		fig = plt.figure(figsize=(15,10))
		axs = fig.add_subplot(121)
		patches, _ = axs.pie(df_tmp[ (df_tmp["user_ip"] == usr) ].query_word.value_counts(),
												colors=clrs,
												wedgeprops=dict(width=0.6,
																				edgecolor="#2ef3",
																				linewidth=0.2,
																				),
												)
		
		#axs.axis('equal')
		#axs.set_title(f"USER: {usr}\nQuery Phrases (Unique: {len(df_tmp[ (df_tmp['user_ip'] == usr) ].query_word.value_counts())} Total: {cq})")
		
		ax2 = fig.add_subplot(122)
		ax2.axis("off")

		ax2.legend(patches,
							[ f"{l} {v*100:.1f} %" for l, v in zip(	df_tmp[ (df_tmp["user_ip"] == usr) ].query_word.value_counts(normalize=True).index, 
																											df_tmp[ (df_tmp["user_ip"] == usr) ].query_word.value_counts(normalize=True).values,
																						)
							],
							loc="center",
							frameon=False,
							fontsize=16 if len(df_tmp[ (df_tmp["user_ip"] == usr) ].query_word.value_counts()) < 15 else 12,
							ncol=3 if len(df_tmp[ (df_tmp["user_ip"] == usr) ].query_word.value_counts()) > 25 else 1,
							)
		#fig.canvas.draw()
		plt.suptitle(f"USER: {usr}\nQuery Phrases (Unique: {len(df_tmp[ (df_tmp['user_ip'] == usr) ].query_word.value_counts())} Total: {cq})")
		plt.tight_layout()
		plt.savefig(os.path.join( RES_DIR, f"pie_usr_{usr}_query_phrases.png" ), 
								bbox_inches="tight",
								)
		#plt.clf()
		plt.close(fig)
		########################################### QUERY PHRASES ###########################################

		########################################### OCR TERMS ###########################################
		fig = plt.figure(figsize=(28,15))
		axs = fig.add_subplot(121)
		patches, _ = axs.pie(df_tmp[ (df_tmp["user_ip"] == usr) ].nwp_content_parsed_term.value_counts(),
												colors=clrs,
												wedgeprops=dict(width=0.8,
																				edgecolor="#2ef3",
																				linewidth=0.2,
																				),
												)
		
		#axs.axis('equal')		
		ax2 = fig.add_subplot(122)
		ax2.axis("off")

		ax2.legend(patches,
							[ f"{l} {v*100:.1f} %" for l, v in zip(	df_tmp[ (df_tmp["user_ip"] == usr) ].nwp_content_parsed_term.value_counts(normalize=True).index, 
																											df_tmp[ (df_tmp["user_ip"] == usr) ].nwp_content_parsed_term.value_counts(normalize=True).values,
																						)
							],
							loc="upper left" if len(df_tmp[ (df_tmp["user_ip"] == usr) ].nwp_content_parsed_term.value_counts()) > 25 else "center",
							frameon=False,
							fontsize=20 if len(df_tmp[ (df_tmp["user_ip"] == usr) ].nwp_content_parsed_term.value_counts()) < 55 else 18,
							ncol=2 if len(df_tmp[ (df_tmp["user_ip"] == usr) ].nwp_content_parsed_term.value_counts()) > 55 else 1,
							)
		
		#fig.canvas.draw()
		#plt.tight_layout()
		plt.suptitle(f"USER: {usr}\nNewspaer Content Parsed Terms (Unique: {len(df_tmp[ (df_tmp['user_ip'] == usr) ].nwp_content_parsed_term.value_counts())} Total: {c_ocr})")
		plt.savefig(os.path.join( RES_DIR, f"pie_usr_{usr}_nwp_content_parsed_term.png" ), 
								bbox_inches="tight",
								)
		#plt.clf()
		plt.close(fig)
		########################################### OCR TERMS ###########################################

	MY_DICT["Nwp_Parsed_Terms"] = lst_ocr
	MY_DICT["None"] = lst_nan
	MY_DICT["Query_Phrases"] = lst_q

	#print(MY_DICT)

	fig, axs = plt.subplots(figsize=(25, 13))
	WIDTH = 0.35
	BOTTOM = 0
	magnifier = 1
	bar_width = magnifier * 0.25
	spare_width = 0.8 * (1 - ( 2 * bar_width ) )

	for k, v in MY_DICT.items():
		#print(k, len(v), v)
		axs.bar(x=df_tmp["user_ip"].value_counts()[:N].index,
						height=v,
						width=WIDTH,
						bottom=BOTTOM, 
						color=clrs[list(MY_DICT.keys()).index(k)],
						label=k,
						edgecolor="#450f30",
						linewidth=.5,
						)
		BOTTOM += np.array(v)

	cell_text = [ MY_DICT[k] for k, v in MY_DICT.items() ]
	the_table = axs.table(cellText=cell_text,
												rowLabels=list(MY_DICT.keys()),
												colLabels=df_tmp["user_ip"].value_counts()[:N].index,
												rowLoc='center',
						 						colLoc='center',
									 			cellLoc='center',
												loc='bottom')
	for (row, col), cell in the_table.get_celld().items():
		if row == 0:
			cell.get_text().set_rotation(90)
			cell.set_height(0.03)
	
	the_table.auto_set_font_size(False)
	the_table.set_fontsize(12.2)
	the_table.scale(1, 3)
	axs.set_xticklabels([])	

	axs.set_xlim(-spare_width, N - spare_width)

	axs.legend(	loc="upper right",
							frameon=False,
							ncol=len(MY_DICT),
							fontsize=15,
							)

	plt.suptitle(f"Top-{N} Users | Unique: {len(df_tmp['user_ip'].value_counts())} (total: {df_tmp['user_ip'].shape[0]})")
	axs.set_ylabel('User Activity [Presence]')
	#axs.set_xlabel('\nUsers')
	#axs.tick_params(axis='x', rotation=90)
	axs.spines[['top', 'right']].set_visible(False)
	plt.savefig(os.path.join( RES_DIR, f"top{N}_usrs_QU_OCR_NaN.png" ), bbox_inches="tight")
	plt.clf()
	plt.close(fig)

	#return

	plt.subplots()
	p = sns.barplot(x=df_tmp["user_ip"].value_counts()[:N].index,
									y=df_tmp["user_ip"].value_counts()[:N].values,
									palette=clrs, 
									saturation=1,
									edgecolor="#1c1c1c",
									linewidth=2,
									)

	p.set_xticklabels(df_tmp["user_ip"].value_counts()[:N].index, size=11)

	p.axes.set_title(f"Top-{N} Users | Unique: {len(df_tmp['user_ip'].value_counts())}",)
	plt.ylabel("Presence",)
	plt.xlabel("\nUser",)
	plt.xticks(rotation=90)
	for container in p.containers:
			p.bar_label(container,
									label_type="center",
									padding=3,
									size=10,
									color="black",
									rotation=90,
									bbox={"boxstyle": "round", 
												"pad": 0.4, 
												"facecolor": "orange", 
												"edgecolor": "black", 
												"alpha": 0.9,
												}
									)

	sns.despine(left=True, bottom=True)
	plt.savefig(os.path.join( RES_DIR, f"top_{N}_users_activity_tracking.png" ), bbox_inches='tight')
	plt.clf()
	plt.close()

def plot_hourly_activity(df,RES_DIR, Nu=25):
	df_count = df.groupby(df["timestamp"].dt.hour)[["user_ip", "query_word", "nwp_content_parsed_term",]].count().reset_index()

	fig, axs = plt.subplots()
	
	df_count.set_index("timestamp").plot( rot=0, 
								ax=axs,
								kind='bar',
								xlabel="Hour (o'clock)", 
								ylabel="Activity",
								title=f"24-Hour Activity",
								color=clrs,
								alpha=0.6,
								)
	
	df_count.set_index("timestamp").plot(	kind='line', 
								rot=0, 
								ax=axs, 
								marker="*", 
								linestyle="-", 
								linewidth=0.5,
								color=clrs,
								label=None,
								legend=False,
								xlabel="Hour (o'clock)", 
								ylabel="Activity",
								)

	plt.legend(	["Users", "Query Words", "OCR Terms"],
							loc="upper left", 
							frameon=False,
							ncol=df_count.shape[0],
							)
	
	
	plt.savefig(os.path.join( RES_DIR, f"24h_activity.png" ), )
	plt.clf()
	plt.close(fig)

	#time_window = df_count["timestamp"].shape[0]/4
	time_window = int(24 / 4)
	
	print(df_count.columns)
	for col in df_count.columns[1:]:
			fig, axs = plt.subplots()
			print(col)
			df_count[col].plot(	kind='line', 
													rot=0, 
													ax=axs, 
													marker="*", 
													linestyle="-", 
													linewidth=0.5,
													color=clrs,
													label=None,
													legend=False,
													xlabel="Hour (o'clock)", 
													ylabel="Activity",
													title=f"Mean($\mu$) & Standard Deviation ($\sigma$): {col}"
												)

			for i in range(4):
				print(i)
				print(i*time_window , (i+1)*time_window)
				df_sliced = df_count[i*time_window: ((i+1)*time_window)+1]
				print(df_sliced)

				axs.plot(df_sliced["timestamp"], 
								[df_sliced.mean()[col] for t in df_sliced["timestamp"]],
								color="red",
								linestyle="-",
								linewidth=0.7,
								label=f"($\mu$ = {df_sliced.mean()[col]:.1f})\t{i*time_window}:{(i+1)*time_window} o'clock",
								)

				axs.fill_between(df_sliced["timestamp"], 
								[ df_sliced.mean()[col] - 3 * df_sliced.std()[col] for t in df_sliced["timestamp"]],
								[ df_sliced.mean()[col] + 3 * df_sliced.std()[col] for t in df_sliced["timestamp"]],
								alpha=0.1,
								color=my_cols[i],
								label=f"+/- 3$\sigma$\t{i*time_window}:{(i+1)*time_window} o'clock",
								)

				axs.fill_between(df_sliced["timestamp"], 
								[ df_sliced.mean()[col] - 1 * df_sliced.std()[col] for t in df_sliced["timestamp"]],
								[ df_sliced.mean()[col] + 1 * df_sliced.std()[col] for t in df_sliced["timestamp"]],
								alpha=0.14,
								color=my_cols[i],
								label=f"+/- 1$\sigma$\t{i*time_window}:{(i+1)*time_window} o'clock",
								)


				#print(df_sliced_grouped)
				
				print("-"*60)
			
				axs.legend(	#["Users", "Query Words", "OCR Terms"],
										loc="upper left", 
										frameon=False,
										#ncol=df_count.shape[0],
										)
				plt.savefig(os.path.join( RES_DIR, f"{col}_mean_std.png" ), )
			plt.close() # TODO: double check the place!!!

def plot_usr_doc_type(df,RES_DIR, Nu=25):
	df_cleaned = df.assign(document_type=df['document_type'].str.split(',')).explode('document_type')
	
	print(df_cleaned["document_type"].value_counts())
	print("#"*40)

	#return
	df_count = df.groupby(df["document_type"])[["user_ip", "query_word", "nwp_content_parsed_term",]].count().reset_index()
	print(df_count)
	print("#"*150)

	df_count_usr = df_cleaned.groupby(df_cleaned["user_ip"])[["document_type"]].count().reset_index().sort_values(by="document_type", ascending=False)
	print(df_count_usr)
	print("#"*150)


	print( df_cleaned[df_cleaned["document_type"]=="NEWSPAPER"][["user_ip","document_type"]])
	print("/"*150)
	
	# TOP-N users:	
	df_topN_users = df_cleaned["user_ip"].value_counts().head(Nu).rename_axis('user_ip').reset_index(name='occurrence')
	print(df_topN_users)

	fig, axs = plt.subplots()
	#users:
	dfgr_dt_usr = df.groupby('document_type')['user_ip']#.sort_values(by="user_ip", ascending=False)
	print(len(dfgr_dt_usr.get_group("JOURNAL")))
	print(">"*10)
	print(dfgr_dt_usr.get_group("JOURNAL"))

	df_doc_type_vs_users = dfgr_dt_usr.count() # change to pandas df
	df_doc_type_vs_users.plot( rot=90,
								ax=axs,
								kind='bar',
								xlabel="Document Type", 
								ylabel="Count",
								title=f"User Intrests of {df_doc_type_vs_users.shape[0]} Raw Document Type",
								color=clrs,
								alpha=0.6,
								fontsize=11,
								)
	axs.spines[['top', 'right']].set_visible(False)
	axs.bar_label(axs.containers[0])
	plt.savefig(os.path.join( RES_DIR, f"usr_vs_doc_type.png" ), )
	plt.clf()
	plt.close(fig)

def main():
	print("#"*60)
	print(f"\t\tDATA ANALYSIS & VISUALIZATION")
	print("#"*60)

	result_directory = make_result_dir(infile=args.inputDF)
	df = load_df(infile=args.inputDF)
	print(f"DF: {df.shape}")
	print("%"*140)
	print(df.info(verbose=True, memory_usage="deep"))
	print("%"*90)
	print(df[["user_ip", "search_query_phrase", "search_results"]].tail(10))
	"""
	cols = list(df.columns)
	print(len(cols), cols)
	print("#"*150)
	print(df.head(10))
	print("-"*150)
	print(df.tail(10))
	print(df.isna().sum())
	print("-"*150)
	print(df[df.select_dtypes(include=[object]).columns].describe().T)
	#return
	"""
	
	# missing features:
	#plot_missing_features(df, RES_DIR=result_directory)
	#return

	# 24h activity:
	#plot_hourly_activity(df, RES_DIR=result_directory)

	# users:
	#plot_user(df, RES_DIR=result_directory)

	# language
	#plot_language(df, RES_DIR=result_directory)

	# doc_type
	#plot_doc_type(df, RES_DIR=result_directory)

	# publication places
	#plot_publication_places(df, RES_DIR=result_directory)

	# users vs document_type:
	#plot_usr_doc_type(df, RES_DIR=result_directory)

	# query words & terms:
	plot_query_phrases(df, RES_DIR=result_directory)
	#plot_nwp_content_parsed_term(df, RES_DIR=result_directory)

if __name__ == '__main__':
	os.system('clear')
	main()