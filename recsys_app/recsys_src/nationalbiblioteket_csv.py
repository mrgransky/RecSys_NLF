import re
import string
import os
import sys
import joblib
import time

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns

import matplotlib
matplotlib.use("Agg")

# more info for adjustment of rcparams:
# https://matplotlib.org/stable/tutorials/introductory/customizing.html
sz=13 # >>>>>>>>> 12 original <<<<<<<<<<<
params = {
	'figure.figsize':	(sz*1.7, sz*1.0),  # W, H
	'figure.dpi':		200,
	'figure.autolayout': True,
	#'figure.constrained_layout.use': True,
	'legend.fontsize':	sz*0.8,
	'axes.labelsize':	sz*0.2,
	'axes.titlesize':	sz*0.2,
	'xtick.labelsize':	sz*1.0,
	'ytick.labelsize':	sz*1.0,
	'lines.linewidth' :	sz*0.1,
	'lines.markersize':	sz*0.8,
	'font.size':		sz*1.0,
	'font.family':		"serif",
}
pylab.rcParams.update(params)

sns.set(font_scale=1.3, 
				style="white", 
				palette='deep', 
				font="serif", 
				color_codes=True,
				)

files_list = ["digi_hakukaytto_v1.csv", 
							"digi_nidekaytto_v1.csv",
							"digi_sivukaytto_v1.csv",
							]

usr_ = {'alijani': '/lustre/sgn-data/vision', 
				'alijanif':	'/scratch/project_2004072/Nationalbiblioteket',
				"xenial": 	f"{os.environ['HOME']}/Datasets/Nationalbiblioteket",
				}

languages={"FINNISH": False, "ENGLISH": True}

NLF_DATASET_PATH = usr_[os.environ['USER']]

#dpath = os.path.join( usr_[os.environ['USER']], f"no_ip_logs" )
#rpath = os.path.join( dpath[:dpath.rfind("/")], f"results" )

dpath = os.path.join( NLF_DATASET_PATH, f"no_ip_logs" )
rpath = os.path.join( NLF_DATASET_PATH, f"results" )
dfs_path = os.path.join( NLF_DATASET_PATH, f"dataframes" )

search_idx, volume_idx, page_idx = 0, 1, 2

if not os.path.exists(rpath): 
	print(f"\n>> Creating DIR:\n{rpath}")
	os.makedirs( rpath )

if not os.path.exists(dfs_path): 
	print(f"\n>> Creating DIR:\n{dfs_path}")
	os.makedirs( dfs_path )

name_dict = {
	search_idx: [	"search_usage_id", 
								"material_type", 
								"date", 
								"search_phrase", 
								"date_start", 
								"date_end", 
								"publication_places",
								"publishers",
								"titles",
								"languages",
								"pages",
								"results",
								"rights",
								"fuzzy_search", 
								"illustrations",
								"index_prefix",
								"tags",
								"authors",
								"collections",
								"type",
								"text_meta",
								"text_ocr",
								"clip_keywords", #TODO: must be verified: Finnish: 'leike_asiasanat'
								"clip_categories",
								"clip_subjects",
								"clip_generated",
								"duration_ms",
								"order",
								"require_all_search_terms",
								"find_volumes",
								"last_page",
								"no_access_results",
								"import_time",
								"import_start_date",
							],
	volume_idx: ["volume_usage_id", 
								"volume_id", 
								"date", 
								"referer", 
								"robot", 
								"access_grounds", 
								"access_grounds_details", 
								"user_agent",
							],
	page_idx: 	["page_usage_id", 
								"page_id", 
								"date", 
								"referer", 
								"robot", 
								"access_grounds", 
								"access_grounds_details", 
								"user_agent", 
							],
							}

def basic_visualization(df, name=""):
	print(f">> Visualizing missing data of {name} ...")

	print(f">>>>> Barplot >>>>>")
	g = sns.displot(
			data=df.isna().melt(value_name="Missing"),
			y="variable",
			hue="Missing",
			multiple="stack",
			height=16,
			#kde=True,
			aspect=1.2,
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
	plt.savefig(os.path.join( rpath, f"{name}_missing_barplot.png" ), )
	plt.clf()

	print(f">>>>> Heatmap >>>>>")
	f, ax = plt.subplots()
	ax = sns.heatmap(
			df.isna(),
			cmap=sns.color_palette("Greys"),
			cbar_kws={'label': 'NaN (Missing Data)', 'ticks': [0.0, 1.0]},
			)

	ax.set_ylabel(f"Samples\n\n{df.shape[0]}$\longleftarrow${0}")
	ax.set_yticks([])
	ax.xaxis.tick_top()
	ax.tick_params(axis='x', labelrotation=90)
	plt.suptitle(f"Missing {name} Data (NaN)")
	plt.savefig(os.path.join( rpath, f"{name}_missing_heatmap.png" ), )
	plt.clf()

	print(f">>>>> Histogram >>>>>")
	hist = df.hist()
	plt.suptitle(f"Histogram {name} Data")
	plt.savefig(os.path.join( rpath, f"{name}_histogram.png" ), )
	plt.clf()

def get_df(idx, adjust_cols=True, keep_original=False):
	fname = os.path.join(dpath, files_list[idx])
	print(f">> Reading {fname} ...")

	df = pd.read_csv(fname, low_memory=False,)
	df['pvm'] = pd.to_datetime(df['pvm']) # ESSENTIAL to get datetime behavior in pandas

	if ('kielet' in list(df.columns)) and (keep_original==False):
		df['kielet'] = df['kielet'].str.replace(' ','', regex=True)
		df['kielet'] = df['kielet'].str.replace('FIN','FI', regex=True)
		df['kielet'] = df['kielet'].str.replace('SWE','SE', regex=True)
		df['kielet'] = df['kielet'].str.replace('ENG','EN', regex=True)
		df['kielet'] = df['kielet'].str.replace('RUS','RU', regex=True)
		df['kielet'] = df['kielet'].str.replace('GER','DE', regex=True)
		df['kielet'] = df['kielet'].str.replace('FRE','FR', regex=True)

		df['kielet'] = df['kielet'].str.replace('FIU,FI','FI,FIU', regex=True)
		df['kielet'] = df['kielet'].str.replace('SE,FI','FI,SE', regex=True)
		df['kielet'] = df['kielet'].str.replace('FI,FI','FI', regex=True)
		df['kielet'] = df['kielet'].str.replace('SE,SE','SE', regex=True)
		df['kielet'] = df['kielet'].str.replace('EN,FI','FI,EN', regex=True)
		df['kielet'] = df['kielet'].str.replace('DE,FI','FI,DE', regex=True)
		df['kielet'] = df['kielet'].str.replace('KRL,FI','FI,KRL', regex=True)
		df['kielet'] = df['kielet'].str.replace('EST,FI','FI,EST', regex=True)
		df['kielet'] = df['kielet'].str.replace('FR,FI','FI,FR', regex=True)

		df['kielet'] = df['kielet'].str.replace('EN,SE','SE,EN', regex=True)
		df['kielet'] = df['kielet'].str.replace('RU,FI','FI,RU', regex=True)
		df['kielet'] = df['kielet'].str.replace('RU,EN','EN,RU', regex=True)

	if adjust_cols:
		df.columns = name_dict.get(idx)
	
	return df

def save_dfs(qlang="ENGLISH"):
	rename_columns = languages[qlang]
	print(f">> Saving in {qlang} => rename_columns: {rename_columns} ...")

	search_df = get_df(idx=search_idx, adjust_cols=rename_columns)
	page_df = get_df(idx=page_idx, adjust_cols=rename_columns)
	volume_df = get_df(idx=volume_idx, adjust_cols=rename_columns)

	dfs_dict = {
		"search":	search_df,
		"vol":	volume_df,
		"pg":		page_df,
	}

	fname = "_".join(list(dfs_dict.keys()))+f"_dfs_{qlang}.dump"
	print(f">> Dumping {os.path.join( dfs_path, fname)} ...")
	joblib.dump(	dfs_dict, 
								os.path.join( dfs_path, f"{fname}" ),
								compress='lz4', # zlib more info: https://joblib.readthedocs.io/en/latest/auto_examples/compressors_comparison.html#sphx-glr-auto-examples-compressors-comparison-py
								)
	fsize = os.stat( os.path.join( dfs_path, f"{fname}" ) ).st_size / 1e6

	print(f">> Dumping Done => size: {fsize:.1f} MB ...")

def load_dfs(fpath=""):
	fsize = os.stat( fpath ).st_size / 1e9
	print(f">> Loading {fpath} | size: {fsize:.1f} GB ...")
	st_t = time.time()

	d = joblib.load(fpath)

	elapsed_t = time.time() - st_t

	s_df = d["search"]
	v_df = d["vol"]
	p_df = d["pg"]
	elapsed_t = time.time() - st_t
	print(f"\n>> LOADING COMPLETED in {elapsed_t:.2f} s!")

	print(f"\n>> Search_DF: {s_df.shape} Tot. missing data: {s_df.isnull().values.sum()}")
	#print( list(s_df.columns ) )
	#print()

	print(s_df.head(25))
	print("-"*130)
	print(s_df.tail(25))

	#print(s_df.isna().sum())
	#print(s_df.info(verbose=True, memory_usage='deep'))
	#print("/"*150)
	#print(s_df.dtypes)
	print("#"*130)

	print(f"\n>> Volume_DF: {v_df.shape} Tot. missing data: {v_df.isnull().values.sum()}")
	print(v_df.head(25))
	print("-"*130)
	print(v_df.tail(25))
	print("#"*130)

	print(f"\n>> Page_DF: {p_df.shape} Tot. missing data: {p_df.isnull().values.sum()}")
	print(p_df.head(25))
	print("-"*130)
	print(p_df.tail(25))
	print("#"*130)

	"""
	print( s_df["material_type"].value_counts() )
	print("-"*150)
	print( s_df["publishers"].value_counts() )
	print("-"*150)
	print( s_df["publication_places"].value_counts() )
	print("-"*150)
	print( s_df["languages"].value_counts() )
	print("-"*150)
	print( s_df["rights"].value_counts() )
	print("-"*150)
	print( s_df["fuzzy_search"].value_counts() )
	print("-"*150)
	print( s_df["illustrations"].value_counts() )
	print("-"*150)
	print( s_df["tags"].value_counts() )

	print("-"*150)
	print( s_df["authors"].value_counts() )

	print("-"*150)
	print( s_df["collections"].value_counts() )

	print("-"*150)
	print( s_df["type"].value_counts() )

	print("-"*150)
	print( s_df["require_all_search_terms"].value_counts() )

	print("-"*150)
	print( s_df["no_access_results"].value_counts() )

	print(f"\n>> Volume_DF: {v_df.shape} Tot. missing data: {v_df.isnull().values.sum()}")
	print( list(v_df.columns ) )
	print()
	print(v_df.head(5))
	print(v_df.isna().sum())
	#print(v_df.dtypes)
	print(v_df.info(verbose=True, memory_usage='deep'))
	print("#"*130)

	print(f"\n>> Page_DF: {p_df.shape} Tot. missing data: {p_df.isnull().values.sum()}")
	print( list(p_df.columns ) )
	print()
	print(p_df.head(5))
	print(p_df.isna().sum())
	print(p_df.info(verbose=True, memory_usage='deep'))
	#print(p_df.dtypes)
	print("#"*130)
	"""

	print(f"\nSearch_DF: {s_df.shape} Volume_DF: {v_df.shape} Page_DF: {p_df.shape}")
	return s_df, v_df, p_df

def plt_bar(df, name="", N=25):
	plt.figure()
	df["languages"].value_counts().sort_values(ascending=False)[:N].plot(	kind="barh", 
																																				title=f"Top {N} most frequent languages in NLF search engine")
	plt.savefig(os.path.join( rpath, f"{name}_lang.png" ), )
	plt.ylabel("Languages")
	plt.xlabel("Counts")
	
	plt.clf()

def plot_language_year(df, name="", N=6):
	o_year_unq, o_year_count = np.unique(df['date'].dt.year, return_counts=True)
	print(f"\n\n\n>> Orig DF: {o_year_unq.shape[0]}\n{o_year_unq}\n{o_year_count}\n\n")
	
	df_tmp = df.dropna(axis=0, how="any", subset=["languages"]).reset_index(drop=True)

	df_tmp['year'] = df_tmp['date'].dt.year

	#print(df_tmp.shape)
	#print(df_tmp.head(36))
	#print(list(df_tmp.columns))

	lu, lc = np.unique(df_tmp["languages"], return_counts=True)
	print(lu.shape[0], lu, lc)
	
	print(f"\n>> sorting for Top {N} ...")
	lc_sorted_idx = np.argsort(-lc)

	language_ung = lu[lc_sorted_idx][:N]
	language_counts = lc[lc_sorted_idx][:N]
	print(language_ung.shape[0], language_ung, language_counts)
	
	year_unq, year_count = np.unique(df_tmp["year"], return_counts=True)

	print(year_unq.shape[0], year_unq, year_count)

	fig, axs = plt.subplots(nrows=1, ncols=2)

	clrs = ['#1f77b4', 
					'#cc9911', 
					'#e377c2', 
					'#7f7f7f', 
					'#99ff99',
					'#ff7f0e', 
					'#16b3ff',
					'#9467bd', 
					'#d62728', 
					'#0ecd19', 
					'#ffcc99', 
					'#bcbd22', 
					'#ffc9', 
					'#17becf',
					'#2ca02c', 
					'#8c564b', 
					'#ff9999',
					]

	patches, lbls, pct_texts = axs[0].pie(language_counts,
																				labels=language_ung, 
																				autopct='%1.1f%%', 
																				#startangle=180,
																				#radius=3, USELESS IF axs[0].axis('equal')
																				#pctdistance=1.5,
																				#labeldistance=0.5,
																				rotatelabels=True,
																				#counterclock=False,
																				colors=clrs,
																				)
	for lbl, pct_text in zip(lbls, pct_texts):
		pct_text.set_rotation(lbl.get_rotation())

	axs[0].axis('equal')
	#axs[0].set_title(f"Top {N} Searched Languages in NLF")



	LANGUAGES = {}

	for l in language_ung:
			lst = []
			for y in year_unq:
					#print(l, y)
					c = df_tmp[(df_tmp["year"] == y) & (df_tmp["languages"] == l) ].languages.count()
					#print(c)
				
					lst.append(c)
			LANGUAGES[l] = lst

	print(LANGUAGES)


	WIDTH = 0.35
	BOTTOM = 0

	for k, v in LANGUAGES.items():
		#print(k, v)
		axs[1].bar(x=year_unq, 
							height=v, 
							width=WIDTH,
							bottom=BOTTOM, 
							color=clrs[list(LANGUAGES.keys()).index(k)],
							label=k,
							)
		BOTTOM += np.array(v)


	axs[1].set_ylabel('Counts')
	axs[1].set_xlabel('Year')
	#axs[1].set_title(f'Top {N} Languages by Year')

	axs[1].legend(loc="best", 
								frameon=False,
								#ncol=len(LANGUAGES), 
								)

	plt.suptitle(f"Top {N} searched languages in NLF by year")
	plt.savefig(os.path.join( rpath, f"{name}_lang_year.png" ), )

def main():
	# rename_columns: True: saving doc in english
	# rename_columns: False: saving doc in Finnish (Original) => no modification!

	#QUERY_LANGUAGE = "FINNISH"
	QUERY_LANGUAGE = "ENGLISH"

	#save_dfs(qlang=QUERY_LANGUAGE)

	search_df, vol_df, pg_df = load_dfs( fpath=os.path.join(dfs_path, f"search_vol_pg_dfs_{QUERY_LANGUAGE}.dump") )

	plt_bar( search_df, name=f"search_{QUERY_LANGUAGE}" )
	plot_language_year( search_df, name=f"search_{QUERY_LANGUAGE}" )
	basic_visualization(search_df, name=f"search_{QUERY_LANGUAGE}")
	basic_visualization(vol_df, name=f"volume_{QUERY_LANGUAGE}")
	basic_visualization(pg_df, name=f"page_{QUERY_LANGUAGE}")

if __name__ == '__main__':
	os.system('clear')
	main()