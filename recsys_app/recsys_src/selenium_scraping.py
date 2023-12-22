from utils import *

from bs4 import BeautifulSoup
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common import exceptions

def get_all_search_details(URL):
	st_t = time.time()

	SEARCH_RESULTS = {}
	
	options = Options()
	options.headless = True

	#options.add_argument("--remote-debugging-port=9230") # alternative: 9222
	options.add_argument("--remote-debugging-port=9222")
	options.add_argument("--no-sandbox")
	options.add_argument("--disable-gpu")
	options.add_argument("--disable-dev-shm-usage")
	options.add_argument("--disable-extensions")
	options.add_experimental_option("excludeSwitches", ["enable-automation"])
	options.add_experimental_option('useAutomationExtension', False)
	
	driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
	
	driver.get(URL)
	print(f"Scraping {driver.current_url}")
	try:
		medias = WebDriverWait(driver, 
													timeout=10,
													).until(EC.presence_of_all_elements_located((By.CLASS_NAME, 'result-row'))) # alternative for 'result-row': 'media'
		for media_idx, media_elem in enumerate(medias):
			#print(f">> Result: {media_idx}")
			outer_html = media_elem.get_attribute('outerHTML')
			
			#print(media_elem.text)
			#print()
			result = scrap_newspaper(outer_html)
			SEARCH_RESULTS[f"result_{media_idx}"] = result
			#print("-"*120)
	except (exceptions.StaleElementReferenceException,
					exceptions.NoSuchElementException,
					exceptions.TimeoutException,
					exceptions.WebDriverException,
					exceptions.SessionNotCreatedException,
					exceptions.InvalidArgumentException,
					exceptions.InvalidSessionIdException,
					exceptions.InsecureCertificateException,
					ValueError,
					TypeError,
					EOFError,
					AttributeError,
					RuntimeError,
					Exception,
					) as e:
		print(f"\t<!> Selenium: {type(e).__name__} line {e.__traceback__.tb_lineno} of {__file__}: {e.args}")
		return
	print(f"\t\t\tFound {len(medias)} media(s) => {len(SEARCH_RESULTS)} search result(s) | Elapsed_t: {time.time()-st_t:.2f} s")
	#print(json.dumps(SEARCH_RESULTS, indent=2, ensure_ascii=False))
	return SEARCH_RESULTS

def scrap_newspaper(HTML):
	query_newspaper = dict.fromkeys([
		"newspaper_title",
		"newspaper_issue", 
		"newspaper_date", 
		"newspaper_publisher", 
		"newspaper_publication_place", 
		"newspaper_page", 
		"newspaper_import_date",
		"newspaper_thumbnail",
		"newspaper_snippet",
		"newspaper_snippet_highlighted_words",
		"newspaper_content_ocr",
		"newspaper_content_ocr_highlighted_words",
		"newspaper_link",
		"newspaper_document_type",
		])

	my_parser = "lxml"
	#my_parser = "html.parser"
	soup = BeautifulSoup(HTML, my_parser)
	#print(soup.prettify())
	#return None

	all_newspaper_info = get_np_info(INP_SOUP=soup)
	#print(len(all_newspaper_info), all_newspaper_info)

	np_title = soup.find("div", class_="main-link-title link-colors")
	np_issue_date = soup.find("span", class_="font-weight-bold")

	pg_snippet = soup.find("div", class_="search-highlight-fragment ng-star-inserted")	
	pg_imported_date = soup.find("div", class_="import-date ng-star-inserted")
	thumbnail = soup.find("img")
	pg_link = soup.find("a")
	
	if thumbnail: query_newspaper["newspaper_thumbnail"] = "https://digi.kansalliskirjasto.fi" + thumbnail.get("src")
	if pg_link: query_newspaper["newspaper_link"] = "https://digi.kansalliskirjasto.fi" + pg_link.get("href")
	if np_title: query_newspaper["newspaper_title"] = np_title.text
	if pg_imported_date: query_newspaper["newspaper_import_date"] = pg_imported_date.text
	if pg_snippet:
		query_newspaper["newspaper_snippet"] = pg_snippet.text
		query_newspaper["newspaper_snippet_highlighted_words"] = [tag.text for tag in pg_snippet.findChildren('em' , recursive=False)]
	
	if all_newspaper_info[-1]: query_newspaper["newspaper_page"] = all_newspaper_info[-1].split()[1] # remove sivu, sida page: ex) sivu 128 => 128
	#if all_newspaper_info[1]: query_newspaper["newspaper_issue"] = all_newspaper_info[1]
	#if all_newspaper_info[2]: query_newspaper["newspaper_date"] = all_newspaper_info[2]
	if all_newspaper_info[3]: query_newspaper["newspaper_publisher"] = all_newspaper_info[3]
	if all_newspaper_info[4]: query_newspaper["newspaper_publication_place"] = all_newspaper_info[4]
	
	# OCR Content:
	if pg_link: 
		ttl, dtyp, issue, publisher, pub_date, pub_place, lang, trm, hw, pg, txt = scrap_ocr_page_content(query_newspaper["newspaper_link"])
		query_newspaper["newspaper_content_ocr"] = txt
		query_newspaper["newspaper_content_ocr_highlighted_words"] = hw
		query_newspaper["newspaper_issue"] = issue
		query_newspaper["newspaper_date"] = pub_date
		query_newspaper["newspaper_document_type"] = dtyp
		query_newspaper["newspaper_language"] = lang
	return query_newspaper

def get_np_info(INP_SOUP):	
	selectors = [
		'span.badge.badge-secondary.ng-star-inserted',                                             #badge 
		'span.font-weight-bold span.ng-star-inserted:has(span[translate])',                        #issue 
		'span.font-weight-bold span.ng-star-inserted:last-child',                                  #date 
		'span.font-weight-bold ~ span.ng-star-inserted:-soup-contains(", ")',                      #publisher 
		'span.font-weight-bold ~ span.ng-star-inserted:-soup-contains(", ") + span.ng-star-inserted:-soup-contains(", ")', #city 
		'span.font-weight-bold ~ span.ng-star-inserted:-soup-contains(":")',                       #page 
	]
	desired_list = [ None if s[0] is None else ( s[0].get_text(' ', strip=True)[2:] if '-soup-contains' in s[1] else s[0].get_text(' ', strip=True)) for s in [ ( INP_SOUP.select_one(sel), sel) for sel in selectors] ]
	return desired_list

def scrap_clipping_page(URL):
	#print(f"Scraping clipping page: {URL}")
	parsed_url, parameters = get_parsed_url_parameters(URL)
	#print(f"Parameters:\n{json.dumps(parameters, indent=2, ensure_ascii=False)}")
	#print()
	#print(f"parsed_url : {parsed_url}")

	offset_pg=(int(parameters.get('page')[0])-1)*20 if "page=" in URL else 0
	clipping_pg_api = f"https://digi.kansalliskirjasto.fi/rest/article-search/search-by-type?offset={offset_pg}&count=20"
	payload = {	"categoryIds": parameters.get('categoryId') if parameters.get('categoryId') else [],
							"collections": parameters.get('collection') if parameters.get('collection') else [],
							"endDate": parameters.get('endDate')[0] if parameters.get('endDate') else None,
							"exactCollectionMaterialType": "false", # TODO: must be investigated!!
							"fuzzy": parameters.get('fuzzy')[0] if parameters.get('fuzzy') else "false",
							"generalTypes": parameters.get('formats') if parameters.get('formats') else [],
							"includeCollected": parameters.get('includeCollected')[0] if parameters.get('includeCollected') else "false",
							"keywords": parameters.get('keyword') if parameters.get('keyword') else [],
							"onlyCollected": "false", # TODO: must be investigated!!
							"orderBy": parameters.get('orderBy')[0] if parameters.get('orderBy') else "CREATED_DESC",
							"query": parameters.get('query')[0] if parameters.get('query') else "",
							"queryTargetsMetadata": parameters.get('qMeta')[0] if parameters.get('qMeta') else "false",
							"queryTargetsOcrText": parameters.get('qOcr')[0] if parameters.get('qOcr') else "true",
							"requireAllKeywords": parameters.get('requireAllKeywords')[0]  if parameters.get('requireAllKeywords') else "false",
							"startDate": parameters.get('startDate')[0] if parameters.get('startDate') else None,
							"subjectIds": parameters.get('subjectId') if parameters.get('subjectId') else [],
							"titles": parameters.get('title') if parameters.get('title') else [],
							}
	headers = {	'Content-type': 'application/json',
							'Accept': 'application/json; text/plain; */*', 
							'Cache-Control': 'no-cache',
							'Connection': 'keep-alive',
							'Pragma': 'no-cache',
							}

	try:
		st_t = time.time()
		r = requests.post(url=clipping_pg_api, 
											json=payload, 
											headers=headers,
											)
		res = r.json()
		#print(res.keys())
		CLIPPING_RESULTS = res.get("rows")
		#print(f"\t\tFound {len(CLIPPING_RESULTS)} clipping result(s) | Elapsed_t: {time.time()-st_t:.2f} s")
		#print(json.dumps(CLIPPING_RESULTS, indent=2, ensure_ascii=False))
	except (requests.exceptions.Timeout,
					requests.exceptions.ConnectionError, 
					requests.exceptions.RequestException, 
					requests.exceptions.TooManyRedirects,
					requests.exceptions.InvalidSchema,
					ValueError, 
					TypeError, 
					EOFError, 
					RuntimeError,
					json.JSONDecodeError,
					json.decoder.JSONDecodeError,
					Exception, 
				) as e:
		print(f"{type(e).__name__} line {e.__traceback__.tb_lineno} in {__file__}: {e.args}")
		return
	return CLIPPING_RESULTS

def scrap_collection_page(URL):
	#print(f"Scraping collection page: {URL}")
	st_t = time.time()
	parsed_url, parameters = get_parsed_url_parameters(URL)
	#print(f"Parsed url:\n{json.dumps(parameters, indent=2, ensure_ascii=False)}")
	#print()
	#print(f"parsed_url : {parsed_url}")

	offset_pg=(int(parameters.get('page')[0])-1)*20 if "page=" in URL else 0
	collection_pg_api = f"https://digi.kansalliskirjasto.fi/rest/binding-search/search/binding?offset={offset_pg}&count=200"
	#print(collection_pg_api)
	payload = {	"authors": parameters.get('author') if parameters.get('author') else [],
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
							"orderBy": parameters.get('orderBy')[0] if parameters.get('orderBy') else "DATE_DESC",
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
	headers = {	'Content-type': 'application/json',
							'Accept': 'application/json; text/plain; */*', 
							'Cache-Control': 'no-cache',
							'Connection': 'keep-alive',
							'Pragma': 'no-cache',
							}

	try:
		r = requests.post(url=collection_pg_api, 
											json=payload, 
											headers=headers,
											)
		res = r.json()
		#print(res.keys())
		COLLECTION_RESULTS = res.get("rows")
		#print(f"\t\tFound {len(COLLECTION_RESULTS)} search result(s) | Elapsed_t: {time.time()-st_t:.2f} s")
		#print(json.dumps(COLLECTION_RESULTS, indent=2, ensure_ascii=False))
	except (requests.exceptions.Timeout,
					requests.exceptions.ConnectionError, 
					requests.exceptions.RequestException, 
					requests.exceptions.TooManyRedirects,
					requests.exceptions.InvalidSchema,
					ValueError, 
					TypeError, 
					EOFError, 
					RuntimeError,
					json.JSONDecodeError,
					json.decoder.JSONDecodeError,
					Exception, 
					) as e:
		print(f"{type(e).__name__} line {e.__traceback__.tb_lineno} in {__file__}: {e.args}")
		return
	return COLLECTION_RESULTS

def scrap_search_page(URL):
	#print(f"Scraping: {URL}")
	st_t = time.time()
	parsed_url, parameters = get_parsed_url_parameters(URL)
	#print(f"Parsed url:\n{json.dumps(parameters, indent=2, ensure_ascii=False)}")
	#print()

	#print(f"parsed_url : {parsed_url}")

	offset_pg=(int(parameters.get('page')[0])-1)*20 if "page=" in URL else 0
	
	search_pg_api = f"https://digi.kansalliskirjasto.fi/rest/binding-search/search/binding?offset={offset_pg}&count=20"
	#print(search_pg_api)
	
	payload = {	"authors": parameters.get('author') if parameters.get('author') else [],
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
	
	headers = {	'Content-type': 'application/json',
							'Accept': 'application/json; text/plain; */*', 
							'Cache-Control': 'no-cache',
							'Connection': 'keep-alive',
							'Pragma': 'no-cache',
							}

	try:
		r = requests.post(url=search_pg_api, 
											json=payload, 
											headers=headers,
											)

		#print(r.headers)
		#print(r.status_code)

		res = r.json()
		# a list of up to 20 results, each of which contains: 
		#print(res.keys()): ['bindingId', 'bindingTitle', 'publicationId', 'generalType', 'authorized', 'authors', 'pageNumber', 'language', 'publisher', 'issue', 'importDate', 'dateAccuracy', 'placeOfPublication', 'textHighlights', 'terms', 'score', 'url', 'thumbnailUrl', 'date']
		SEARCH_RESULTS = res.get("rows") 
		#print(f"\t\tFound {len(SEARCH_RESULTS)} search result(s) | Elapsed_t: {time.time()-st_t:.2f} s")
		#print(json.dumps(SEARCH_RESULTS, indent=2, ensure_ascii=False))
	except (requests.exceptions.Timeout,
					requests.exceptions.ConnectionError, 
					requests.exceptions.RequestException, 
					requests.exceptions.TooManyRedirects,
					requests.exceptions.InvalidSchema,
					ValueError, 
					TypeError, 
					EOFError, 
					RuntimeError,
					json.JSONDecodeError,
					json.decoder.JSONDecodeError,
					Exception, 
					) as e:
		print(f"{type(e).__name__} line {e.__traceback__.tb_lineno} in {__file__}: {e.args}")
		return
	return SEARCH_RESULTS

def scrap_ocr_page_content(URL):
	print(f"Scraping newspaper content page: {URL}")

	if "&page=" in URL:
		up_url = URL
	else:
		up_url = f"{URL}&page=1"

	#print(f"\tUpdated: {up_url}")
	parsed_url, parameters = get_parsed_url_parameters(up_url)
	#print(f"Parsed url | OCR extraction: {parameters}")
	#print(f"parsed_url : {parsed_url}")
	
	api_url = f"https://digi.kansalliskirjasto.fi/rest/binding-search/ocr-hits/{parsed_url.path.split('/')[-1]}"
	try:
		hgltd_wrds = [d.get("text") for d in requests.get(api_url, params=parameters).json()]
	except json.JSONDecodeError as jve:
		print(f"JSON empty response:\n{jve}")
		hgltd_wrds = []
	api_nwp = f"https://digi.kansalliskirjasto.fi/rest/binding?id={parsed_url.path.split('/')[-1]}"
	nwp_info = requests.get(api_nwp).json()
	
	#print(list(nwp_info.get("bindingInformation").keys()))
	#print(list(nwp_info.get("bindingInformation").get("citationInfo").keys()))
	#print(nwp_info.get("bindingInformation").get("citationInfo").get("refWorksLanguage"))
	#print(nwp_info.get("bindingInformation").get("citationInfo").get("refWorksOutputLanguage")) # English (30)
	#print()
	
	title = nwp_info.get("bindingInformation").get("publicationTitle") # Uusi Suometar 
	doc_type = nwp_info.get("bindingInformation").get("generalType") # NEWSPAER
	issue = nwp_info.get("bindingInformation").get("issue") # 63
	publisher = nwp_info.get("bindingInformation").get("citationInfo").get("publisher") # Uuden Suomettaren Oy
	pub_date = nwp_info.get("bindingInformation").get("citationInfo").get("localizedPublishingDate") # 16.03.1905
	pub_place = nwp_info.get("bindingInformation").get("citationInfo").get("publishingPlace") # Helsinki, Suomi
	lang = nwp_info.get("bindingInformation").get("citationInfo").get("refWorksLanguage") # English

	txt_pg_url = f"{parsed_url.scheme}://{parsed_url.netloc}{parsed_url.path}/page-{parameters.get('page')[0]}.txt"
	ocr_api_url = f"https://digi.kansalliskirjasto.fi/rest/binding/ocr-data?bindingId={parsed_url.path.split('/')[-1]}&page={parameters.get('page')[0]}&oldOcr=false"
	#print(f">> ocr_api_url: {ocr_api_url}")
	text_response = checking_(txt_pg_url)
	if text_response: # 200
		txt = text_response.text
	else:
		txt = None

	return title, doc_type, issue, publisher, pub_date, pub_place, lang, parameters.get("term"), hgltd_wrds, parameters.get("page"), txt

def scrap_newspaper_content_page(URL):
	print(f"URL: {URL:<150}", end="")
	st_t = time.time()
	NWP_CONTENT_RESULTS = {}

	if "&page=" in URL:
		up_url = URL
	else:
		up_url = f"{URL}&page=1"

	#print(f"\t\tUpdated: {up_url}")
	parsed_url, parameters = get_parsed_url_parameters(up_url)
	#print(json.dumps(parameters, indent=2, ensure_ascii=False))
	#print(f"parsed_url : {parsed_url}")
	
	ocr_api_url = f"https://digi.kansalliskirjasto.fi/rest/binding/ocr-data?bindingId={parsed_url.path.split('/')[-1]}&page={parameters.get('page')[0]}&oldOcr=false"
	txt_pg_url = f"{parsed_url.scheme}://{parsed_url.netloc}{parsed_url.path}/page-{parameters.get('page')[0]}.txt"	
	#print(f"<> ocr_api_url: {ocr_api_url}")
	text_response = checking_(txt_pg_url)
	if text_response: # 200
		NWP_CONTENT_RESULTS["text"] = text_response.text

	api_url = f"https://digi.kansalliskirjasto.fi/rest/binding-search/ocr-hits/{parsed_url.path.split('/')[-1]}"
	rs_api_url = checking_(url=api_url, prms=parameters)
	
	try:
		hgltd_wrds = [d.get("text") for d in rs_api_url.json()]
		# if rs:
		# 	hgltd_wrds = [d.get("text") for d in rs.json()]
	except (json.JSONDecodeError,
					json.decoder.JSONDecodeError,
					Exception,
				) as e:
		print(f"<!> {e}")
		hgltd_wrds = []

	api_nwp = f"https://digi.kansalliskirjasto.fi/rest/binding?id={parsed_url.path.split('/')[-1]}"
	try:
		#nwp_info = requests.get(api_nwp).json() # check 
		r = checking_(url=api_nwp, prms=None)
		if r: # 200
			nwp_info = r.json()
		
			#print(list(nwp_info.get("bindingInformation").keys()))
			#print(list(nwp_info.get("bindingInformation").get("citationInfo").keys()))
			#print(nwp_info.get("bindingInformation").get("citationInfo").get("refWorksLanguage"))
			#print(nwp_info.get("bindingInformation").get("citationInfo").get("refWorksOutputLanguage")) # English (30)
			#print()
			"""
			title = nwp_info.get("bindingInformation").get("publicationTitle") # Uusi Suometar 
			doc_type = nwp_info.get("bindingInformation").get("generalType") # NEWSPAER
			issue = nwp_info.get("bindingInformation").get("issue") # 63
			publisher = nwp_info.get("bindingInformation").get("citationInfo").get("publisher") # Uuden Suomettaren Oy
			pub_date = nwp_info.get("bindingInformation").get("citationInfo").get("localizedPublishingDate") # 16.03.1905
			pub_place = nwp_info.get("bindingInformation").get("citationInfo").get("publishingPlace") # Helsinki, Suomi
			lang = nwp_info.get("bindingInformation").get("citationInfo").get("refWorksLanguage") # English
			"""

			NWP_CONTENT_RESULTS["title"] = nwp_info.get("bindingInformation").get("publicationTitle") # Uusi Suometar 
			NWP_CONTENT_RESULTS["document_type"] = nwp_info.get("bindingInformation").get("generalType") # NEWSPAER
			NWP_CONTENT_RESULTS["issue"] = nwp_info.get("bindingInformation").get("issue") # 63
			NWP_CONTENT_RESULTS["publisher"] = nwp_info.get("bindingInformation").get("citationInfo").get("publisher") # Uuden Suomettaren Oy
			NWP_CONTENT_RESULTS["publication_date"] = nwp_info.get("bindingInformation").get("citationInfo").get("localizedPublishingDate") # 16.03.1905
			NWP_CONTENT_RESULTS["publication_place"] = nwp_info.get("bindingInformation").get("citationInfo").get("publishingPlace") # Helsinki, Suomi
			NWP_CONTENT_RESULTS["language"] = nwp_info.get("bindingInformation").get("citationInfo").get("refWorksLanguage") # English
			NWP_CONTENT_RESULTS["parsed_term"] = parameters.get("term")
			NWP_CONTENT_RESULTS["highlighted_term"] = hgltd_wrds
			NWP_CONTENT_RESULTS["page"] = parameters.get("page")
	except (requests.exceptions.Timeout,
					requests.exceptions.ConnectionError, 
					requests.exceptions.RequestException, 
					requests.exceptions.TooManyRedirects,
					requests.exceptions.InvalidSchema,
					json.decoder.JSONDecodeError,
					json.JSONDecodeError,
					Exception, 
					ValueError, 
					TypeError, 
					EOFError, 
					RuntimeError,
	) as e:
		print(f"{type(e).__name__} line {e.__traceback__.tb_lineno} in {__file__}: {e.args}\n{up_url}")
		return
		"""
		title = None
		doc_type = None
		issue = None
		publisher = None
		pub_date = None
		pub_place = None
		lang = None
		"""

	#return title, doc_type, issue, publisher, pub_date, pub_place, lang, parameters.get("term"), hgltd_wrds, parameters.get("page"), txt
	print(f"\tElapsed_t: {time.time()-st_t:.3f} sec")
	return NWP_CONTENT_RESULTS

if __name__ == '__main__':
	os.system("clear")
	#url = 'https://digi.kansalliskirjasto.fi/search?query=sj%C3%A4lvst%C3%A4ndighetsdag&formats=NEWSPAPER&formats=JOURNAL&formats=PRINTING&formats=BOOK&formats=MANUSCRIPT&formats=MAP&formats=MUSIC_NOTATION&formats=PICTURE&formats=CARD_INDEX&orderBy=RELEVANCE'
	#url = 'https://digi.kansalliskirjasto.fi/search?page=36&query=kantasonni&formats=NEWSPAPER&orderBy=RELEVANCE'

	# clippings:
	url = 'https://digi.kansalliskirjasto.fi/clippings?query=economic%20crisis&fuzzy=true&formats=NEWSPAPER&formats=JOURNAL&startDate=2023-02-21&endDate=2023-02-28&categoryId=7&categoryId=6&subjectId=6&subjectId=9&subjectId=11&title=fk20100613&title=fk20100478&keyword=%20%20%20%20turvattomuus&keyword=%20Alma%20Josefina%20Jakovaara&collection=24&collection=475&collection=477&orderBy=RELEVANCE&includeCollected=true&resultMode=THUMB'
	#url = 'https://digi.kansalliskirjasto.fi/clippings?query=sj%C3%A4lvstandighetsdag&orderBy=RELEVANCE&resultMode=THUMB'
	#url = 'https://digi.kansalliskirjasto.fi/clippings?orderBy=RELEVANCE&resultMode=THUMB'
	
	#url = "https://digi.kansalliskirjasto.fi/search?fuzzy=true&qOcr=false&qMeta=true&hasIllustrations=true&showUnauthorizedResults=true&pages=1-15&query=sj%C3%A4lvstandighetsdag&importTime=CUSTOM&startDate=1918-12-12&endDate=1918-12-26&importStartDate=2023-02-14&collection=24&collection=681&author=-i.%20-n&author=-a%20-g%201807-1879.&author=-HM&tag=udk:012:929Tiilil%C3%A4,Osmo&tag=udk:001.891&publisher=10.%20Pr.&publisher=13.%20Prikaati&publicationPlace=Alavojakkala&publicationPlace=Alavus&publicationPlace=Amsterdam&title=fd2015-pp00007208&title=fd2020-00026250&lang=CZE&lang=CHU&formats=PRINTING&formats=BOOK&formats=CARD_INDEX&orderBy=RELEVANCE"
	#url = 'https://digi.kansalliskirjasto.fi/search?query=economic%20crisis&orderBy=RELEVANCE'
	#url = 'https://digi.kansalliskirjasto.fi/search?query=sj%C3%A4lvst%C3%A4ndighetsdag&formats=JOURNAL&orderBy=RELEVANCE' # <6 : 4
	#url = 'https://digi.kansalliskirjasto.fi/search?page=62&query=Katri%20ikonen%20&orderBy=DATE'
	#url = 'https://digi.kansalliskirjasto.fi/search?query=%22TAAVI%20ELIAKSENPOIKA%22%20AND%20%22SIPPOLA%22&formats=NEWSPAPER&orderBy=RELEVANCE' # no result is returned! => timeout!
	#scrap_search_page(URL=url)
	#scrap_ocr_page_content(URL='https://digi.kansalliskirjasto.fi/sanomalehti/binding/2247833?term=självständighetsdagen&term=Självständighetsdagen')
	scrap_clipping_page(URL=url)