from recsys_app.recsys_src.gui_backend import *
from django.shortcuts import render, redirect
from django.http import JsonResponse
import datetime
from django.views.decorators.csrf import csrf_exempt
from uuid import uuid4
from urllib.parse import unquote
from ipware import get_client_ip
import logging

# Set up logging
logging.basicConfig(
		level=logging.INFO,
		format='%(asctime)s - %(levelname)s - %(message)s',
		handlers=[
				logging.FileHandler('user_tracking.log'),
				logging.StreamHandler()
		]
)
logger = logging.getLogger(__name__)

MAX_NUM_RECOMMENDED_TOKENS: int = 23
CURRENT_NUM_RECOMMENDED_TOKENS: int = 5
DIGI_BASE_URL: str = "https://digi.kansalliskirjasto.fi/search?requireAllKeywords=true&query="
TIMESTAMP_1ST: int=1913
TIMESTAMP_2ND: List[int]=[1914, 1934] # WWI: 	28 july 1914 – 11 nov. 1918 # pink
TIMESTAMP_3RD: List[int]=[1935, 1945] # WWII: 1 sep. 1939 – 2 sep. 1945 # blue
TIMESTAMP_END: int=1946

df = pd.DataFrame(
	columns=[
		'query_prompt',
		'recommendation_result',
		'user_feedback',
		'recommendation_result_link', 
	]
)

@csrf_exempt
def process_feedback(request):
	user_name = request.session.get('user_name', 'x_Unkown_User_x')  # Retrieve user_name from session
	if request.method == 'POST':
		print(f"Processing Feedback in progress for USER: {user_name} ...")
		try:
			data = json.loads(request.body)
			print(type(data), len(data))
			print(data)
			print(f"<>"*50)
			for entry in data:
				print(len(entry), entry)
				print("#"*50)
				df.loc[len(df)] = [
					entry['userQueryPrompt'],
					match.group(1) if (match:=re.search(r'\+\s*(\w+)', entry['recsysResultText'])) else np.nan,
					entry['userFeedback'],
					entry['recsysResultLink'], 
				]

			print(df.info(memory_usage="deep"))
			print(df)
			# Save DataFrame (if needed)
			df.to_csv(f'USER_{user_name}_Feedback.csv', index=False)
			return JsonResponse({'success': True})
		except Exception as e:
			print(f"<!> ERR: {e}")
			return JsonResponse({'success': False, 'error': str(e)})
	else:
		return JsonResponse({'success': False, 'error': 'Invalid request method'})

def submit_feedback(request):
	if request.method == 'POST':
		user_name = request.session.get('user_name', 'x_Unkown_User_x')  # Retrieve user_name from session
		input_query = request.POST.get('input_query', '')
		recommendation_results = request.POST.getlist('recommendation_results[]')
		user_feedback = request.POST.getlist('user_feedback[]')
		print(user_name)
		print(input_query)
		print(recommendation_results)
		print(user_feedback)
	# 	# Ensure data is valid
	# 	if not input_query or not recommendation_results or not user_feedback:
	# 		return JsonResponse({'success': False, 'error': 'Invalid data'})

	# 	# Construct DataFrame columns
	# 	columns = ['query_prompt'] + [f'rec{i+1}' for i in range(len(recommendation_results))]
	# 	columns += [f'user_feedback_{i+1}' for i in range(len(user_feedback))]

	# 	# Create or load DataFrame
	# 	try:
	# 		# If DataFrame exists, load it
	# 		df = pd.read_csv(f'{user_name}.csv')
	# 	except FileNotFoundError:
	# 		# Create a new DataFrame if it doesn't exist
	# 		df = pd.DataFrame(columns=columns)

	# 	# Prepare data for new row
	# 	new_data = {
	# 		'query_prompt': input_query,
	# 		**{f'rec{i+1}': rec for i, rec in enumerate(recommendation_results)},
	# 		**{f'user_feedback_{i+1}': feedback for i, feedback in enumerate(user_feedback)}
	# 	}

	# 	# Append new row to DataFrame
	# 	df = df.append(new_data, ignore_index=True)

	# 	# Save DataFrame to CSV (or database)
	# 	df.to_csv(f'{user_name}.csv', index=False)

	# 	return JsonResponse({'success': True})
	else:
		print(f"ERRRORRRR")
		return JsonResponse({'success': False, 'error': 'Invalid request method'})

def get_detailed_user_info():
	try:
		response = requests.get('http://ip-api.com/json')
		data = response.json()
		ip_info = {
			'ip_address': data['query'],
			'location': f"{data['city']}, {data['regionName']}, {data['country']}",
			'isp': data['isp'],
			'latitude': data['lat'],
			'longitude': data['lon'],
			'timezone': data['timezone'],
			'organization': data['org'],
			'as_number': data['as'],
			'as_name': data.get('asname', None),
			'mobile': data.get('mobile', False),
			'proxy': data.get('proxy', False)
		}
		ip_address = data['query']
		location = f"{data['city']}, {data['regionName']}, {data['country']}"
		isp = data['isp']
		lat, lon = data['lat'], data['lon']
		timezone = data['timezone']
		org = data['org'] # organization
		as_number = data['as']
		as_name = data.get('asname', None)
		mobile = data.get('mobile', False)
		proxy = data.get('proxy', False)
		print(f"IP Address: {ip_address} Location: {location} ISP: {isp}".center(170, "-"))
		print(f"(Latitude, Longitude): ({lat}, {lon}) Time Zone: {timezone} Organization: {org} AS Number: {as_number}, AS Name: {as_name} Mobile: {mobile}, Proxy: {proxy}")
		print("-"*170)
		return ip_info
	except requests.exceptions.RequestException as e:
		print(f"Error: {e}")

def get_user_info(request):
	"""Helper function to gather user information"""
	# Get IP address
	client_ip, is_routable = get_client_ip(request)
	
	# Get user agent
	user_agent = request.META.get('HTTP_USER_AGENT', 'Unknown')
	
	# Get session or cookie info
	user_name = request.session.get('user_name', 'x_Unknown_User_x')
	session_id = request.session.session_key
	
	# Get referrer if available
	referrer = request.META.get('HTTP_REFERER', 'Direct Access')
	
	return {
			'ip_address': client_ip if client_ip else 'Unknown IP',
			'is_routable': is_routable,
			'user_agent': user_agent,
			'user_name': user_name,
			'session_id': session_id,
			'referrer': referrer,
			'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
	}

@csrf_exempt
def track_click(request):
	if request.method == 'POST':
		try:
			# Get user information
			user_info = get_user_info(request)
			detailed_user_info = get_detailed_user_info()
			# Get click data
			data = json.loads(request.body)
			clicked_recommendation = data.get('clicked_recommendation')
			input_query = data.get('input_query')
			segment_info = data.get('segment_info')
			# Create log message
			log_message = (
				f"\nUser Activity Report\n"
				f"{'='*150}\n"
				f"Timestamp: {user_info['timestamp']}\n"
				f"IP Address: {user_info['ip_address']} (Routable: {user_info['is_routable']}) {detailed_user_info.get('ip_address')}\n"
				f"User Name: {user_info['user_name']}\n"
				f"Session ID: {user_info['session_id']}\n"
				f"User Agent: {user_info['user_agent']}\n"
				f"Referrer: {user_info['referrer']}\n"
				f"Search Query: {input_query}\n"
				f"Clicked Recommendation: {clicked_recommendation}\n"
			)
			# Add segment info if available
			if segment_info:
					log_message += (
							f"\nPie Chart Interaction\n"
							f"{'-'*80}\n"
							f"Time Range: {segment_info['timeRange']}\n"
							f"Number of Pages: {segment_info['yearlyPages']}\n"
					)
			log_message += "="*150 + "\n"
			# Log the information
			logger.info(log_message)
			# Store in session for user history
			if not request.session.get('user_history'):
				request.session['user_history'] = []
			request.session['user_history'].append(
				{
					'timestamp': user_info['timestamp'],
					'query': input_query,
					'recommendation': clicked_recommendation,
					'segment_info': segment_info
				}
			)
			return JsonResponse(
				{
					'status': 'success',
					'tracked_info': {
						'ip': user_info['ip_address'],
						'timestamp': user_info['timestamp']
					}
				}
			)
		except Exception as e:
			logger.error(f"Error tracking click: {str(e)}")
			return JsonResponse(
				{
					'status': 'error',
					'message': 'Error tracking user activity'
				}, 
				status=500,
			)
	return JsonResponse({'status': 'error', 'message': 'Invalid request method'}, status=400)

def generate_random_username():
	return f"user_{random.randint(10, 9999)}"

def check_password(request):
	if request.method == 'POST':
		user_name = request.POST.get('userNameBlock', '')
		if not user_name:
			user_name = generate_random_username()
		request.session['user_name'] = user_name  # Save user_name in session
		password = request.POST.get('password', '')
		if password == '12345':
			# next_url = request.GET.get('next', 'main_page')
			next_url = request.GET.get('next', '/home/')
			return redirect(next_url)
			# return redirect('main_page')  # Redirect to the main_page view
		else:
			return render(
				request,
				'recsys_app/password_page.html', 
				{'error_message': 'Incorrect password. Try again...'},
			)
	else:
		return render(request, 'recsys_app/password_page.html')

def main_page(request, query=None):
	# Check if the user is authenticated:
	user_name = request.session.get('user_name')  # Retrieve user_name from session

	# if not user_name:
	# 	return redirect('check_password') # Redirect to password page

	if not user_name:
		# Build the 'next' parameter to redirect the user back to their intended query page
		next_url = f"/home/query={query}" if query else "/home/"
		return redirect(f"/?next={next_url}")

	context = {
		'user_name': user_name,
		'welcome_text': "Welcome to User-based Recommendation System!<br>What are you looking after?",
		'max_length_recSys': MAX_NUM_RECOMMENDED_TOKENS,
		'curr_length_recSys': CURRENT_NUM_RECOMMENDED_TOKENS,
		'digi_base_url': DIGI_BASE_URL,
		'timestamp_1st': TIMESTAMP_1ST,
		'timestamp_2nd': TIMESTAMP_2ND,
		'timestamp_3rd': TIMESTAMP_3RD,
		'timestamp_end': TIMESTAMP_END,
	}
	print(f"Who is using the system? < {user_name} > {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}".center(180, "-"))
	if request.method == 'POST' or query:
		# Get query either from POST or URL parameter
		RAW_INPUT_QUERY = (request.POST.get('query', '') if request.method == 'POST' else unquote(query)).lower()
		if request.method == 'POST':
			# Redirect to URL with query parameter
			return redirect('main_page_with_query', query=RAW_INPUT_QUERY)

		context["input_query"] = RAW_INPUT_QUERY
		raw_query_nlf_results = get_nlf_pages(INPUT_QUERY=RAW_INPUT_QUERY)
		if raw_query_nlf_results and raw_query_nlf_results > 0 and clean_(docs=RAW_INPUT_QUERY):
			recSys_results, recSys_results_total_nlf_num_pages, recSys_results_nlf_yearly_pages = get_recsys_results(
				query_phrase=RAW_INPUT_QUERY, 
				nTokens=MAX_NUM_RECOMMENDED_TOKENS+7,
				ts_1st=TIMESTAMP_1ST,
				ts_2nd=range(TIMESTAMP_2ND[0], TIMESTAMP_2ND[1]+1, 1),
				ts_3rd=range(TIMESTAMP_3RD[0], TIMESTAMP_3RD[1]+1, 1),
				ts_end=TIMESTAMP_END,
			)
			print(f"Debug - RecSys Results: {recSys_results}")  # Debug print
			print(f"request.POST.get('isRecSys'): {request.POST.get('isRecSys')}")
			# if request.POST.get('isRecSys') == "true" and recSys_results and len(recSys_results)>0:
			if recSys_results and len(recSys_results)>0:
				context['max_length_recSys'] = min(MAX_NUM_RECOMMENDED_TOKENS, len(recSys_results))
				context['curr_length_recSys'] = min(CURRENT_NUM_RECOMMENDED_TOKENS, len(recSys_results))
				context['recsys_results_total_nlf_pages'] = recSys_results_total_nlf_num_pages[:MAX_NUM_RECOMMENDED_TOKENS]
				context['recsys_results_nlf_yearly_nPGs'] = recSys_results_nlf_yearly_pages[:MAX_NUM_RECOMMENDED_TOKENS]
				print(f"For user: < {user_name} > we found {len(recSys_results)} Recommendation Result(s):\n{recSys_results}")
				print(len(recSys_results_total_nlf_num_pages), recSys_results_total_nlf_num_pages)
				# print(len(recSys_results_nlf_yearly_pages), recSys_results_nlf_yearly_pages) # separate page numbers for pie chart
				print("*"*150)
				context['recommendation_results'] = recSys_results[:MAX_NUM_RECOMMENDED_TOKENS]
			else:
				logging.error("No recsys results Found!")
				context['recommendation_results'] = None
		else:
			logging.error("Invalid query or no results found.")
			context['recommendation_results'] = None
	return render(request, 'recsys_app/index.html', context)
	# return render(request, 'recsys_app/index_slider.html', context)

def instruction_page(request):
	return render(request, 'recsys_app/instruction.html')

def about_us_page(request):
	return render(request, 'recsys_app/about_us.html')