from recsys_app.recsys_src.gui_backend import *
from django.shortcuts import render, redirect
from django.http import JsonResponse
import datetime
from django.views.decorators.csrf import csrf_exempt
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
			return redirect('main_page')  # Redirect to the main_page view
		else:
			return render(
				request,
				'recsys_app/password_page.html', 
				{'error_message': 'Incorrect password. Try again...'},
			)
	else:
		return render(request, 'recsys_app/password_page.html')

@csrf_exempt
def track_click(request):
	if request.method == 'POST':
		user_name = request.session.get('user_name', 'x_Unkown_User_x')  # Retrieve user_name from session
		data = json.loads(request.body)
		clicked_recommendation = data.get('clicked_recommendation')
		input_query = data.get('input_query')
		segment_info = data.get('segment_info')
		print(f"\nUser < {user_name} > searched for query: < {input_query} > clicked on Recommendation: < {clicked_recommendation} >")
		if segment_info:
			print(f" [Interested in] Pie chart Year Distribution ".center(80, "+"))
			print(
				f"Segment: < {segment_info['timeRange']} > "
				f"|nPGs| = {segment_info['yearlyPages']}"
				.center(80, " ")
			)
		print("#"*140)
		return JsonResponse({'status': 'success'})
	return JsonResponse({'status': 'error'}, status=400)

def main_page(request):
	user_name = request.session.get('user_name', 'x_Unkown_User_x')  # Retrieve user_name from session
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
	if request.method == 'POST':
		RAW_INPUT_QUERY = request.POST.get('query', '').lower()
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
			if request.POST.get('isRecSys') == "true" and recSys_results and len(recSys_results)>0:
				context['max_length_recSys'] = min(MAX_NUM_RECOMMENDED_TOKENS, len(recSys_results))
				context['curr_length_recSys'] = min(CURRENT_NUM_RECOMMENDED_TOKENS, len(recSys_results))
				context['recsys_results_total_nlf_pages'] = recSys_results_total_nlf_num_pages[:MAX_NUM_RECOMMENDED_TOKENS]
				context['recsys_results_nlf_yearly_nPGs'] = recSys_results_nlf_yearly_pages[:MAX_NUM_RECOMMENDED_TOKENS]
				print(f"For user: < {user_name} > we found {len(recSys_results)} Recommendation Result(s):\n{recSys_results}")
				print(len(recSys_results_total_nlf_num_pages), recSys_results_total_nlf_num_pages)
				print(len(recSys_results_nlf_yearly_pages), recSys_results_nlf_yearly_pages)
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