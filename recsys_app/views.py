from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from recsys_app.recsys_src.gui_backend import *

USR: str = "XXXXXX"
MAX_NUM_RECOMMENDED_TOKENS: int = 20
CURRENT_NUM_RECOMMENDED_TOKENS: int = 5
DIGI_BASE_URL: str = "https://digi.kansalliskirjasto.fi/search?requireAllKeywords=true&query="
# recSys_results_nlf_num_pages: List[int] = [10, 11, 42, 55, 88, 155, 250, 922, 2, 426]

df = pd.DataFrame(
	columns=[
		'query_prompt',
		'recommendation_result',
		'user_feedback',
		'recommendation_result_link', 
	]
)

def generate_random_username():
	return f"user_{random.randint(100, 999)}"

@csrf_exempt
def process_feedback(request):
	if request.method == 'POST':
		print(f"Processing Feedback in progress for USER: {USR} ...")
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
			df.to_csv(f'USER_{USR}_Feedback.csv', index=False)
			return JsonResponse({'success': True})
		except Exception as e:
			print(f"<!> ERR: {e}")
			return JsonResponse({'success': False, 'error': str(e)})
	else:
		return JsonResponse({'success': False, 'error': 'Invalid request method'})

def submit_feedback(request):
	if request.method == 'POST':
		# user_name = request.POST.get('user_name', '')  # Assuming 'user_name' is passed with the request
		user_name = USR
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

def check_password(request):
	global USR
	if request.method == 'POST':
		user_name = request.POST.get('user_name', '')
		if not user_name:
			user_name = generate_random_username()
		USR = user_name
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

def main_page(request):
	context = {
		'user_name': USR,
		'welcome_text': "Welcome to User-based Recommendation System!<br>What are you looking after?",
		'max_length_recSys': MAX_NUM_RECOMMENDED_TOKENS,
		'curr_length_recSys': CURRENT_NUM_RECOMMENDED_TOKENS,
		'digi_base_url': DIGI_BASE_URL,
	}
	if request.method == 'POST':
		query = request.POST.get('query', '')
		context["input_query"] = query
		raw_query_nlf_results = get_num_results(URL=f"{BASE_DIGI_URL}" + urllib.parse.quote_plus(query))
		# print(f'we found {raw_query_nlf_results} docs in NLF')
		if request.POST.get('isRecSys') == "true" and raw_query_nlf_results > 0:
			# print(f">> RecSys POST entered qu: {query} request.POST.get('isRecSys'): {request.POST.get('isRecSys')}")
			recSys_results, recSys_results_nlf_num_pages = get_recsys_results(query_phrase=query, nTokens=100)
			
			if recSys_results and len(recSys_results)>0:
				context['max_length_recSys'] = min(MAX_NUM_RECOMMENDED_TOKENS, len(recSys_results))
				context['curr_length_recSys'] = min(CURRENT_NUM_RECOMMENDED_TOKENS, len(recSys_results))
				context['recommendation_results_nlf_found_pages'] = recSys_results_nlf_num_pages[:MAX_NUM_RECOMMENDED_TOKENS]
			# context['recommendation_results'] = [f"Token_{i}" for i in range(20)]
			print(recSys_results, recSys_results_nlf_num_pages)
			context['recommendation_results'] = recSys_results[:MAX_NUM_RECOMMENDED_TOKENS]
		else:
			print(f"ERROORRR! => must go to alert!!")
			context['recommendation_results'] = None
	return render(request, 'recsys_app/index.html', context)

def help_page(request):
	return render(request, 'recsys_app/help.html')

def about_us_page(request):
	return render(request, 'recsys_app/about_us.html')