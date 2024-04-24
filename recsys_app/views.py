import random
from django.shortcuts import render, redirect
from recsys_app.recsys_src.gui_backend import *

USR: str = "XXXXXX"
MAX_NUM_RECOMMENDED_TOKENS: int = 20

def generate_random_username():
	return f"user_{random.randint(100, 999)}"

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
			return render(request, 
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
	}
	if request.method == 'POST':
		query = request.POST.get('query', '')
		context["input_query"] = query
		if request.POST.get('isRecSys') == "true":
			recSys_results = get_recsys_results(query_phrase=query, nTokens=100)
			# print(f"recSys_results: {len(recSys_results)}")
			if recSys_results and len(recSys_results)>0:
				context['max_length_recSys'] = min(MAX_NUM_RECOMMENDED_TOKENS, len(recSys_results))
			# print(f">> RecSys POST entered qu: {query} request.POST.get('isRecSys'): {request.POST.get('isRecSys')}")
			# context['recommendation_results'] = [f"Token_{i}" for i in range(20)]
			context['recommendation_results'] = recSys_results
		else:
			print(f"ERROORRR!")
	return render(request, 'recsys_app/index.html', context)

def help_page(request):
	return render(request, 'recsys_app/help.html')

def about_us_page(request):
	return render(request, 'recsys_app/about_us.html')