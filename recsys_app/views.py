from django.shortcuts import render
from recsys_app.recsys_src.gui_backend import *


def about_us_page(request):
	return render(request, 'recsys_app/about_us.html')

def main_page(request):
	left_image_url = "https://www.topuniversities.com/sites/default/files/profiles/logos/tampere-university_5bbf14847d023f5bc849ec9a_large.jpg"
	right_image_url = "https://netpreserve.org/resources/logo_KK.fi_-150x150.png"
	context = {
		'left_image_url': left_image_url,
		'right_image_url': right_image_url,
		'welcome_text': "Welcome to User-based Recommendation System!<br>What are you looking after?",
	}
	if request.method == 'POST':
		query = request.POST.get('query', '')
		context["input_query"] = query
		if request.POST.get('isRecSys')=="true":
			# print(f">> RecSys POST entered qu: {query} request.POST.get('isRecSys'): {request.POST.get('isRecSys')}")
			context['recommendation_results'] = get_recsys_results(query_phrase=query, nTokens=15)
			# context['recommendation_results'] = [f"Token_{i}" for i in range(10)]
		else:
			print(f"ERROORRR!")
	return render(request, 'recsys_app/index.html', context)