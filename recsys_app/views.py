from django.shortcuts import render
import numpy as np

digi_base_url = "https://digi.kansalliskirjasto.fi/search"

def get_recsys_result(qu: str="This is a sample query phrase!"):
	return [f"Token_{i+1}" for i in np.arange(10)]


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
		library_link = f'{digi_base_url}?query={query}'
		context['library_link'] = library_link
		context['recommendation_result'] = get_recsys_result(qu=query),
	return render(request, 'recsys_app/index.html', context)