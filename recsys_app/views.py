from django.shortcuts import render
from PIL import Image, ImageTk
import base64
from io import BytesIO
import requests
digi_base_url = "https://digi.kansalliskirjasto.fi/search"

def main_page(request):
	left_image_path = "https://www.topuniversities.com/sites/default/files/profiles/logos/tampere-university_5bbf14847d023f5bc849ec9a_large.jpg"
	right_image_path = "https://netpreserve.org/resources/logo_KK.fi_-150x150.png"
	
	left_image = Image.open(BytesIO(requests.get(left_image_path).content)).resize((200, 200), Image.Resampling.LANCZOS)
	right_image = Image.open(BytesIO(requests.get(right_image_path).content)).resize((200, 200), Image.Resampling.LANCZOS)

	tk_left_image = pil_image_to_base64(left_image)
	tk_right_image = pil_image_to_base64(right_image)

	context = {
		'tk_left_image': tk_left_image,
		'tk_right_image': tk_right_image,
		'welcome_text': "Welcome to User-based Recommendation System!<br>What are you looking after?",
	}

	if request.method == 'POST':
		query = request.POST.get('query', '')
		# Do something with the entered query, for example, print it
		print(f'User entered query: {query}')

		# Assuming you want to generate a link to the National Library based on the entered query
		library_link = f'{digi_base_url}?query={query}'

		# Include the link in the context
		context['library_link'] = library_link

	return render(request, 'recsys_app/index.html', context)

def pil_image_to_base64(pil_image):
	rgb_image = pil_image.convert('RGB')
	buffered = BytesIO()
	rgb_image.save(buffered, format="JPEG")
	img_str = base64.b64encode(buffered.getvalue()).decode()
	return img_str