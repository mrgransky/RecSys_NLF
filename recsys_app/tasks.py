# recsys_app/tasks.py
from celery import shared_task
from recsys_app.recsys_src.gui_backend import get_recsys_results

@shared_task
def async_get_recsys_results(query_phrase: str, nTokens: int = 5):
	return get_recsys_results(query_phrase, nTokens)