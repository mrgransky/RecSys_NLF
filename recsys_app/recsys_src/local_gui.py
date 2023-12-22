import tkinter as tk
from tkinter import messagebox
import urllib
from PIL import Image, ImageTk
import webbrowser
import time
import requests
from io import BytesIO

def close_window(count=8):
	if count > 0:
		countdown_lbl.config(text=f"Thanks for using our service, Have a Good Day!\n\nclosing in {count} sec...")
		root.after(1000, close_window, count-1)
	else:
		root.destroy()

def on_entry_click(event):
	query = entry.get()
	if query == "Query keywords...":
		entry.delete(0, tk.END)
		entry.config(fg='black')  # Change text color to black

def on_entry_leave(event):
	query = entry.get()
	if query == "":
		entry.insert(0, "Query keywords...")
		entry.config(fg='grey')  # Change text color to gray

def generate_link():
	query = entry.get()
	if query and query != "Query keywords...":
		encoded_query = urllib.parse.quote(query)
		base_url = "https://digi.kansalliskirjasto.fi/search"
		link = f"{base_url}?query={encoded_query}"
		nlf_link_lable.config(text=f"NLF: {link}", fg='blue', cursor='hand2', font = "Helvetica 10")
		nlf_link_lable.bind("<Button-1>", lambda e: webbrowser.open(link))
	else:
		nlf_link_lable.config(text="Enter a valid search query to proceed!", fg='red', )
		messagebox.showerror('Error', 'Enter a valid search query to proceed!')

def recSys_cb():
	query = entry.get()
	if query and query != "Query keywords...":
		recys_lbl.config(text=f"Since You searched < {query} >\nYou might be interested in: TK1, TK2, TK3, TK4", fg='green', font = "Helvetica 15 bold")
	else:
		recys_lbl.config(text="Enter a valid search query first", fg='red', )

def trending_today_cb():
	trn_lemmas="Suomi | Helsinki | "
	trn_lbl=tk.Label(root, text=trn_lemmas )

def clean_search_entry():
	entry.delete(0, "end")  # delete all the text in the entry
	entry.insert(0, 'Query keywords...')
	entry.config(fg='grey')
	nlf_link_lable.destroy()

def clean_recsys_entry():
	return

root = tk.Tk()
root.title("TAU | National Library of Finland Recommendation System")
root.geometry('990x1050')

left_image_path="https://www.topuniversities.com/sites/default/files/profiles/logos/tampere-university_5bbf14847d023f5bc849ec9a_large.jpg"
right_image_path="https://digi.kansalliskirjasto.fi/images/logos/logo_fi_darkblue.png"
left_image = Image.open(BytesIO(requests.get(left_image_path).content)).resize((180, 120), Image.Resampling.LANCZOS)
right_image = Image.open(BytesIO(requests.get(right_image_path).content)).resize((180, 120), Image.Resampling.LANCZOS)

# load images from directory
# base_dir="/home/farid/Hämtningar/"
# left_image_path = base_dir+"tau.jpg"
# right_image_path = base_dir+"nlf.png"
# left_image = Image.open(left_image_path).resize((150, 150), Image.Resampling.LANCZOS)
# right_image = Image.open(right_image_path).resize((150, 150), Image.Resampling.LANCZOS)

tk_left_image = ImageTk.PhotoImage(left_image)
tk_right_image = ImageTk.PhotoImage(right_image)

left_image_lbl = tk.Label(root, image=tk_left_image)
right_image_lbl = tk.Label(root, image=tk_right_image)

welcome_lbl = tk.Label(root, text="Welcome!\nWhat are you looking after, today?", font=('Georgia 13'), justify="center")

entry = tk.Entry(root, width=120, fg='grey', bg="purple", borderwidth=5, font=('Georgia 12'))
entry.insert(0, "Query keywords...")

entry.bind('<FocusIn>', on_entry_click)
entry.bind('<FocusOut>', on_entry_leave)

search_btn=tk.Button(root, text="Search NLF", width=20, command=generate_link)
clean_search_btn = tk.Button(root, text="Clean", width=20, command=clean_search_entry)
nlf_link_lable = tk.Label(root, text="", fg='blue', cursor='hand2')

entry.bind('<FocusIn>', on_entry_click)
entry.bind('<FocusOut>', on_entry_leave)

rec_btn = tk.Button(root, text="Recommend Me", width=20, command=recSys_cb)
clean_recsys_btn = tk.Button(root, text="Clear", width=20, command=clean_recsys_entry)
recys_lbl = tk.Label(root, text="")

exit_btn = tk.Button(root, text="Exit", width=15, command=lambda: close_window())
countdown_lbl = tk.Label(root, text="")

left_image_lbl.grid(row=0, column=0, pady=20, padx=0,)
right_image_lbl.grid(row=0, column=2, pady=20, padx=0,)

welcome_lbl.grid(row=1, column=1, padx=0, pady=0)

entry.grid(row=2, column=0, columnspan=3, padx=0, pady=0, )

search_btn.grid(row=3, column=0, padx=0, pady=0)
clean_search_btn.grid(row=3, column=2, padx=0, pady=0)
nlf_link_lable.grid(row=4, column=0, padx=0, pady=0)

rec_btn.grid(row=5, column=0, padx=0, pady=0)
clean_recsys_btn.grid(row=5, column=2, padx=0, pady=0)
recys_lbl.grid(row=6, column=1, padx=0, pady=0)

exit_btn.grid(row=7, column=1, padx=0, pady=0)
countdown_lbl.grid(row=8, column=1, padx=0, pady=0)

root.mainloop()