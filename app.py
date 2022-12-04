from flask import Flask, render_template, request, redirect, url_for

from bsbi import BSBIIndex
from compression import VBEPostings

app = Flask(__name__)
app.secret_key = "tp4InfoRetrieval"
app.config['TEMPLATES_AUTO_RELOAD'] = True

BSBI_instance = BSBIIndex(data_dir = 'collection', \
                          postings_encoding = VBEPostings, \
                          output_dir = 'index')

class my_dictionary(dict): 
  
    # __init__ function 
    def __init__(self): 
        self = dict() 
          
    # Function to add key:value 
    def add(self, key, value): 
        self[key] = value 

def print_content(x):
	result = []
	for i in range(len(x)):
		result.append(open(x[i], "r").read())

	return result

@app.route("/")
def index():
	return render_template("index.html")


@app.route("/search", methods=["POST", "GET"])
def searchr():
	global query
	global title
	global results

	if request.method == "POST":
		query = request.form["query"]
		retrieve = BSBI_instance.retrieve_tfidf(query, k = 10)
		title = [i[1] for i in retrieve]
		content = print_content(title)
		results = my_dictionary()

		for i in range(len(retrieve)):
			results.add(title[i], content[i])

		return redirect(url_for("searchr"))

	return render_template("search.html", title=title, results=results, query=query)


if __name__ == '__main__':
	app.run(debug=True)
	