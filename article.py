import nltk
from newspaper import Article

#get the article
url = 'https://economictimes.indiatimes.com/industry/services/education/delays-in-conducting-examinations-can-impede-admission-chances/articleshow/77412054.cms'
article = Article(url)

#natural language processing
article.download()
article.parse()
nltk.download('punkt')
article.nlp()

#get the author
article.authors

#get the published date
article.publish_date

#get the top image
article.top_image

#get article text
print(article.text)

#get a summary of the article
print(article.summary)

