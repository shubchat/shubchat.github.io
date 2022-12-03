+++
title = "Recommending great short stories"
description = "Recommending great short stories"
date = 2019-06-24T02:13:50Z
author = "Shubh Chatterjee"
+++


# Tutorial on development of an AI engine for recommending great short stories

## Introduction

In today’s hyper-digital age there are multiple products vying for your attention, whether it is for social networking (Facebook, Twitter, Reddit, etc) or Movie recommendations(Netflix, Prime, Hulu). What each of these products aims for, is to enhance your engagement for the service you are trying to receive using a product. More engagement means more time spent on a portal which means higher possibility of getting a subscription sign up, Selling targeted ads or other revenue opportunities. The key to higher engagement is hyper-personalized recommendation for users based on who you are, where you are, what your personality is and what are your likes or dislikes. A technology agent should provide the most optimum & personalized customer journey for a product and this is where things become really interesting.

## Recommendation engines

Recommendation engines are all around us, any good tech product that you use for a service most probably has a personalized recommendation component to it. Below are a few examples :

**Listening to Music**

![](https://cdn-images-1.medium.com/max/2000/0*cWIZDAES1lOW0-OF.png)

**Watching a TV Show**

![](https://cdn-images-1.medium.com/max/2032/0*VN5W2y4il24ySvNV.jpg)

**Social networking**

![](https://cdn-images-1.medium.com/max/2000/0*bIJum9mu41BmnkGu.jpg)

This and many more, just think of any tech service that you are probably using and it is somehow attempting to serve you a more personalized path of navigating through the product experience. They are able to do it by using the data they have about you and other users on its portal. The more users they might have the better there recommendation algorithm can be.

## Why do we need a recommendation engine for short stories?

Among the plethora of recommendation based products (where all are paid products), we have services for music, movies, social network but for live reading short prose there is no viable alternative. Amazon does have a good recommendation engine for what to read next but it for a book or anthology and not for short text, stories, prose, etc. **The idea of this project was to enhance the reading of short enriched pieces of text while recommending the best next short reading for a user based on their likes and preferences.**

## **A free reading portal for short stories called readnet**

We developed readnet a web portal where anyone can go and read some fantastic prose by some well-known author for free. The web app also recommends other short stories based on the current short story the user might be reading . The user journey is as simple as below three points:

1. User visits the web app

![](https://cdn-images-1.medium.com/max/2850/1*yNEhQNEJuwutwGAwRtsrSA.png)

2.User is served six short stories to choose for,Hence, starting their reading adventure

![](https://cdn-images-1.medium.com/max/2000/1*1_6Q92yp7xNYof6Mta7PaQ.gif)

3.Once the user has selected a story that might be to the users liking there is an option to get more stories recommended

![](https://cdn-images-1.medium.com/max/2000/1*083hcOrGkzknhk5UXimNaw.gif)

An extremely simple workflow from a user experience perspective but in the backend when it was being developed there were substantial smarts developed which we would now be looking into.

## How did we develop Readnet?

There are below steps which were involved in the development process:

1. Content collection

1. Development of the recommendation algorithm

1. Creation of source database for serving stories and recommendations

1. Backend API development for serving readers texts and recommendations

1. Front end UI development for a good user reading experience

1. Deployment of the web app on to Heroku platform

### Step-1: Content collection

All of the short stories on readnet are free from copyright as that is the only way we could make it available no charge for the userbase. One of the largest repository for free out of copyright books and stories is [https://www.gutenberg.org/](https://www.gutenberg.org/)(Project Gutenberg). **All the content that you would find on the site has been extracted from Project Gutenberg(Please consider a small [donation ](https://www.gutenberg.org/wiki/Gutenberg:Project_Gutenberg_Needs_Your_Donation)to this outstanding initiative).**

The task on hand was navigating the huge collection of over +64K ebooks/stories on Gutenberg and identifying short stories for use on readnet recommendation engine. I found a huge collection of short stories at ([http://www.gutenberg.org/ebooks/search/?query=Short+Stories](http://www.gutenberg.org/ebooks/search/?query=Short+Stories))

![](https://cdn-images-1.medium.com/max/2756/1*1n1-OvOzz7XOOH9p7Tefyw.png)

Now that we know from where we can get the short stories we needed to extract these into files for text processing and development of web app. This was easily done by using the amazing Beautiful soup library in python. First, we get the links of the short stories we want on our webapp

    /*Get the list of short stories*/
    while n<1000:
        url = f"[http://www.gutenberg.org/ebooks/search/?query=Short+Stories&start_index={n](http://www.gutenberg.org/ebooks/search/?query=Short+Stories&start_index={n)}"
        page = requests.get(url)    
        data = page.text
        soup = BeautifulSoup(data)
        for link in soup.find_all('a'):
            List.append(link.get('href'))

    /*Get the urls for the corresponding short stories*/

    from bs4 import BeautifulSoup, SoupStrainer
    import pandas as pd
    import requests
    url_download=[]
    url=f"[http://www.gutenberg.org/files/](http://www.gutenberg.org/files/)"
    df=pd.read_csv('ID_shortstories2.csv')
    Ids=list(df['ID_shortstories'])

    for Id in Ids:
        url2=url+str(Id)+'/'
        page = requests.get(url2)    
        data = page.text
        soup = BeautifulSoup(data)
        zips=[]
        for link in soup.find_all('a'):
            
            if link.get('href').endswith('.txt'):
                zips.append(link.get('href'))
        if len(zips)==0:
            print("The ID {} has no txt".format(Id))
        else:
            url3=url2+zips[0]
            #print(url3)
            url_download.append(url3)

Next, we extract all short stories into separate text documents.

    import os
    import urllib.request
    check=[]

    DOWNLOADS_DIR = 'books/'

    # For every line in the file
    for url in open('links.txt'):
        # Split on the rightmost / and take everything on the right side of that
       # name2=os.path.split(url)[-1]
        #name2=os.path.split(urllib.parse.urlparse(url).path)[-1]
        #print(name2)
        name = url.split('/')[-1]
        name2=name.split(sep='txt')[0]
        name3=name2+'txt'
        #print(name)

    # Combine the name and the downloads directory to get the local filename
        filename = os.path.join(DOWNLOADS_DIR, name3)
        print(filename)
        
        # Download the file if it does not exist
        if not os.path.isfile(filename):
            
            print(url)
            urllib.request.urlretrieve(f"{url}",f"{filename}")

Now we will get the Title, author and language of short stories we have extracted

    import os

    bookno=[]
    Title=[]
    Author=[]
    Language=[]
    for a in os.listdir(path='books'):
        if a.endswith('.txt'):

    with open(f"books/{a}",'r',errors='ignore') as f:

    Title1=[]
                Author1=[]
                Language1=[]
                for line in f:

    y=line.split()
                    #raise Exception("The files is {}".format(f))
                    #print(len(y))
                    #if y[1]=='Title:':
                    if (len(y)>0 and y[0]=='Title:'):
                         Title1=y.copy()
                    if (len(y)>0 and y[0]=='Author:'):
                        Author1=y.copy()
                    if (len(y)>0 and y[0]=='Language:'):
                        Language1=y.copy()

    bookno.append(a)
            Title.append(Title1)
            Author.append(Author1)
            Language.append(Language1)

Finally, we get everything into a pandas data frame for ease of visualization and analysis

    import pandas as pd
    db_books=pd.DataFrame()
    db_books['bookno']=bookno

    # db_books['Title']=[T for T in Title if T!='Title:' ]
    # db_books['Author']=[A for A in Author if A!='Author']
    # db_books['Language']=[L for L in Language if L!='Language']

    db_books['Title']=[' '.join(T) for T in Title]
    db_books['Author']=[' '.join(A) for A in Author]
    db_books['Language']=[' '.join(L) for L in Language]

Now, post this with some simple edits and checks and we have our source data ready.

### Step-2: Development of the recommendation algorithm

The recommendation algorithm that we would be using is an unsupervised model which based on how similar are two pieces of texts, assigns a corresponding similarity index. There are two sub-processes in here, first, we convert all the texts of corresponding short stories into [TFIDF](http://www.tfidf.com/) (Term frequency-inverse document frequency). We need to do that as algorithms/math can only interact on a mathematical or vectorized representation.

    from sklearn.feature_extraction.text import  TfidfVectorizer
    from sklearn.feature_extraction import text
    my_stop_words = text.ENGLISH_STOP_WORDS.union(['gutenberg','ebook','online','distributed','transcriber','etext','note','copyright',"start",'project','end','produced','proofreading','team','http','www','pgdp','net','illustrated'])

    vectorizer=TfidfVectorizer(stop_words=my_stop_words)
    vectorizer.fit(stories['content'])

    X_vector=vectorizer.transform(stories['content'])

Next, based on how similar([Cosine similarity](https://www.sciencedirect.com/topics/computer-science/cosine-similarity)) two pieces of texts are we would assign respective rankings. For example, which are the top5 short stories similar to story A and so on.

    from sklearn.metrics.pairwise import cosine_similarity
    similarity_matrix=cosine_similarity(X_vector)

**Step-3: Creation of source database for serving stories and recommendations**

Now we have short stories with their metadata(Author, language, Title) and also which stories are most similar to each other now we need to store all of them into a database which can be queried by our web app. I am using a Postgres database but depending upon the comfort of the developer any other RDBMS can be used.

    /*First the metadata*/

    
    from sqlalchemy import create_engine
    engine = create_engine('postgresql:///stories')

    db_books.to_sql("metadata", engine,if_exists='append',index=False)

    /*Then the stories*/
    stories.to_sql("short_stories", engine,if_exists='append',index=False)

In addition to this, we will need another table lets name it recos where for each text(short story) we would have top-five similar texts.

![](https://cdn-images-1.medium.com/max/2000/1*qd2E3yQxP_m3GFlaocNcrQ.png)

### Step-4: Backend API development for serving readers texts and recommendations

Now we have all the meat for making our product. We have:

1. The text for short stories

1. We have the metadata for stories

1. We have top 5 recommendations for each of the short stories

We now need to develop an API which based on the front end user interaction pushes the texts and data that is needed from a product perspective. We will be using flask for it which is an amazingly good microframework for quickly development and serving of the backend of a product. Below is a complete code for how you can develop that

    import os

    from flask import Flask, session,render_template,request,session,redirect,url_for,escape,flash

    from flask_session import Session

    from werkzeug import check_password_hash, generate_password_hash

    from itsdangerous import URLSafeTimedSerializer

    from sqlalchemy import create_engine

    from sqlalchemy.orm import scoped_session, sessionmaker

    import requests

    import os

    import pandas as pd

    import time

    # Set up database

    engine = create_engine(os.getenv("DATABASE_URL"))

    db = scoped_session(sessionmaker(bind=engine))

    app = Flask(__name__)

    # Set the secret key to some random bytes. Keep this really secret!

    app.secret_key ='#######'

    Security_password_salt='########'

    @app.route("/",methods=["GET","POST"])

    def home():

    stories=db.execute("select metadata.bookno,metadata.title,metadata.author,short_stories.content from metadata LEFT JOIN short_stories on metadata.bookno=short_stories.bookno order by random() LIMIT 6").fetchall()

    return render_template("index.html",story=stories)

    @app.route("/recommendations",methods=["GET","POST"])

    def reco():

    bookno=request.form.get("bookno")

    # return bookno

    recos=db.execute("select * from recos where bookno=:bookno",{'bookno':bookno}).fetchone()

    reco1=db.execute("select metadata.bookno,metadata.title,metadata.author,short_stories.content from metadata LEFT JOIN short_stories on metadata.bookno=short_stories.bookno  where metadata.bookno=:bookno",{'bookno':recos.first_reco}).fetchone()

    reco2=db.execute("select metadata.bookno,metadata.title,metadata.author,short_stories.content from metadata LEFT JOIN short_stories on metadata.bookno=short_stories.bookno  where metadata.bookno=:bookno",{'bookno':recos.second_reco}).fetchone()

    reco3=db.execute("select metadata.bookno,metadata.title,metadata.author,short_stories.content from metadata LEFT JOIN short_stories on metadata.bookno=short_stories.bookno  where metadata.bookno=:bookno",{'bookno':recos.third_reco}).fetchone()

    reco4=db.execute("select metadata.bookno,metadata.title,metadata.author,short_stories.content from metadata LEFT JOIN short_stories on metadata.bookno=short_stories.bookno  where metadata.bookno=:bookno",{'bookno':recos.fourth_reco}).fetchone()

    reco5=db.execute("select metadata.bookno,metadata.title,metadata.author,short_stories.content from metadata LEFT JOIN short_stories on metadata.bookno=short_stories.bookno  where metadata.bookno=:bookno",{'bookno':recos.fifth_reco}).fetchone()

    return render_template("recommend.html",reco1=reco1,reco2=reco2,reco3=reco3,reco4=reco4,reco5=reco5)

    # reco2=db.execute("select * from metadata where bookno=:bookno",{'bookno':recos.second_reco}).fetchone()

    # reco3=db.execute("select * from metadata where bookno=:bookno",{'bookno':recos.third_reco}).fetchone()

    # reco4=db.execute("select * from metadata where bookno=:bookno",{'bookno':recos.fourth_reco}).fetchone()

    # reco5=db.execute("select * from metadata where bookno=:bookno",{'bookno':recos.fifth_reco}).fetchone()

    # return render_template("recommend.html",reco1=reco1,reco2=reco2,reco3=reco3,reco4=reco4,reco5=reco5)

### Step-5: Front end UI development for a good user reading experience

If you are like me who is mainly a data scientist and not a front end or product guy you are more likely not familiar with the details of how the front end designs work and what are the best practices around it. In today’s era of rapid specialization, its always better to have a T-Shaped skill set and some basic knowledge of HTML, CSS, and Javascript is really useful. There are some fantastic frameworks out there that make our life extremely easy for example Bootstrap([https://getbootstrap.com/](https://getbootstrap.com/)). As we are just developing a prototype it will be much more useful for us to use Bootstrap to hit the market fast rather than spending a huge amount of time on front end development. We will be using a very popular template called the freelancer template. Just clone the template from git on to your local and make the changes that are needed([https://github.com/BlackrockDigital/startbootstrap-freelancer](https://github.com/BlackrockDigital/startbootstrap-freelancer)) for your version. The main changes that are needed are in the templates. I had two HTML files one for landing page and another for recommendations.

### Step-6: Deployment of the web app on to Heroku platform

Now you would have the product ready on your personal machine with a database, a flask code-based API and a front end developed using Bootstrap4. Now, you would want to deploy this in front of the world so that anyone with a computing device with access to the internet should be able to connect to it and start reading. There are multiple platforms available out there for it like Heroku, Pythonanywhere, Digital Ocean, etc. We have used Heroku but feel free to use any of the platforms you might be comfortable with to do so. It is literally point and click as long you have a requirement.txt file created for your python environment for the project. Below are simple steps using the UI of Heroku:

1. Login to Heroku

1. Create a new project

1. Use Github as a deployment method

1. Migrate the database from your local to Heroku

1. Deploy the project

### **Final Thoughts**

The above five steps, in a nutshell, summarise how to quickly develop a recommendation engine from ground zero to something which is very much usable. Other enhancements that could be tried on these are that you could develop a CI/CD pipeline using Travis CI or something similar to quickly enhance the product and test things out.

Thanks for taking time out for reading this! Wishing you all the best.
