+++
title = "AI-Quiz Master"
categories= ["ML", "DL"]
tags= ["ML","Deep Learning"]
description = "Develop an AI app which could be a quiz master for our training needs"
date = 2014-09-28T02:13:50Z
author = "Shubh Chatterjee"
+++


# Develop an AI app which could be a quiz master for our training needs

One of the most overlooked domains in today’s knowledge economy is the need for rapid and frequent re-skilling of our workforce. A knowledge worker without frequent ability and resources to improve their skill set will move towards redundancy, as we are now moving towards an age where we all will be expected to work in synergy with AI-powered systems for whom learning is just a patch update or system upgrade. We the humans, on the other hand, are always in need of learning resources which are able to transform our knowledge base easily and effectively.

One of the best ways humans learn is by **“deliberate practice”. **Psychologist [K. Anders Ericsson](https://en.wikipedia.org/wiki/K._Anders_Ericsson), a professor of Psychology at [Florida State University](https://en.wikipedia.org/wiki/Florida_State_University), has been a pioneer in researching deliberate practice and what it means. According to Ericsson:
> *People believe that because expert performance is qualitatively different from a normal performance the expert performer must be endowed with characteristics qualitatively different from those of normal adults. We agree that expert performance is qualitatively different from normal performance and even that expert performers have characteristics and abilities that are qualitatively different from or at least outside the range of those of normal adults. However, we deny that these differences are immutable, that is, due to innate talent. Only a few exceptions, most notably height, are genetically prescribed. Instead, we argue that the differences between expert performers and normal adults reflect a life-long period of deliberate effort to improve performance in a specific domain.*

**One of the best ways to do deliberate practice for a knowledge worker is to learn by testing himself or herself**. If huge quantities of knowledge in text/audio and video can be converted into tailored quizzes or trivia, which will evolve as per the learning ability and style of the learner, such a tool could be a holy grail to do deliberate practice for a modern-day knowledge worker. But, the challenge is who will do this, **we live in a world where there is a huge shortage of good teachers who can transform the way knowledge is being consumed by us. This should make us look towards technology for a solution to this problem**.

**I am proposing an AI-powered application which abstracts away from the need of a content developer to design knowledge quiz.** This application is able to construct knowledge questions with multiple answers from a source text across domains. Let us look at one such application, this is for development on a news quiz app to test subjects for their knowledge of current happenings.

## News Quiz app

To develop a news quiz app the source we picked is [https://newsapi.org/](https://newsapi.org/).Through this API we get breaking news headlines and search for articles from over 30,000 news sources and blogs. The content we source from this API is in JSON format. Below is a code extract code that is extracting **“Australian Sports”** news from **“The Australian” **(this does not seem to be working as of now)for the last 2 days. The news is to be published in English and is sorted by the published date.

    from newsapi import *

    newsapi = NewsApiClient(api_key='********************')
    from datetime import datetime, timedelta
    today=datetime.today().strftime('%Y-%m-%d')
    ldate=(datetime.today()-timedelta(days=2)).strftime('%Y-%m-%d')

    all_articles = newsapi.get_everything(q='Australia Sports',
                                          sources='The Australian',

                                         from_param=ldate,
                                          to=today,
                                          language='en',
                                          sort_by='publishedAt',
                                          page=1,
                                          page_size=100

    )

Below is how the extract looks like :

    'author': 'Nicholas Mendola',
       'content': 'AL AIN, United Arab Emirates (AP) China fought back from a goal down to beat Thailand 2-1 in the round of 16 of the Asian Cup on Sunday and avoid a potentially embarrassing upset.\r\nSupachai Jaided put Thailand ahead in the 31st minute, turning and shooting de… [+2755 chars]',
       'description': "While some were still sleeping, Manchester City worked as efficient a goal as you'll see this weekend.",
       'publishedAt': '2019-01-20T19:11:13Z',
       'source': {'id': None, 'name': 'Nbcsports.com'},
       'title': 'Man City carves up Huddersfield with clinical goal (video)',
       'url': 'https://soccer.nbcsports.com/2019/01/20/man-city-carves-up-huddersfield-with-clinical-goal-video/',
       'urlToImage': 'https://nbcprosoccertalk.files.wordpress.com/2019/01/ap_19020544004451-e1548002632357.jpg?w=1200'},
      {'author': None,
       'content': "Kiwi convert Brad Shields is the latest injury concern for Eddie Jones ahead of England's Six Nations rugby kickoff in Ireland on February 2.\r\n Loose forward Shields, who has become a key part of the England makeup since defecting from New Zealand last year, … [+2155 chars]",
       'description': "Kiwi convert Brad Shields is the latest injury concern for Eddie Jones ahead of England's Six Nations rugby kickoff in Ireland on February 2.",
       'publishedAt': '2019-01-20T19:04:01Z',
       'source': {'id': None, 'name': 'Stuff.co.nz'},
       'title': "Brad Shields joins Owen Farrell on Eddie Jones' England injury worry-list",
       'url': 'https://www.stuff.co.nz/sport/rugby/international/110049826/brad-shields-joins-owen-farrell-on-eddie-jones-england-injury-worrylist',
       'urlToImage': 'https://resources.stuff.co.nz/content/dam/images/1/s/k/5/j/j/image.related.StuffLandscapeSixteenByNine.1420x800.1tiqzm.png/1548011041867.jpg'},
      {'author': 'Isaac Chotiner',
       'content': 'In her new book, Merchants of Truth: The Business of News and the Fight for Facts, Jill Abramson, the former executive editor of the Times, examines how four large American news organizations are surviving the age of the Internet and Donald Trump. Abramsons a… [+23401 chars]',
       'description': 'In her new book, the former executive editor of the New York Times says that she was “determined to capture this moment of wrenching transition” in journalism.',
       'publishedAt': '2019-01-20T18:55:09Z',
       'source': {'id': None, 'name': 'Newyorker.com'},
       'title': 'How Journalism Survives: An Interview with Jill Abramson',
       'url': 'https://www.newyorker.com/news/the-new-yorker-interview/how-journalism-survives-an-interview-with-jill-abramson',
       'urlToImage': 'https://media.newyorker.com/photos/5c437b090e0a446c5c034086/16:9/w_1200,h_630,c_limit/Chotiner-JillAbramson.jpg'},
      {'author': 'Sandeep Dwivedi',
       'content': 'Ishant Sharma with his wife Pratima.\r\nIshant Sharma takes a long pause before he utters a barely audible I think. Actually, he hasn’t yet thought it through. He is merely buying time. He pauses again, thinks more. I think … that was the first time … hmm … I c… [+13393 chars]',
       'description': 'Ishant Sharma is a much better bowler than he was six years ago, and as Sandeep Dwivedi finds out, he has his close-knit support system to thank for emerging from the biggest nightmare of his career.',
       'publishedAt': '2019-01-20T18:39:19Z',
       'source': {'id': None, 'name': 'Indianexpress.com'},
       'title': 'iShant 2.0: Rebooted, upgraded',
       'url': 'https://indianexpress.com/article/sports/cricket/ishant-sharma-indian-cricket-team-5547521/',
       'urlToImage': 'https://images.indianexpress.com/2019/01/ishant-sharma.jpg?w=759'},
      {'author': 'Karen Crouse - The New York Times',
       'content': 'MELBOURNE, AUSTRALIA—A veteran right-hander with an all-court game and a one-handed backhand, the hands-down greatest player of his era, was trying to dodge an upset in a fourth-round match of a grand slam tournament against a former junior world No. 1 with l… [+5362 chars]',
       'description': 'Stefanos Tsitsipas, who broke through at Rogers Cup in Toronto last summer, reaches first grand slam quarterfinal with four-set win over legend Roger Federer in Melbourne.',
       'publishedAt': '2019-01-20T18:33:31Z',
       'source': {'id': None, 'name': 'Thestar.com'},
       'title': 'Defending champion Federer upset by Tsitsipas at Australian Open',
       'url': 'https://www.thestar.com/sports/tennis/2019/01/20/defending-champion-federer-upset-by-tsitsipas-at-australian-open.html',
       'urlToImage': 'https://images.thestar.com/WozmFzs6uwbzrzDDxavD4nsup2s=/1200x848/smart/filters:cb(1548010096665)/https://www.thestar.com/content/dam/thestar/sports/tennis/2019/01/20/defending-champion-federer-upset-by-tsitsipas-at-australian-open/federer_tsitsipis.jpg'},
      {'author': 'Nicholas Mendola',
       'content': 'AL AIN, United Arab Emirates (AP) China fought back from a goal down to beat Thailand 2-1 in the round of 16 of the Asian Cup on Sunday and avoid a potentially embarrassing upset.\r\nSupachai Jaided put Thailand ahead in the 31st minute, turning and shooting de… [+2755 chars]',
       'description': 'Dele Alli scored and was injured in the win, which started with Fernando Llorente scoring an own goal.',
       'publishedAt': '2019-01-20T17:54:42Z',
       'source': {'id': None, 'name': 'Nbcsports.com'},
       'title': 'The “other” Harry! Winks scores late for Spurs (video)',
       'url': 'https://soccer.nbcsports.com/2019/01/20/the-other-harry-winks-scores-late-for-spurs-video/',
       'urlToImage': 'https://nbcprosoccertalk.files.wordpress.com/2019/01/ap_19020580752409-e1548006824106.jpg?w=1200'},
      {'author': 'PTI',
       'content': 'Pakistan had finished 11th without winning a single match at the 16-nation World Cup held in Bhubaneswar during November-December last year. (Representational Image)\r\nPakistan selectors Sunday effected a wholesale change in the country’s hockey squad for the … [+2975 chars]',
       'description': 'After trials held in Islamabad, chief selector Islahuddin Siddiqui announced Sunday that captain Muhammad Rizwan (Sr) and 10 other players, mostly seniors, were dropped from the team.',
       'publishedAt': '2019-01-20T17:12:29Z',
       'source': {'id': None, 'name': 'Indianexpress.com'},
       'title': 'Pakistan name team for Pro-League Hockey, drop 11 players from World Cup side',
       'url': 'https://indianexpress.com/article/sports/hockey/pakistan-name-team-for-pro-league-hockey-drop-11-players-from-world-cup-side-5547499/',
       'urlToImage': 'https://images.indianexpress.com/2018/08/hockey-m.jpg?w=759'}

Now it’s quite difficult to make sense of the all the text above in this format what we can do is to break it up into different information streams. Below code uses the information stream we got from News API and splits it into Title, Description, source, and date.

    Title=[]
    Description=[]
    Source=[]
    Date=[]

    #no_pages=np.ceil(all_articles['totalResults']/len(all_articles['articles']))
    no_pages=(all_articles['totalResults']/len(all_articles['articles']))+ (all_articles['totalResults'] % len(all_articles['articles']) > 0)

    i=1
    while i<=no_pages:

    all_articles = newsapi.get_everything(q='Australia Sports',
                                              sources='The Australian',

    from_param=ldate,
                                          to=today,
                                          language='en',
                                          sort_by='publishedAt',
                                          page=i,
                                          page_size=100

    )
        j=0
        while j<=len(all_articles['articles'])-1:
            Title.append(all_articles['articles'][j]['title'])
            Description.append(all_articles['articles'][j]['description'])
            Source.append(all_articles['articles'][j]['source']['name'])
            Date.append(all_articles['articles'][j]['publishedAt'])

    j+=1
        i+=1

Let's look at the information that we have been able to split and how it looks like now:

First 4 Titles in the list:

    ['England beat Australia but lose Quad Series',
     'Reebok Workout Plus White & Gum (Size 7~14) $20 (Was $130) + Postage @ JD Sports',
     'Man City carves up Huddersfield with clinical goal (video)',
     "Brad Shields joins Owen Farrell on Eddie Jones' England injury worry-list"]

The corresponding description of these titles in the description list:

    ['England beat Australia 52-49 but fall just short of the winning margin needed to take the Quad Series title in London.',
     'Deal: Reebok Workout Plus White & Gum (Size 7~14) $20 (Was $130) + Postage @ JD Sports, Store: JD Sports Australia, Category: Fashion & Apparel',
     "While some were still sleeping, Manchester City worked as efficient a goal as you'll see this weekend.",
     "Kiwi convert Brad Shields is the latest injury concern for Eddie Jones ahead of England's Six Nations rugby kickoff in Ireland on February 2."]

What are the sources of these news snippets:

    ['BBC News', 'Ozbargain.com.au', 'Nbcsports.com', 'Stuff.co.nz']

We have been able to change the raw API data into meaningful information in different lists. The next step is the generation of questions from these snippets .for example, for the news snippet “‘England beat Australia but lose Quad Series’’’, we would like a question like “Who did England beat but lost Quad series?”,and then we will have multiple options for the same.

Another thought was instead of questions we have fill in the blank like — — — — — — -beat Australia but lose Quad Series. And we will have multiple options for this. On further brain-storming, we decided to go ahead with fill in the blank as it will be easier to construct different pieces around it.

## Fill in the blank for news content

As this is a proof of concepts we will keep things simple, our app will shuffle through the title of news identify all the entities(organization, person, geopolitical entity) and will replace one of the words with a blank(____). We will have 4 options for each query. One will be the word which has been replaced by a blank and others will be words from other news titles. Let's check out how we do that.

We will use spacy for this. Spacy is one of the most popular and robust natural language processing toolbox that exists. We will use one of the pre-trained models([en_core_web_lg](https://spacy.io/models/en#en_core_web_lg)) in the toolbox. It is an English multi-task CNN trained on OntoNotes, with GloVe vectors trained on Common Crawl. Let’s look at an example, Below is a title from a news

    'Aus Open Day 8 Roundup reaction Down Under as Djokovic Williams Raonic through to QF PHOTOS '

Let's use spacy on this one to identify what are the entities in this statement.

    import spacy
    nlp = spacy.load('en_core_web_lg')
    doc=nlp(' '.join(Title[1:2]))
    #doc=nlp(' '.join(u))

    for ent in doc.ents:
        #print(ent)
        print(str(ent.text),str(ent.label_))

What do we get:

    Day 8 DATE
    Roundup & ORG
    Djokovic PERSON
    Williams & Raonic ORG

We see the model is able to identify Djokovic as a person(It does stuff it up by tagging Williams and Raonic as an organization) but we never said it is perfect.

## Generate question, answers and options pair for each title in the corpus extracted from news API

Below code generates three lists one with fill in the blanks one with the corresponding answer and other with the category(‘ORG’,’PERSON’,’GPE’) that the answer belongs to.

    import random
    import spacy
    nlp = spacy.load('en_core_web_lg')
    Query=[]
    #check=[]
    Ans=[]
    Options=[]
    category=[]
    i=1
    j=0

    **/*For all news articles we extract from news api*/**

    while i <=all_articles['totalResults']:
        str1 = nlp(' '.join(Title[j:i])) /*We use spacy */
    **/*creating and empty dictionary which will have all the extracted entities with the corresponding labels(GPE,person,organization etc)*/ **   

    dict={}
            
        for ent in str1.ents:
            
            #print(i,j)
            #print(str(ent.text),str(ent.label_))
            
            try:
                dict[str(ent.text)]=str(ent.label_)
            except:
                pass
    **/*Here I am creating a list of answers from the dictionary provided they are either of 'ORG','PERSON','GPE' */**

    **/*Also there are lists which store the actual statement and the category('ORG','PERSON','GPE') that the answer belongs to */**

        check=[]    
        for key,value in dict.iteritems():
            #print(key)
            #print(value)

    if value in ('ORG','PERSON','GPE'):

    check.append(key)
            #print len(check)
                #print(check)
        if len(check)>1:
            #print('True')
            #print(check)
            Ans.append(' '.join(random.sample(check,1)))
            #Query.append(str(' '.join(Title[j:i])))
            Query.append(str1)
            category.append(dict[''.join(Ans[-1])])
        elif len(check)==1:
            Ans.append(' '.join(check))
            #print(Title[j:i])
            #Query.append(str(' '.join(Title[j:i])))
            Query.append(str1)
            category.append(dict[''.join(Ans[-1])])
        else:
            Query=Query
            category=category

    #         check=(stripNonAlphaNum(' '.join(Title[j:i])))
    #         statement=str(" ".join(check))
    #         ans=

    i+=1
        j+=1

Some more housekeeping regarding the questions in the texts and replacing special values which we may not need.

    Quiz=[]
    i=0
    char=['[',']','"',"'","'"]

    while i<=len(Query)-1:
        text=str(Ans[i])
        for c in char:
            text=text.replace(c,"")
            Ans[i]=text
        #print(text)

    #Quiz.append(str(Query[i]).replace(str(text),'------------'))
        
        check=str(Query[i]).replace(str(text),'------------')
        
        if '------------' not in check:
            check=str(Query[i]).replace(str(text)+"'s",'------------')
        Quiz.append(check)
                                      
                                      
        #Quiz.append(str(Query[i]).replace(str(text)+"'s",'------------'))
        i+=1

The first 5 quiz questions:

    Quiz[:5]

    output:
    ["Foot Locker Introduces 'Power Store' Model in North America with ------------ in Metro Detroit",
     '------------ obituary',
     'Video shows drunk passenger attempting to hijack Uber in ------------',
     '------------: Technology, Fashion, Philly Fans',
     'BiP: Turkey’s Most Downloaded ------------']

Corresponding answers:

    Ans[:5]

    output:
    ['New Store', 'William Field', 'California', 'Jonah Bolden Q&A', 'Local App']

As now we have been able to get a question and answer list the next step is to get a list with the options. Now to get a list of the options we have to have similar other not-correct entities for a question. To get that, as all our answers are either of (‘ORG’,’PERSON’,’GPE’) that is an organization, person or geopolitical entity. We will create a pool of answers from each of this domain which can be used to create an option list

    i=1
    j=0
    PERSON=[]
    ORG=[]
    LOCATION=[]

    while i <=all_articles['totalResults']:
        str1 = nlp(' '.join(Title[j:i]))
        for ent in str1.ents:
            
            if str(ent.label_)=='PERSON':
                
                try:
                    
                    PERSON.append(str(ent.text))
                except:
                       pass
                
            
            if str(ent.label_)=='ORG':
                
                try:
                    ORG.append(str(ent.text))
                except:
                       pass
            
            if str(ent.label_)=='GPE':
                
                try:
                    LOCATION.append(str(ent.text))
                except:
                       pass

    #print(json.dumps(comprehend.detect_entities(Text=str1,LanguageCode='en'), sort_keys=True, indent=4))
        #print(i)
        #print(str1)
        i+=1
        j+=1

    #Remove all the duplicates and symbols

    i=0
    char=['[',']','"',"'","'"]

    while i<=(len(PERSON)-1):
        text=str(PERSON[i])
        for c in char:
            text=text.replace(c,"")
            #print(i)
            PERSON[i]=text
        i+=1
            
    i=0      
    while i<=(len(ORG)-1):
        text=str(ORG[i])
        for c in char:
            text=text.replace(c,"")
            #print(i)
            ORG[i]=text
        i+=1
            
    i=0     
    while i<=(len(LOCATION)-1):
        text=str(LOCATION[i])
        for c in char:
            text=text.replace(c,"")
            LOCATION[i]=text
        i+=1

    PERSON_final=[]

    for i in PERSON:
        
        if i not in PERSON_final:
            PERSON_final.append(i)
            
            
    ORG_final=[]

    for i in ORG:
        
        if i not in ORG_final:
            ORG_final.append(i)

    LOCATION_final=[]

    for i in LOCATION:
        
        if i not in LOCATION_final:
            LOCATION_final.append(i)

So now we have three lists which have a pool of all the appearing entities from a particular domain we are concerned with.

    PERSON_final[:5]

    output:
    ['William Field',
     'Jonah Bolden Q&A',
     'Naomi Osaka',
     'Ranji Trophy',
     'Cheteshwar Pujara']

    ORG_final[:5]

    output:
    ['Foot Locker Introduces ',
     'Power Store',
     'New Store',
     'Local App',
     'Luxury Hotels Opening']

    LOCATION_final[:5]

    output:
    ['Metro Detroit', 'California', 'Turkey', 'England', 'Karnataka']

Creating a list of options for each question now . What we need is for each question pick 3 other options(one option will be the answer itself) which is not the answer and is from the specific domain of that answer (‘ORG’,’PERSON’,’GPE’).

    #Lets create options for all the questions

    import random
    num_to_select =3 
    options=[]
    i=0
    j=1

    while i<len(category):

    if ''.join(category[i:j])=='ORG':
            #print(i)
            Final_ORG2=ORG_final[:]
            #while ''.join(Ans[i:j]) in Final_ORG2:

    Final_ORG2.remove(''.join(Ans[i:j]))

    randsamp=[]
            randsamp=random.sample(Final_ORG2,num_to_select)

    randsamp.append(''.join(Ans[i:j]))
            options.append(randsamp)

    elif ''.join(category[i:j])=='PERSON':
            Final_PERSON2=PERSON_final[:]
            #while ''.join(Ans[i:j]) in Final_PERSON2:

    Final_PERSON2.remove(''.join(Ans[i:j]))

    randsamp=[]
            randsamp=random.sample(Final_PERSON2,num_to_select)

    randsamp.append(''.join(Ans[i:j]))
            options.append(randsamp)

    elif ''.join(category[i:j])=='GPE':
            Final_LOCATION2=LOCATION_final[:]
            #while ''.join(Ans[0:1]) in Final_LOCATION2:

    Final_LOCATION2.remove(''.join(Ans[i:j]))

    randsamp=[]
            randsamp=random.sample(Final_LOCATION2,num_to_select)
            randsamp.append(''.join(Ans[i:j]))
            options.append(randsamp)

    i+=1
        j+=1

Now we have everything ready and processed to push into production. Let us look at what all the data crunching and natural language manipulation we have done for us.

    Quiz[:10]

    output:
    [**"Foot Locker Introduces 'Power Store' Model in North America with ------------ in Metro Detroit"**,
     '------------ obituary',
     'Video shows drunk passenger attempting to hijack Uber in ------------',
     '------------: Technology, Fashion, Philly Fans',
     'BiP: Turkey’s Most Downloaded ------------',
     'West Indies vs ------------ live stream: how to watch Test cricket from anywhere',
     '19 ------------ in 2019',
     "------------: Noodle company apologises for 'white-washing'",
     '------------ 2018 Semifinal Live Streaming: When and where to watch ------------ semifinal Live in IST?',
     'Ranji Trophy 2018-19: All eyes on Cheteshwar Pujara and Mayank Agarwal in Saurashtra vs ------------ semi-final']

    category[:10]

    output:

    [**'ORG',**
     'PERSON',
     'GPE',
     'PERSON',
     'ORG',
     'GPE',
     'ORG',
     'PERSON',
     'PERSON',
     'GPE']

    options[:10]

    output:
    [**['Ranji Trophy 2018 Semifinal Live Streaming',
      'Macquarie Sports Radio',
      'Kiwis Win the Toss & Opt',
      'New Store']**,
     ['Russel Arnold', 'Hakeem al-Araibi', 'Alberto Zaccheroni', 'William Field'],
     ['Japan', 'Qatar', 'Metro Detroit', 'California'],
     ['ODI', 'Adda', 'Brooke Blurton', 'Jonah Bolden Q&A'],
     ['Kiwis Win the Toss & Opt',
      'The Value Of V.F. Corp',
      'Covington Catholic',
      'Local App'],
     ['YouTube', 'Yamaguchi Prefecture', 'Ind', 'England'],
     ['EFL', 'Activate', 'Virgin', 'Luxury Hotels Opening'],
     ['Alberto Zaccheroni', 'AJ Pritchard', 'Rafael NadAI', 'Naomi Osaka'],
     ['Mens US10', 'Russell Arnold', 'Chris Harrison', 'Ranji Trophy'],
     ['ICC', 'Iraq', 'India', 'Karnataka']]

    Ans[:10]

    output:

    [**'New Store'**,
     'William Field',
     'California',
     'Jonah Bolden Q&A',
     'Local App',
     'England',
     'Luxury Hotels Opening',
     'Naomi Osaka',
     'Ranji Trophy',
     'Karnataka']

Each list is built serially where first fill in the blank in Quiz corresponds to its answer in Ans and so on with options and category. I have highlighted one of these in the above output space for a better understanding.

**We have been able to create a knowledge quiz from random excerpts of news from an API. This is a structured method which can be used on text across domains and the subsequent quiz which is generated can be a strong tool for testing and deliberate practice to get real knowledge developed in the learners.**

