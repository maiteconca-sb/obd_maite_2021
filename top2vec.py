

from top2vec import Top2Vec
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime


customers_reviews = pd.read_csv("/Users/maite.conca/PycharmProjects/OBD/files_reviews/customer_contact_2021_v6.csv", parse_dates=['received'])
customers_reviews.head(5)


customers_reviews['date'] = pd.to_datetime(customers_reviews['received'], errors='coerce', format='%Y-%m-%d')
customers_reviews['week'] = customers_reviews['date'].dt.week
customers_reviews['year'] = customers_reviews['date'].dt.year

## Count of topics per time
reviews_by_day = customers_reviews.groupby(['date'])['externalHandle'].count().reset_index(name = 'count_reviews')

x = np.array(reviews_by_day['date'])
y = np.array(reviews_by_day['count_reviews'])

plt.close()
plt.plot(x,y, color='#EC8A00')
plt.title("Number of Reviews per day", fontweight='bold', fontsize = 14, family = 'monospace')
plt.xlabel('Time')
plt.ylabel('Count of Reviews')
plt.savefig('images/no_reviews_per_day.png')
plt.show()

reviews_by_week = customers_reviews.groupby(['year', 'week'])['externalHandle'].count().reset_index(name = 'count_reviews')
reviews_by_week['year_week'] = list(range(len(reviews_by_week)))
x = np.array(reviews_by_week['year_week'])
y = np.array(reviews_by_week['count_reviews'])

plt.figure(figsize=(12,6))
plt.close()
plt.plot(x,y, color='#EC8A00')
plt.title("Number of Reviews per Week", fontweight='bold', fontsize = 14, family = 'monospace')
plt.xlabel('Time')
plt.ylabel('Count of Reviews')
plt.show()
plt.savefig('images/no_reviews_per_week_v2.png')

## Histogram of topics per week/or day

plt.figure(figsize=(12,6))
x1 = reviews_by_day['count_reviews']
plt.hist(x1, color='#7F0442', label='Day', bins = 30)
plt.title("Distribution Number of Reviews per Day", fontweight='bold', fontsize = 14, family = 'monospace')
plt.xlabel("Number of Reviews")
plt.legend()
plt.savefig('images/distribution_per_day.png')
plt.show()

plt.figure(figsize=(12,6))
x1 = reviews_by_week['count_reviews']
plt.hist(x1, color='#7F0442', label='Day', bins = 30)
plt.title("Distribution Number of Reviews per Week", fontweight='bold', fontsize = 14, family = 'monospace')
plt.xlabel("Number of Reviews")
plt.legend()
plt.savefig('images/distribution_per_week.png')
plt.show()

## Which day is more popular to leave a review

reviews_by_day['weekday'] = reviews_by_day['date'].dt.weekday
reviews_day_of_the_week = reviews_by_day.groupby(['weekday'])['count_reviews'].sum().reset_index(name = 'sum_reviews')

dict_day_week =  {
      0 : "Monday"
    , 1:"Tuesday"
    , 2: "Wednesday"
    , 3: "Thursday"
    , 4: "Friday"
    , 5: "Saturday"
    , 6: "Sunday"
 }
reviews_day_of_the_week['weekday_name'] = reviews_day_of_the_week['weekday'].map(dict_day_week)
x = np.array(reviews_day_of_the_week['weekday_name'])
y = np.array(reviews_day_of_the_week['sum_reviews'])

plt.close()
plt.figure(figsize=(12,6))
plt.bar(x,y, color='#EC8A00')
plt.xlabel('Day of the Week')
plt.ylabel('Count of Reviews')
plt.title("Number of Reviews per Day of the Week", fontweight='bold', fontsize = 14, family = 'monospace')
plt.savefig('images/no_reviews_per_day_of_the_week.png')
plt.show()

## How many customers are doing reviews
customers_reviews['externalHandle'].nunique()
## Are the same customers the same time?
customers_reviews.groupby(['externalHandle'])['channel'].count().reset_index(name = 'count').sort_values(by= 'count', ascending=False).head(20)

count_by_customer = customers_reviews.groupby(['externalHandle'])['channel'].count().reset_index(name = 'count')
plt.figure(figsize=(8,4))
x1 = count_by_customer['count']
plt.hist(x1, color='#7F0442', label='Customers', bins = 30)
plt.title("Distribution Number of Reviews per Customer", fontweight='bold', fontsize = 14, family = 'monospace')
plt.xlabel("Number of Reviews")
plt.legend()
plt.savefig('images/distribution_per_customer.png')
plt.show()

customers_reviews['comment'][0]

## Train the model
customers_reviews_list = list(customers_reviews["comment"])

model = Top2Vec(customers_reviews_list,speed="deep-learn", workers=8 )

"""
documents: Input corpus, should be a list of strings.
speed: This parameter will determine how fast the model takes to train. The 'fast-learn' option is the fastest and will generate the lowest quality vectors. The 'learn' option will learn better quality vectors but take a longer time to train. The 'deep-learn' option will learn the best quality vectors but will take significant time to train.
workers: The amount of worker threads to be used in training the model. Larger amount will lead to faster training.
"""

## Get the number of topics
model.get_num_topics()

## Get the topics
topic_words, word_scores, topic_nums = model.get_topics(117)

## View the topics
topic_words[1]

documents, document_scores, document_ids = model.search_documents_by_topic(topic_num=32, num_docs=10)
for doc, score, doc_id in zip(documents, document_scores, document_ids):
    print(f"Document: {doc_id}, Score: {score}")
    print("-----------")
    print(doc)
    print("-----------")
    print()



for topic in topic_nums[86:90]:
    model.generate_topic_wordcloud(topic, background_color="white")
    plt.show()
    plt.savefig('images/topic_worldcloud'+str(topic) )

### Use Top2Vec for Semantic Search
topic_words, word_scores, topic_scores, topic_nums = model.search_topics(keywords=["offer", "great"], num_topics=5)
for topic in topic_nums:
    model.generate_topic_wordcloud(topic, background_color="white")
    plt.show()
    plt.savefig('images/topic_worldcloud_semantic_search' + str(topic))


topic_sizes, topic_nums = model.get_topic_sizes()

topic_words, word_scores, topic_nums = model.get_topics(122)

customer_review_list_10 =  customer_review_list*10


topic_words, word_scores, topic_nums = model.get_topics(17)


topic_words, word_scores, topic_scores, topic_nums = model.search_topics(keywords=["offer"], num_topics=5)

words, word_scores = model.similar_words(keywords=["offers"], keywords_neg=[], num_words=20)
for word, score in zip(words, word_scores):
    print(f"{word} {score}")

topic_words, word_scores, topic_scores, topic_nums = model.search_topics(keywords=["offer"], num_topics=5)
for topic in topic_nums:
    model.generate_topic_wordcloud(topic)
    plt.show()

documents, document_scores, document_ids = model.search_documents_by_topic(topic_num=1, num_docs=5)
for doc, score, doc_id in zip(documents, document_scores, document_ids):
    print(f"Document: {doc_id}, Score: {score}")
    print("-----------")
    print(doc)
    print("-----------")
    print()


documents, document_scores, document_ids = model.search_documents_by_keywords(keywords=["offers", "negative"], num_docs=3)
for doc, score, doc_id in zip(documents, document_scores, document_ids):
    print(f"Document: {doc_id}, Score: {score}")
    print("-----------")
    print(doc)
    print("-----------")
    print()


words, word_scores = model.similar_words(keywords=["app"], keywords_neg=[], num_words=20)
for word, score in zip(words, word_scores):
    print(f"{word} {score}")