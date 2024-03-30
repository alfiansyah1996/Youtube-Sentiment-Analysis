import streamlit as st
st.set_theme('light')

col1,col2 = st.columns([1,3])
with col1:
    st.image(
            "https://cdn-icons-png.freepik.com/256/1384/1384060.png",
            width=140,
        )   
with col2:
    st.title("SENTIMENTER")
    st.markdown('''<div style="text-align: justify;">
                <span style="color:red">
                <strong>ANALYZE YOUTUBE SENTIMENTS IN REAL-TIME WITH STREAMLIT</strong>
                </span>
                </div>''', unsafe_allow_html=True)
st.markdown("""---""")


import re

import re
def extract_video_id(url):
    pattern = r'(?:https?://)?(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/)([a-zA-Z0-9_-]{11})'
    match = re.search(pattern, url)
    if match:
        return match.group(1)
    else:
        return None 

video_url = st.text_input(label = "Enter Youtube Video URL")
video_id = extract_video_id(video_url)
if st.button("Run"):
  st.markdown("""---""")
  if video_id:
    col1,col2,col3 = st.columns([1,2,1])
    with col2:
            st.markdown('''<div style="text-align: center;">
                        <strong>YOUTUBE VIDEO</strong>
                        </div>''', unsafe_allow_html=True)
            st.video(f"https://www.youtube.com/watch?v={video_id}")
    st.markdown("""---""")
    import googleapiclient.discovery
    import pandas as pd
    api_service_name = "youtube"
    api_version = "v3"
    api_key = "AIzaSyCoA4H8NUwdDyf69S-PliKTlH0Cc-61kzE"
    youtube = googleapiclient.discovery.build(
        api_service_name, api_version, developerKey=api_key)
    request = youtube.commentThreads().list(
        part="snippet",
        videoId= video_id,
        maxResults=100)
    response = request.execute()
    comments = []
    for item in response['items']:
        comment = item['snippet']['topLevelComment']['snippet']
        comments.append([
            comment['authorDisplayName'],
            comment['publishedAt'],
            comment['likeCount'],
            comment['textOriginal'],
        ])
    while (1 == 1):
        try:
            nextrun = response['nextPageToken']
        except KeyError:
            break
        nextrun = response['nextPageToken']
        nextRequest = youtube.commentThreads().list(part="snippet", videoId=video_id, maxResults=100, pageToken=nextrun)
        response = nextRequest.execute()
        for item in response['items']:
            comment = item['snippet']['topLevelComment']['snippet']
            comments.append([
                comment['authorDisplayName'],
                comment['publishedAt'],
                comment['likeCount'],
                comment['textOriginal'],
            ])
    df = pd.DataFrame(comments, columns=['user_name', 'comment_at', 'total_like', 'comment_text'])
    df['id'] = df.index
    import re 
    df['comment_text'] = df['comment_text'].str.lower()
    df['comment_text'] = df['comment_text'].apply(lambda x: re.sub(r'[^\w\s]', '', x))
    import nltk
    try:
        nltk.data.find('vader_lexicon')
    except LookupError:
        nltk.download('vader_lexicon')
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    model = SentimentIntensityAnalyzer()
    results = {}
    for i, row in df.iterrows():
        comment_text = row['comment_text']
        id = row['id']
        results[id] = model.polarity_scores(comment_text)
    vaders = pd.DataFrame(results)
    vaders = vaders.T
    vaders = vaders.reset_index().rename(columns={'index': 'id'})
    df = df.merge(vaders, how='left', on='id')
    df['sentiment_class'] = 'neutral'
    df.loc[df['compound'] > 0, 'sentiment_class'] = 'positive'
    df.loc[df['compound'] < 0, 'sentiment_class'] = 'negative'
    st.markdown('''<div style="text-align: center;">
                        <strong>DATA VISUALIZATION</strong>
                        </div>''', unsafe_allow_html=True)
    col1,col2, = st.columns([3,2])
    with col1:
        import matplotlib.pyplot as plt
        st.set_option('deprecation.showPyplotGlobalUse', False)
        colors = {'positive': 'tab:green', 'negative': 'tab:red', 'neutral': 'tab:blue'}
        sentiment_counts = df['sentiment_class'].value_counts().sort_index()
        plt.figure(figsize=(6, 3))
        for sentiment_class, count in sentiment_counts.items():
            plt.barh(sentiment_class, count, color=colors[sentiment_class])
            plt.text(count, sentiment_class, str(count), va='center')
        plt.xlabel('Total')
        plt.ylabel('Sentiment Class')
        st.pyplot()
    with col2:
        plt.figure(figsize=(2, 2))
        plt.pie(sentiment_counts, 
                labels=sentiment_counts.index, 
                colors=[colors[x] for x in sentiment_counts.index], 
                autopct='%1.1f%%', 
                startangle=140)
        plt.axis('equal')
        st.pyplot()
    st.markdown("""---""")
    st.markdown('''<div style="text-align: center;">
                        <strong>RAW DATA</strong>
                        </div>''', unsafe_allow_html=True)
    df

