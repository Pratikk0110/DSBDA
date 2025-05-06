z# Plot feedback distribution
sns.countplot(data=df, x='feedback')
plt.title("Positive vs Negative Feedback")
plt.xlabel("Feedback (0 = Negative, 1 = Positive)")
plt.ylabel("Number of Reviews")
plt.show()
df['lower_text'] = df['verified_reviews'].str.lower()
df['no_punct_text'] = df['lower_text'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
# Emoji pattern
emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)

df['no_emoji_text'] = df['no_punct_text'].apply(lambda x: emoji_pattern.sub(r'', x))
df['tokens'] = df['no_emoji_text'].apply(word_tokenize)
stop_words = set(stopwords.words('english'))

df['filtered_tokens'] = df['tokens'].apply(lambda x: [word for word in x if word not in stop_words])
print(df[['verified_reviews', 'filtered_tokens']].head())
