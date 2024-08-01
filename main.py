import matplotlib.pyplot as plt
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from tqdm import tqdm
import nltk


nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

# Load the CSV file with file path
df = pd.read_csv()

# Sentiment Analyzer Initialization
sia = SentimentIntensityAnalyzer()

# Dictionary to store results
res = {}

# Iteration over each row in the DataFrame
for i, row in tqdm(df.iterrows(), total=len(df)):
    text = str(row['content'])  # Convert the title to string
    myid = row['id']


    if text and text != 'nan':
        res[myid] = sia.polarity_scores(text)
    else:
        res[myid] = {'neg': 0.0, 'neu': 0.0, 'pos': 0.0, 'compound': 0.0}

# Converts the results dictionary to a DataFrame
vaders = pd.DataFrame(res).T
vaders = vaders.reset_index().rename(columns={'index': 'id'})

# Merging of the sentiment scores with the original DataFrame
vaders = vaders.merge(df, how='left', on='id')

# Calculate and print the average compound score
average_compound = vaders['compound'].mean()
print(f"Average Compound Score: {average_compound}")

# Compound Score Display
print(res)
print(vaders[['id', 'compound']])

# Plot for count of reviews by stars
ax = df['rating'].value_counts().sort_index().plot(kind='bar', title='Count of Reviews by Stars', figsize=(10, 5))
plt.xlabel('Star Rating')
plt.ylabel('Count')
plt.show()


