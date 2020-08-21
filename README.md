# Project: NewsSentiment & TickerTracking

Contributors:

* Jatinder Singh
* Renae Martinez
* Heena Roy

## Project Abstract
Analyze stock data in correlation to sentiments from different stock news sources and determine what (if any) impact it has on stock prices.

## Scope and Assumptions
- Ticker: TSLA
    - To create a working prototype, we decided to look at one stock only. We picked TSLA as it seems to have more daily news and articles than many other stocks, as well as volatility in its price. 


- Stock price source: YFinance
    - We chose to use YFinance sdk to download historical stock prices as it was fairly easy to use and gave us all we needed.
    

- Time Period: 2019
    - We did not want to include 2020 as the year has seen some unprecedented events in the history of the stock market.


- News/Articles Data: StockNewsAPI.com & BreakingAPI.com
    - Initial TSLA articles were pulled from StockNewsAPI. However, StockNewsApi does not return full articles, only snippets, headlines, URLs and its own sentiment analysis scores. A custom function was created to take the news URL from StockNewsAPI, pass it to BreakingAPI, and pull the full text of the article. 

|![StockNewsAPI Dataframe](./images/df_sna1.png "DataFrame created from data downloaded from StockNewsApi")|
|---|
|DataFrame created from data downloaded from StockNewsApi|

|![w BreakingAPI DataFrame](./images/df_ba1.png "Modified Dataframe after downloading full text from breakingapi.com")|
|---|
|Modified Dataframe after downloading full text from breakingapi.com|


## Data Preparation, Cleaning and Processing

### Tools for Extracting Sentiments

Sentiment analysis is a machine learning technique that detects polarity – positive or negative - within text. The models often focus on polarity but also on emotions e.g. angry, happy, sad, etc. or even intentions, interested or not interested. We only looked at polarity, a range of positive to negative, +1 to -1.

Because our focus is on stock market returns, we looked at articles that spoke to a particular stock – Tesla in our case – and decided to see if sentiments of news articles do, in fact, have a bearing on stock prices. We only looked at the articles of 2 sources, The Motley Fool and Zacks Investment Research for the year 2019. Full texts of 289 articles were processed through 2 different Sentiment Analysis Models:

- VADER (Valence Aware Dictionary and sEntiment Reasoner) is a lexicon and rule-based sentiment analysis tool that is specifically attuned to sentiments in social media. This tool gives us 4 rating scores: Positive, Neutral, Negative, and Compound. Positive, Neutral, and Negative are given a ranking score within each document, with a Compound score for total polarity. The Compound score was used in our analysis. 
 
- TextBlob is a python library for processing textual data. This tool gives us 2 ratings scores: Polarity and Subjectivity. The Subjectivity scored whether the document was objectively or subjectively weighted, while the Polarity score rated the article based on positive or negative sentiment. The Polarity score was used in our analysis. Unlike VADER, input text needs to be preprocessed to tokens, because of this we ran this model twice, we wanted to see if this had any effect on the polarity and the predictive power of the sentiment:

    - Lemmatizer breaks the word down to its root, dropping contractions, plurals, etc. 
    
    - PorterStemmer breaks the word down to its roots as does the Lemmatizer, however stemming is much stricter at stripping the words.
    

- We also used a sentiment score from StockNewsAPI.  We took their analysis at face value.

### Bag of Words Visualizations

After processing the Sentiments Analysis, we were left with a "Bag of Words" from the full articles. We realize that these words could be of value as well, and additional analysis could be done to see if there is any predictive power. However because that was not within the scope of the project, we did not pursue it further. We did play with the "Bag of Words" to create a few visuals below:

|![Tesla Word Cloud](./images/tsla_image.gif "Tesla Word Cloud") | ![Sunburst](./images/sunburst.gif "Tesla Sunburst")|![Co-Occurence Network](./images/co_occurrence.gif "Tesla Co-Occurrence Network") | ![PLDavis](./images/pldavis.gif "Tesla Word Community Topics")|
|:---:|:---:|:---:|:---:|
| Tesla Word Cloud | Tesla Word Sunburst | Tesla Word Co-Occurrence Network | Tesla Word Community Topics |


### Merging Stock Prices and News Sentiment Data

We created dataframes containing TSLA stock data, sentiment analysis from stocknewsapi and CSV files created by our custom code and joined them to create a final dataframe that was used for regression analysis.

|![StockNewsAPI Dataframe 2](./images/df_sna2.png "Stocknewsapi sentiment translated to a numeric value")|
|---|
|Stocknewsapi sentiment translated to a numeric value|

|![Sentiments Dataframe 1](./images/df_senti1.png "Added sentiments from our custom code to the previous dataframe")|
|---|
|Added sentiments from our custom code to the previous dataframe|

|![Sentiments Dataframe 2](./images/df_senti2.png "Cleaned-up dataframe containing only the columns we needed")|
|---|
|Cleaned-up dataframe containing only the columns we needed|

|![Dataframe 1 for Regression](./images/df_reg1.png "Merged Tesla stock closing prices for the respective dates (DF1 for Regression Analysis)")|
|---|
|Merged Tesla stock closing prices for the respective dates (DataFrame 1 used for Regression Analysis)|

|![Sentiments Dataframe 3](./images/df_senti3.png "Dataframe grouped by date and source")|
|---|
|Dataframe grouped by date and source|

|![Dataframe 2 for Regression](./images/df_reg2.png "Source_name values transposed to columns (DF2 for Regression Analysis)")|
|---|
|Source_name values transposed to columns (DataFrame 2 used for Regression Analysis)|


# Regression Analysis

## Sentiment Analysis Data Preparation

Since news articles for Tesla were not available for every trading day, we filled the "missing" sentiment analysis with a hysteresis. We took the adjacent sentiments scores and filled in the "missing" sentiment scores with the average of the two adjacent scores.  The reasoning to use the average is that the sentiment scores would not vary too far from the average.

## Monte Carlo Regression Analysis

A monte carlo analysis was performed on 2 different sentiment analysis data frames to determine which feature (sentiment analysis) has the best correlation to Tesla's daily returns.
- **All Sentiments:** Data frame with a compound sentitment score for all news source articles on Tesla per day. Data frame includes 4 features: StockNews, Vader, Textblob Lemmatized, and Textblob Stemmed.
- **Sentiments by News Source:** Data frame with a compound sentiment score by news source article on Tesla per day. Data frame includes 8 features: Zacks StockNews, Zacks Vader, Zacks Textblob Lemmatized, Zacks Textblob Stemmed, Motley Fool StockNews, Motley Fool Vader, Motley Fool Textblob Lemmatized, and Motley Fool Textblob Stemmed.

The monte carlo analysis was performed using two different monte carlo functions which iterated over 20 days. We chose to cap the monte carlo analysis to 20 days as the impact of a given news article would decay over time and a more current news article would be more relevant to determining a correlation with the daily returns. We chose to perform the monte carlo using the Lasso Regression as the Lasso provides a features selection. From our research there are 2 primary ways to run the Lasso Regression using sklearn:

1. Model 1: Lasso Regression
    - A parameter to running the Lasso Regression model was to define the alpha.  The alpha dictates how restrictive the model will be in assessing how the features impact the target - in our case, Tesla's stock return. Since adjusting the alpha changed the results, we ran both data frames with the largest alpha that resulted in a feature selection.  In our case, the alpha was 0.0001.
    - Referenced source code: https://towardsdatascience.com/ridge-and-lasso-regression-a-complete-guide-with-python-scikit-learn-e20e34bcbf0b


2. Model 2: Lasso Regression using LogisticRegression
    - In order to run the Lasso Regression using LogisticRegression, we first ran a LogisticRegression with parameters penalty L1 and solver liblinear.  We then used the SelectFromModel to fetch the winning features.
    - Referenced source code: https://towardsdatascience.com/feature-selection-using-regularisation-a3678b71e499

### Monte Carlo Results - All Sentiments:

We performed the monte carlo analysis using both models on the All Sentiments data frame. We stored the winning feature with the corresponding coefficient per day and we calculated the overall winner (feature with the highest coefficient) over the 20 days. The graphs below show the overall winning feature with the corresponding number of days shifted per model as well as the daily winning feature per day.    

|![Model1 Regression All Sentiments](./images/Model1_All.png "Model 1 Lasso Regression") | ![Model1 Regression All Sentiments LR](./images/Model2_All.png "Model 1 Lasso Regression using LogisticRegression")|
|:---:|:---:|
| Model 1 (All Sentiments): Daily Returns and Textblob Lemmatized | Model 2 (All Sentiments): Daily Returns and Textblob Lemmatized |

|![Model1 Regression All Sentiments gif](./images/Model1_All_Iterations.gif "Model 1 Lasso Regression") | ![Model1 Regression All Sentiments LR gif](./images/Model2_All_Iterations.gif "Model 1 Lasso Regression using LogisticRegression")|
|:---:|:---:|
| Model 1 (All Sentiments): Winner Feature per Iteration | Model 2 (All Sentiments): Winner Feature per Iteration |

Both models were consistent in choosing the Textblob Lemmatized sentiment analysis with a 0-day shift as the best correlation to Tesla's daily returns. However, when assessing which feature won the most times over the 20 days, it appears that Textblob Stemmed won more than Textblob Lemmatized for Model 1.  Textblob Lemmatized won more for Model 2, which is consistent with Textblob Lemmatized being the overall winner.


### Monte Carlo Results - Sentiments by News Source:

The same analysis was repeated using the Sentiments by News Source data frame. The results are shown below:

|![Model2 Regression News Source](./images/Model1_NewsSource.png "Model 2 Lasso Regression") | ![Model2 Regression News Source LR](./images/Model2_NewsSource.png "Model 2 Lasso Regression using LogisticRegression")|
|:---:|:---:|
| Model 1 (Sentiments by News Source): Daily Returns and Motley Textblob Stemmed | Model 2 (Sentiments by News Source): Daily Returns and Zacks Stocknews |

|![Model2 Regression News Source gif](./images/Model1_News_Iterations.gif "Model 2 Lasso Regression") | ![Model2 Regression News Source LR gif](./images/Model2_News_Iterations.gif "Model 2 Lasso Regression using LogisticRegression")|
|:---:|:---:|
| Model 1 (Sentiments by News Source): Winner Feature per Iteration | Model 2 (Sentiments by News Source): Winner Feature per Iteration |

Model 1 determined that Motley Textblob Stemmed with a 0-day shift has the best correlation to Tesla's daily returns, whereas Model 2 determined that Zacks StockNews with a 1-day shift has the best correlation.  When assessing which feature won the most over the 20-day analysis, it appears that both Motley Textblob Lemmatized and Motley Textblob Stemmed won the most for Model 1 and Zacks StockNews and Zacks Vader won the most for Model 2.

## Conclusion

From analyzing the data, we confirmed that the overall winner shown on the hvplots isn’t necessarily the feature that won the most times over the 20 day iteration. Also, when overlaying the overall winner with the daily returns the correlation wasn't as strong as we would have like to have seen. In short, we wouldn't use these tools to predict stock returns just yet. 

This was a great exercise in learning more about the off-the-shelf regression models, sentiment analysis, and news APIs available. We wanted to add algorithmic trading to the project as well, however, we felt more analysis of the data and performances of the models was needed. Future enhancements to the project include:

- Run analysis over a longer period of time
- Data mine for additional articles on stock for more sentiment analysis data
- Further customization of sentiment analysis on given article
    - Determine if article was written for Tesla vs Tesla being mentioned in the article. 
- Integrate volume of sales data and sell ‘type’ (industry vs retail) into regression analysis


## Resources

1. https://breakingapi.com/article-extraction-api

2. https://medium.com/@Intellica.AI/vader-ibm-watson-or-textblob-which-is-better-for-unsupervised-sentiment-analysis-db4143a39445

3. https://towardsdatascience.com/sentiment-analysis-vader-or-textblob-ff25514ac540

4. https://towardsdatascience.com/create-word-cloud-into-any-shape-you-want-using-python-d0b88834bc32

5. https://www.kaggle.com/takanobu0210/twitter-sentiment-eda-using-nlplot
