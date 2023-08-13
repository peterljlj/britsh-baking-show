#!/usr/bin/env python
# coding: utf-8

# # Great British Bake Off 👩‍🍳🍰🇬🇧
# 
# ## Outline of the Project 
# 
# 
# -  [About the Show 📺](#about_show)
# -  [About the Data 💾](#about_data)
# -  Section 1. [Exploratory Data Analysis 🔎](#section1)  
# -  Section 2. [Popular Ingredients 🍊 🍫](#section2) 
# -  Section 3. [Gender Balance 👩⚖️🧑🏼](#section3)  
# -  Section 4. [Well-Deserved? 🥇](#section4)  
# -  Section 5. [Devilishly Difficult Challenges 😈](#section5)  
# -  Section 6. [Piece of Cake? 🍰](#section6) 
# -  Section 7. [Recipe Name Generator 👩‍🍳🖨️](#section7) 
# -  Section 8. [Dishwashing 🧼🍽️](#section8) 

# In[1]:


import babypandas as bpd
import numpy as np

import matplotlib.pyplot as plt
plt.style.use('ggplot')

import otter
grader = otter.Notebook()


# <a id='about_show'></a>
# ## About the Show 📺
# 
# The Great British Bake Off (known in the US as the Great British Baking Show) is a competition-style television show where amateur bakers participate in themed baking challenges. Each week's episode revolves around a theme; past themes include Bread Week, Cake Week, Vegan Week, and Italian Week. In each episode, the bakers are given three timed challenges based on the week's theme: the Signature Challenge, the Technical Challenge, and the Showstopper Challenge. 
# 
# In the Signature Challenge, the judges broadly specify what the bakers should make, and the bakers have freedom to use flavors, techniques, and recipes as they wish. The Signature Challenge earns its name because it's an opportunity for bakers to express themselves and their unique baking style to both the judges and the viewers at home. Many of the Signature Challenge bakes come from tried-and-tested recipes that contestants like to bake for their friends and families. For example, during Festivals Week in Season 10, the bakers were tasked with creating 24 buns themed around a world festival or holiday. Contestant Henry Bird made [these Chocolate Kardemummabullar](https://thegreatbritishbakeoff.co.uk/recipes/all/henry-chocolate-kardemummabullar/).
# 
# <img src="images/signature_bake.png" width="500" height="500">
# 
# In the Technical Challenge, bakers have no idea what they will be asked to create until the timer for the challenge starts. This means they can't prepare for it, and they have to rely on their baking knowledge and intuition. The Technical Challenge earns its name because it tests the bakers' technical knowledge of baking as a discipline.  Each Technical Challenge is posed by one particular judge, and uses a recipe from the judge's own personal collection. Bakers are provided with ingredients and a recipe, which is usually extremely basic, sometimes lacking ingredient measurements or containing single steps like "make a shortcrust pastry." The finished products are judged blind and ranked from worst to best. An example of a Technical Challenge includes [judge Paul Hollywood's Baklava](https://thegreatbritishbakeoff.co.uk/recipes/all/paul-hollywood-baklava/). 
# 
# <img src="images/baklava.jpg" width="500" height="500">
# 
# The third challenge, the Showstopper, is similar to the Signature Challenge, in that bakers are given requirements ahead of time and have freedom to create their own recipes and prepare ahead of time. The main difference is that the Showstoppers are more challenging and larger-scale. The judges are looking for bakes that are breathtaking in both their appearance and their taste. For example, during Bread Week in Season 6, the bakers were asked to create a 3-D bread sculpture. Contestant Paul Jagger impressed the judges and millions of viewers with his *King of the Jungle* lion sculpture.
# 
# <img src="images/lion_bread_sculpture.png" width="500" height="500">
# 
# Each episode of the show features all three challenges. The contestants' bakes are tasted and assessed by two judges, and at the end of each episode, the hosts announce who will be eliminated from the competition and who will be recognized with a special award of "Star Baker" ⭐ (introduced in Season 2). Typically, one contestant is eliminated and one is crowned Star Baker ⭐, but on occasion there have been special cases in which zero or multiple people were eliminated or awarded Star Baker ⭐. 
# 
# The final episode of each season is held when there are just three bakers remaining. All three bakers compete in the final, and at the end, one winner is chosen and each of the others is considered a "runner-up". 

# <a id='about_data'></a>
# ## About the Data 💾
# 
# For this project, we'll be using a few different datasets, which we've loaded in and saved in DataFrames called
# - `baker_weeks`, 
# - `challenge_results`,  
# - `technical_challenge_recipes`, and
# - `bakers`.
# 
# Note that while the Great British Bake Off has filmed thirteen seasons, our datasets do not include the most recent seasons. Since our datasets come from different [sources](#sources), some of these datasets include more seasons than others. In addition, the number of bakers each season has varied, but all seasons have filmed one episode per week.
#  
# The `baker_weeks` DataFrame includes a breakdown of each baker's performance each week (that is, each episode), for the first eleven seasons of the show. Each row represents information **for one baker for one week**. This means that each baker will appear in the DataFrame multiple times. Bakers will continue to appear in the DataFrame even in weeks after they got eliminated, so these rows will have missing values (`NaN`).  
#   
# The `'Week Name'` column contains the theme of that week's episode. We also have the baker's name, gender ("M" or "F" are the only options), and age, the season number (also called the series number in other DataFrames), and the week number within that season. There are columns that indicate whether each baker was a Star Baker ⭐ that week, was eliminated that week, competed that week, or went on to win the season's competition. A few columns require more explanation about the show:
# - `'Judge'` is either "Mary" or "Prue". For the first seven seasons, the show's two judges were [Paul Hollywood](https://en.wikipedia.org/wiki/Paul_Hollywood) and [Mary Berry](https://en.wikipedia.org/wiki/Mary_Berry). After that, the show switched networks and Mary Berry was replaced by [Prue Leith](https://en.wikipedia.org/wiki/Prue_Leith). Since Paul Hollywood was a judge every season, the `'Judge'` column contains the name of the other judge.
# - `'technical_rank'` contains a number reflecting each baker's ranking in the Technical Challenge (with 1 meaning 1st place, 2 meaning 2nd place, etc.)
# -  `'Signature Handshake'` and `'Showstopper Handshake'` contain information on whether the contestant received a handshake 	🤝 from judge Paul Hollywood as he tasted their bake. Paul has a reputation for giving praise sparingly, and his so-called "[Hollywood Handshakes 	🤝](https://hollywoodhandshakes.com/)" are considered a great honor. 
#   
# Run the cell below to load in the `baker_weeks` data.

# In[2]:


baker_weeks = bpd.read_csv('data/baker_weeks.csv')
baker_weeks


# The `challenge_results` DataFrame contains information on each challenge, with each row representing one baker in one specific episode. As in `baker_weeks`, bakers will reappear multiple times, even after they get eliminated, hence the abundance of `NaN` values. This dataset contains information for the first ten seasons of the show.
# 
# The `'result'` column indicates whether a baker was eliminated or stayed in the competition. Values of "OUT" and "Runner-up" mean the baker was eliminated, and values of "IN", "STAR BAKER", and "WINNER" mean that the baker stayed in the competition. There is one instance of "WD" in this column for someone who withdrew from the competition, and one instance of "A" for someone who was absent one week. We'll ignore both of these.
# 
# The `'technical'` column contains the baker's rank in the Technical Challenge, and the `'signature'` and `'showstopper'` columns contain the names of the recipes the baker prepared for these challenges.  
# 
# Run the cell below to load in the `challenge_results` data.

# In[3]:


challenge_results = bpd.read_csv('data/challenge_results.csv')
challenge_results


# The `technical_challenge_recipes` DataFrame contains information about each recipe that was given as a Technical Challenge in the first nine seasons. The columns specify the season (`'Ssn'`) and episode (`'Ep'`) that each recipe was baked in, which judge's recipe collection it came from (`'Whose'`), and several aspects of the recipe's complexity:
# - number of components (`'Components'`), which are recipes used within the main recipe, such as a frosting or filling,
# - number of ingredients (`'IngredCount'`),
# - number of sentences in the instructions (`'RecipeSentences'`),
# - number of dirty dishes produced (`'Dishes'`), and
# - difficulty (`'DifficultyScore'`). 
# 
# Run the cell below to load in the `technical_challenge_recipes` data.

# In[4]:


technical_challenge_recipes = bpd.read_csv('data/technical_challenge_recipes.csv')
technical_challenge_recipes


# The `bakers` DataFrame contains a row for each baker from the first ten seasons, with detailed information about their results in the show, particularly about their performance in the Technical Challenge:
# - `'technical_winner'`: number of times they won,
# - `'technical_top3'`: number of times they placed in the top three,
# - `'technical_bottom'`: number of times they placed last,
# - `'technical_highest'`: highest (best) rank they ever earned,
# - `'technical_lowest'`: lowest (worst) rank they ever earned, and
# - `'technical_median'`: median of all ranks they ever earned.
# 
# It also includes information about when they appeared on the show and their demographics such as `'occupation'` and `'hometown'`.
# 
# Run the cell below to load in the `bakers` data.

# In[5]:


bakers = bpd.read_csv('data/bakers.csv')
bakers


# Our data comes from a variety of different [sources](#sources) and may contain errors. If you find any errors in the data, do not attempt to fix them; just analyze the data you are given. 

# <a id='section1'></a>
# ## Section 1: Exploratory Data Analysis 🔎

# To start, we’ll perform some exploratory data analysis to get better acquainted with our data.
# 
# A common sentiment among long-time viewers of the show is that the baking challenges are getting harder over time. Does the data support this? 
# 
# **Question 1.1.** Using the `technical_challenge_recipes` DataFrame, create an overlaid line plot that shows the season number on the horizontal axis and on the vertical axis:
# - average number of dirty dishes produced by recipes in that season, 
# - average number of components in recipes in that season,
# - average number of ingredients in recipes in that season, and 
# - average difficulty score of recipes in that season.

# <!-- BEGIN QUESTION -->
# 
# <!--
# BEGIN QUESTION
# name: q1_1
# points: 1
# manual: true
# -->

# In[6]:


# Create your overlaid line plot here.
means = technical_challenge_recipes.groupby('Ssn').mean().reset_index()
plt.plot(means.get('Ssn'), means.get('Dishes'), label = 'Dishes')
plt.plot(means.get('Ssn'), means.get('Components'), label = 'Components')
plt.plot(means.get('Ssn'), means.get('IngredCount'), label = 'Ingredients')
plt.plot(means.get('Ssn'), means.get('DifficultyScore'), label = 'Difficulty')
plt.legend()


# <!-- END QUESTION -->
# 
# 
# 
# Some of the recipes the contestants bake are quite complicated. Let's look at some especially long recipe titles.
# 
# **Question 1.2.** Using the `challenge_results` DataFrame, which Signature Challenge recipe had the longest name? Save the result as `longest_signature`. 
# 
# Similarly, which Showstopper Challenge recipe had the longest name? Save the result as `longest_showstopper`. In both cases, longest means having the most individual characters, including punctuation and whitespace.

# In[7]:


sig = challenge_results.assign(len =challenge_results.get('signature').str.len()).sort_values(by='len',ascending=False)
show = challenge_results.assign(len =challenge_results.get('showstopper').str.len()).sort_values(by='len',ascending=False)


# In[8]:


longest_signature = sig.get('signature').iloc[0]
print("Longest signature name: ", longest_signature, "\n")

longest_showstopper = show.get('showstopper').iloc[0]
print("Longest showstopper name: ", longest_showstopper, "\n")


# In[9]:


grader.check("q1_2")


# Notice that each of these recipes actually involves multiple items. Often the bakers have to make displays of baked goods with multiple components as part of a single challenge.

# Another common sentiment among viewers is that the show favors younger people 👧🏽. To further explore the bakers' ages, let's convert the `'age'` column to a categorical variable:

# **Question 1.3.** Add an additional column called `'age_category'` to the `bakers` DataFrame, based on the following age categorization:
# 
# | Age            | Category    |
# | -------------- | ----------- |
# | (0, 39]        | Young       |
# | (39, 59]       | Middle-Aged |
# | (59, $\infty$] | Elderly     |

# In[10]:


age=np.array([])
for i in np.arange(120):
    if bakers.get('age').iloc[i]<=39:
        age=np.append(age,'Young')
    elif bakers.get('age').iloc[i]<=59:
        age=np.append(age, 'Middle-Aged')
    elif bakers.get('age').iloc[i]>59:
        age=np.append(age, 'Elderly')


# In[11]:


bakers = bakers.assign(age_category=age)
bakers


# In[12]:


grader.check("q1_3")


# **Question 1.4.** Using the information in the new `'age_category'` column, set `age_prop` to a Series indexed by `'age_category'`, where the values are the proportions of bakers in each `'age_category'`.

# In[13]:


age_prop = bakers.groupby('age_category').count().get('series')/120
age_prop


# In[14]:


grader.check("q1_4")


# You should see that a majority of the participants are young!

# Next, we'll investigate baker occupations. Do bakers on the show tend to hold certain types of jobs? Maybe they work in the food industry, do a lot of cooking at home, or have creative jobs like an artist 🎨 or photographer 📷. Some baking challenges even require significant feats of construction 🏗️, so maybe architects or engineers are popular.
# 
# **Question 1.5.** Using the `bakers` DataFrame, create an array of occupations held by more than one contestant on the show. Save the array in a variable called `popular_jobs`.

# In[15]:


bakers


# In[16]:


popular_jobs = np.array(bakers.groupby('occupation').count()[bakers.groupby('occupation').count().get('series')>1].reset_index().get('occupation'))
popular_jobs 


# In[17]:


grader.check("q1_5")


# <a id='section2'></a>
# ## Section 2: Popular Ingredients 🍊 🍫
# 
# 
# Now, we'll try to answer some questions about popular ingredients used in bakers' recipes, and whether there's any connection between certain ingredients and a baker's success in the competition. Our data doesn't exactly include ingredient lists, but we do have recipe titles for the Signature and Showstopper Challenges in `challenge_results`, so we can look for common words there. We'll focus specifically on the Signature Challenge, as it's one in which bakers are able to be creative and showcase a recipe unique to them, and so they have complete freedom to use whatever ingredients they want. 
# 
# 
# The DataFrame below contains all the rows of `challenge_results` with an entry in the `'signature'` column. We've also dropped the columns relating to the Technical and Showstopper Challenges, since we'll be focusing on the Signature Challenge here.

# In[18]:


signatures = bpd.read_csv('data/signatures.csv')
signatures


# **Question 2.1.** We want to clean up the text so we can find words that appear frequently in many recipe titles. Write a function named `clean_up_text` that takes the name of a single recipe as input and returns a cleaned-up version of the name with these changes:
# - Remove any of these characters: `(`, `)`, `'`, `"`, `;`, `,` (open and close parentheses, single and double quotes, semicolons, commas) 
# - Convert to lowercase.
# 
# *Hint*: Use the `.replace()` string method.

# In[19]:


def clean_up_text(recipe):
    '''Returns a lowercase version of recipe with certain special characters removed.'''
    for char in ['(', ')', "'", '"', ';', ',']:
        recipe = recipe.replace(char, '').lower()
    return recipe


# In[20]:


clean_up_text('sdgdLggs, & ; ( e')


# In[21]:


grader.check("q2_1")


# **Question 2.2.** Now that we've created a function to clean the titles, replace the entries in the `'signature'` column of the `signatures` DataFrame with the cleaned version of those recipe titles. Then, assign a new column to the `signatures` DataFrame called `'words'` that contains a list of all the words in the cleaned recipe title, in lowercase. We'll define a word as any chunk of text separated from others by spaces. For example, 
# - a recipe title of `"Mint, Lilac, & Blackberry Cake"`
# - should become `"mint lilac & blackberry cake"` when cleaned,
# - with a corresponding word list of `["mint", "lilac", "&", "blackberry", "cake"]`.

# In[22]:


signature2 = signatures.assign(signature= signatures.get('signature').apply(clean_up_text))


# In[23]:


signatures = signature2.assign(words= signature2.get('signature').str.split())
signatures


# In[24]:


grader.check("q2_2")


# 
# For the next question, you'll need to know something interesting about how lists work in Python: when you sum two lists together, the output is one giant list that contains all the elements in both lists combined. An example is shown below.
# 

# In[25]:


['List', 'combining'] + ['is', 'my', "passion"]


# **Question 2.3.** Combine all the words in the `'words'` column into one big list. Save that list in the variable `all_words`.

# In[26]:


all_words = signatures.get('words').sum()
# Just display the first ten words.
all_words[:10]


# In[27]:


grader.check("q2_3")


# **Question 2.4.** Write a function called `most_common` that takes as input any list of words, and finds the ten most common words in that list. Your function should output a DataFrame with 10 rows, indexed by `'word'`, with one column called `'count'` containing a count of how many times each word appeared in the input list. Order the rows in descending order of `'count'`.
# 
# Then use your function to find the ten most common words in `all_words`. These are the words that appeared the most in Signature Challenge recipe titles. Save the resulting DataFrame as `common_words_df`.
# 
# *Hint*: Leverage the power of `groupby`.

# In[28]:


bpd.DataFrame().assign


# In[29]:


def most_common(word_list):
    '''Returns a DataFrame with the ten most common words in word_list, in descending order.'''
    df = bpd.DataFrame().assign(word = word_list, count=word_list)
    return df.groupby('word').count().sort_values(by='count', ascending=False).iloc[:10]

common_words_df = most_common(all_words)
common_words_df


# In[30]:


grader.check("q2_4")


# You should find that the most common word is one that doesn't give us any information about the recipe. To deal with that, let's omit common words, which are transition words like "and" and "with", as well as words like "cake" and "bread" that appear in the titles of many recipes that were featured in Cake Week or Bread Week.
# 
# **Question 2.5.** Make a list called `words_to_omit` with all the words that appear anywhere in the `'Week Name'` column of the `baker_weeks` DataFrame. 
# 
# The words in `words_to_omit` should be in all lowercase, regardless of their case in the `'Week Name'` column. Also, `words_to_omit` should not have any duplicate words. Even if a word appears in the `'Week Name'` column multiple times, it should only appear once in `words_to_omit`.
# 
# For example, one week's theme was "Pie and Tart", so the words "pie", "and", and "tart" should all be elements of `words_to_omit`.

# In[31]:


omitlist = np.unique(baker_weeks.get('Week Name').str.split().sum())
omit = np.array([])
for char in omitlist:
    omit = np.append(omit, char.lower())


# In[32]:


words_to_omit = list(omit)
# Just display the first ten words.
words_to_omit[:10]


# In[33]:


grader.check("q2_5")


# For the next question, you'll need to use the `in` operator in python. The `in` operator checks if a value is an element of a list. For example:

# In[34]:


"macaroni" in ["macaroni", "and", "cheese"]


# In[35]:


"mac" in ["macaroni", "and", "cheese"]


# **Question 2.6.** Create a new DataFrame called `meaningful`, with the same data as the `signatures` DataFrame plus an extra column called `'meaningful_words'`, containing a list of all the words that appear in the `'words'` column, except with these words omitted:
# - "and"
# - "&"
# - "with"
# - any word in `words_to_omit`
#     
# *Hint*: Create a function that takes as input one entry of the `'words'` column (a single list of words, corresponding to one recipe title) and returns a list of those same words, except with certain ones omitted. To do that, loop through the words in the list and append the words that should not be omitted to an empty array. Finally, convert the array of non-omitted words to a list before returning.

# In[36]:


words_to_omit + ['&', 'with', 'and']


# In[37]:


def omit(words):
    wordslist = np.array([])
    for char in words:
        if char not in words_to_omit + ['&', 'with', 'and']:
            wordslist = np.append(wordslist, char)
    return list(wordslist)


# In[38]:


meaningful = signatures.assign(meaningful_words = signatures.get('words').apply(omit))
meaningful


# In[39]:


grader.check("q2_6")


# **Question 2.7.** Now, find the ten most common words **among only the meaningful ones**. Create a DataFrame called `popular_words` formatted in the same way as `common_words_df`, which you created in Question 2.4.

# In[40]:


popular_words = most_common(meaningful.get('meaningful_words').sum())
popular_words


# In[41]:


grader.check("q2_7")


# The most common word should now be the name of a popular ingredient or flavor in British baking. Yum!
# 
# **Question 2.8.** Now let's try to figure out which meaningful words were most popular in Signature Challenge recipe titles among bakers who were eliminated. These might be harder ingredients or flavors to get right, or ones that are less popular with the judges, and so we might caution future contestants about using these. ⚠️
# 
# Use your `most_common` function to produce a DataFrame with the ten most common meaningful words, among Signature Challenge recipes in which the baker was eliminated that week. Name that DataFrame `common_out`.
# 
# *Hint*: Bakers who are eliminated have a `'result'` of "OUT" or "Runner-up."

# In[42]:


common_out = most_common(meaningful[(meaningful.get('result')=='OUT') | (meaningful.get('result')=='Runner-up')].get('meaningful_words').sum())
common_out


# In[43]:


grader.check("q2_8")


# **Question 2.9.** Now let's look at the meaningful words that were most popular in Signature Challenge recipe titles among bakers who didn't get eliminated. What special ingredients are they using? These might be more well-loved flavors and ingredients, and we might consider them safe choices for baking foods that the judges will enjoy! 😋
# 
# Use your `most_common` function to produce a DataFrame with the ten most common meaningful words, among Signature Challenge recipes in which the baker stayed in the competition that week. Name that DataFrame `common_in`.
# 
# *Hint*: Bakers who stay in the competition have a `'result'` of "IN" or "STAR BAKER" or "WINNER".

# In[44]:


common_in = most_common(meaningful[(meaningful.get('result')=='IN') | (meaningful.get('result')=='STAR BAKER') | (meaningful.get('result')=='WINNER')].get('meaningful_words').sum())

common_in


# In[45]:


grader.check("q2_9")


# You'll notice that some ingredients are common among people who get eliminated and people who stayed, and that's just because they're common recipe ingredients generally. It's more interesting to look at the words that appear in only one of `common_out` and `common_in`. Would you rather have a walnut mushroom pie or a raspberry rhubarb pudding?

# <a id='section3'></a>
# ## Section 3: Gender Balance 👩⚖️🧑🏼
# After watching a couple of episodes, you start to wonder if more female bakers than male bakers have been selected to participate in the Great British Bake Off. Let's check if this is the case.

# **Question 3.1.** Using the `baker_weeks` DataFrame, first count the total number of bakers in the first 11 seasons of the show and assign your answer to the variable `baker_count`.
# 
# Then, compute the proportion of female bakers and the proportion of male bakers in the first 11 seasons of the show. Assign your answers to the variables `observed_female_prop` and  `observed_male_prop`. 
# 
# Notice that `baker_weeks` has a row for each baker for each week, so we can't directly calculate proportions from the `'Gender'` column of that DataFrame.
# 
# *Note*: While several bakers with the same name appeared on the show (there were three Peters and three Kates!) there were never two bakers with the same name appearing in the same season.

# In[46]:


female_count=np.array([])
for i in np.arange(1, 12):
    count = int(baker_weeks[baker_weeks.get('Season')== i][baker_weeks[baker_weeks.get(
        'Season')==i].get('Gender')=='F'].shape[0]/(baker_weeks[baker_weeks.get(
        'Season')==i].get('Week Number').max()))
    female_count = np.append(female_count, count)
female_count.sum()


# In[47]:


male_count=np.array([])
for i in np.arange(1, 12):
    count = int(baker_weeks[baker_weeks.get('Season')== i][baker_weeks[baker_weeks.get(
        'Season')==i].get('Gender')=='M'].shape[0]/(baker_weeks[baker_weeks.get(
        'Season')==i].get('Week Number').max()))
    male_count = np.append(male_count, count)
male_count.sum()


# In[48]:


baker_count = np.array([])
for i in np.arange(1, 12):
    count = baker_weeks[baker_weeks.get('Season')== i].groupby('Baker').count().reset_index().get('Baker').shape[0]
    baker_count = np.append(baker_count, count)
baker_count


# In[49]:


baker_weeks[baker_weeks.get('Season')==1].get('Week Number').max()
baker_weeks[baker_weeks.get('Season')==1].groupby('Baker').count().reset_index().get('Baker').shape[0]


# In[50]:


baker_count = int(baker_count.sum())
observed_female_prop = female_count.sum()/baker_count
observed_male_prop = male_count.sum()/baker_count


print("Female Proportions: " + str(observed_female_prop))
print("Male Proprotions: " + str(observed_male_prop))
print("Number of Bakers: " + str(baker_count))


# In[51]:


grader.check("q3_1")


# You recognize that `observed_female_prop` and `observed_male_prop` are similar but they're not exactly the same. Is this just random chance at play, or are female bakers actually more likely to be on the show? Let's do a hypothesis test with the following hypotheses:
# 
# - **Null Hypothesis**: Bakers on the show are drawn randomly from a population that’s 50% female and 50% male. 
# - **Alternative Hypothesis**: Bakers on the show are not drawn randomly from a population that’s 50% female and 50% male.
# 
# Run the cell below to define a variable `null_distribution` that shows the proportion of each gender according to our model.

# In[52]:


null_distribution = np.array([0.5, 0.5])
null_distribution


# **Question 3.2.** To perform our hypothesis test, we will simulate drawing a random sample of size `baker_count` from the null distribution, and then compute a test statistic on each simulated sample. We must first choose a reasonable test statistic that will help us determine whether to reject the null hypothesis.
# 
# From the options below, find **all** valid test statistics that we could use for this hypothesis test. Save the numbers of your choices in a `list` called `gender_test_statistics`. Valid test statistics are ones that would allow us to distinguish between the null and alternative hypotheses. 
# 
# *Hint*: To determine whether a test statistic is valid, think about which values of the statistic (high, low, moderate) would make you lean towards the null and which would make you lean towards the alternative.
# 
# 1. The absolute difference between the proportion of female bakers and 0.5.
# 2. The absolute difference between the number of male bakers and the number of female bakers. 
# 3. The absolute difference between the number of female bakers and one half of `baker_count`.
# 4. Three times the absolute difference between the proportion of male bakers and 0.5.
# 5. The total variation distance between the gender distribution of bakers and the null distribution.

# In[53]:


gender_test_statistics = [1, 2, 3, 4, 5]
gender_test_statistics


# In[54]:


grader.check("q3_2")


# **Question 3.3.** For this hypothesis test, we'll use as our test statistic the absolute difference between the observed proportion of female bakers and 0.5, the expected proportion under the assumptions of the null hypothesis. Set the variable `observed_gender_stat` to the observed value of this statistic.

# In[55]:


observed_gender_stat = abs(observed_female_prop - 0.5)
observed_gender_stat


# In[56]:


grader.check("q3_3")


# **Question 3.4.** Write a simulation that runs 10,000 times, each time drawing a random sample of size `baker_count` from the null distribution. Keep track of the simulated test statistics in the `gender_stats` array. 

# In[57]:


np.random.choice([1, 0], baker_count, p=[0.5, 0.5]).sum()/baker_count


# In[58]:


gender_stats = np.array([])
for i in np.arange(10000):
    sample = abs(np.random.choice([1, 0], baker_count, p=[0.5, 0.5]).sum()/baker_count - 0.5)
    gender_stats = np.append(gender_stats, sample)
    
# Visualize with a histogram
bpd.DataFrame().assign(gender_stats=gender_stats).plot(kind='hist', density=True, ec='w', figsize=(10, 5));
plt.axvline(x=observed_gender_stat, color='black', linewidth=4, label='observed_gender_stat')
plt.legend();


# In[59]:


grader.check("q3_4")


# **Question 3.5.** Recall that your null hypothesis was that the bakers on the show are drawn randomly from a population that’s 50% female and 50% male. Compute the p-value for this hypothesis test, and save the result to `gender_p_value`.

# In[60]:


gender_p_value = np.count_nonzero(gender_stats <= observed_gender_stat)/10000
gender_p_value


# In[61]:


grader.check("q3_5")


# You should find that the p-value is nowhere near the standard cutoff of 0.05 for statistical significance. So in this case, we fail to reject the null. 
# 
# It's important to note that even though we fail to reject the null, we’re not saying that bakers *were* necessarily drawn randomly from a population that’s 50% female and 50% male. In fact, nothing is random about how people get to be on the show. 
# 
# There are a lot of rules about who can apply to be on the show, and applicants are thoroughly vetted through an extensive [application process](https://gbbo.take-part.co.uk/info/rules) that involves an interview and a background check to ensure that none of the bakers have any sort of professional training or are friends or relatives of the judges. Simply put, bakers on the show are not selected via a purely random process.
# 
# When we say we fail to reject the null, this means that the bakers *could have* been drawn from a model that's 50% female and 50% male, but it doesn't mean they *were*.
# 

# **Question 3.6.** Conceptually, how would you expect the statistics in `gender_stats` to change if `baker_count` were a much larger value, like if the show included hundreds of bakers every season? What effect would that have on the result of the hypothesis test?
# 
# From the options below, save the number of your choice in the variable`gender_stats_change`.
# 
# 1. The values in `gender_stats` would be **smaller**. We'd be **less** likely to reject the null hypothesis if `observed_gender_stat` remained the same.
# 2. The values in `gender_stats` would be **smaller**. We'd be **more** likely to reject the null hypothesis if `observed_gender_stat` remained the same.
# 3. The values in `gender_stats` would be **about the same**. We'd be **equally** likely to reject the null hypothesis if `observed_gender_stat` remained the same.
# 4. The values in `gender_stats` would be **larger**. We'd be **less** likely to reject the null hypothesis if `observed_gender_stat` remained the same.
# 5. The values in `gender_stats` would be **larger**. We'd be **more** likely to reject the null hypothesis if `observed_gender_stat` remained the same.
# 

# In[62]:


gender_stats_change = 4
gender_stats_change


# In[63]:


grader.check("q3_6")


# <a id='section4'></a>
# ## Section 4. Well-Deserved? 🥇

# In this section, we will use permutation testing to decide if different groups of bakers have similar technical abilities, as measured by their rankings in the Technical Challenges. Let's start by looking at our `baker_weeks` DataFrame which has a row for each baker for each week of the show, including for the remainder of the season after they've been eliminated. Let's start by only keeping the data for the bakers that actually competed in each week's episode. Since ten bakers participated in the first episode of Season 1, we'll look at the first ten rows of the resulting `competed` DataFrame. 

# In[64]:


competed = baker_weeks[baker_weeks.get('Competed') == 1]
competed.take(np.arange(10))


# In the `'Technical Rank'` column, contestants are given a ranking for how well they performed in the Technical Challenge, with 1 being the best. Notice in the first ten rows of `competed` shown above, some of the middle rankings are missing. In this episode, the judges didn't reveal everyone's rank and instead just pointed out the top three and bottom three contestants. For reasons like this, our dataset has just a few missing values, which we will ignore for this section. 
# 
# If we want to get a sense of how skilled a baker is, the technical rank is helpful, but needs to be taken in the context of the number of contestants still in the competition. For example, ranking 3rd place in the first week is a lot more impressive than ranking 3rd place in the final week, when there are just three bakers remaining. To address this problem, we'll convert these rankings into *percentiles* to measure skill relative to the number of contestants remaining. 
# 
# For example, if there are four contestants remaining, a technical ranking of:
# - 4 corresponds to the 25th percentile
# - 3 corresponds to the 50th percentile
# - 2 corresponds to the 75th percentile
# - 1 corresponds to the 100th percentile
# 
# **Question 4.1.** Create a DataFrame called `perc` with the same data as `competed`, plus a new column called `'Contestants'`  that contains the number of contestants that competed each week. For example, since the first ten rows of `competed` all correspond to the first week of the first season, in which there were 10 bakers, the first ten entries of the `'Contestants'` column should be 10.  
# 
# We've provided the code to use the `'Contestants'` column and the `'Technical Rank'` column to calculate the percentiles, which we've added in a column called `'Percentile'`.
# 
# *Hint*: Start by counting the number of bakers in each episode.

# In[65]:


total_num=np.array([])
for i in np.arange(competed.groupby('Episode').count().shape[0]):
    number = competed.groupby('Episode').count().iloc[i].get('Season')
    total_num = np.append(total_num, [number] * number)
total_num


# In[66]:


# Your task is to add the Contestants column.
perc = competed.assign(Contestants = total_num)
# We've added the Percentile column for you.
perc = perc.assign(Percentile = np.round((1 - (perc.get('Technical Rank') - 1) / perc.get('Contestants')) * 100, 1))
perc


# In[67]:


grader.check("q4_1")


# Now we are ready to compare two groups of bakers to see if they are comparably skilled. Let's start with comparing the winners to the non-winners. We'll conduct a permutation test with the following hypotheses.
# 
# - **Null Hypothesis** : The `'Percentile'` data for winners comes from the same distribution as the `'Percentile'` data for non-winners. In other words, winners and non-winners perform equally well in Technical Challenges.
# - **Alternate Hypothesis** : The `'Percentile'` data for winners and the `'Percentile'` data for non-winners come from different distributions. Winners perform better in Technical Challenges than non-winners.
# 
# As usual, we'll use the difference in group means as our test statistic. Here, we'll compute that as the mean for the winners minus the mean for the non-winners.

# **Question 4.2.** What is the observed value of the test statistic? Save your answer as `observed`.

# In[68]:


mean = perc.groupby('Winner').mean()

observed = mean.get('Percentile').loc[1] - mean.get('Percentile').loc[0]
observed


# In[69]:


grader.check("q4_2")


# **Question 4.3.** Create 1000 simulated values of the test statistic under the assumptions of the null hypothesis, and save your simulated test statistics in the array `simulated_stats`.  Then create an appropriate visualization showing the distribution of the values in `simulated_stats` array. It may be helpful to also plot the observed value of the test statistic on the same graph. 

# In[70]:


perc.assign(Winner = np.random.permutation(perc.get('Winner')))


# In[71]:


# Run your simulation here.
def difference(data):
    mean = data.groupby('Winner').mean()
    return mean.get('Percentile').loc[1] - mean.get('Percentile').loc[0]

simulated_stats = np.array([])
for i in np.arange(1000):
    shuffled = perc.assign(Winner = np.random.permutation(perc.get('Winner')))
    stat = difference(shuffled)
    simulated_stats = np.append(simulated_stats, stat)
    
simulated_stats = simulated_stats

# Plot your visualization here.
bpd.DataFrame().assign(difference=simulated_stats).plot(kind='hist', density=True, ec='w')
plt.axvline(observed, color='black', linewidth=4, label='observed difference in means')
plt.legend();


# In[72]:


grader.check("q4_3")


# **Question 4.4.** The winning contestants claim that they are more technically skilled than the other contestants. Based on your permutation test, using a p-value cutoff of 0.01, do you think this claim is likely accurate? Set `winners_claim` to True or False.

# In[73]:


winners_claim = True
winners_claim


# In[74]:


grader.check("q4_4")


# 
# 
# Now, we'll do a similar permutation test, but this time comparing the Technical Challenge `'Percentile'` of contestants who received a coveted handshake 🤝 from Paul Hollywood at least once to those who never did.
# 
# 
# ![](https://media.giphy.com/media/3o7TKRoSl2BrFuK75u/giphy.gif)

# **Question 4.5.** Create a new DataFrame called `earned`, indexed by `'Season'` and `'Baker'`, that has a row for each baker who received a handshake 🤝 **at any point** in the season, and a single column called `'Handshake'` containing all ones. 
# 
# Similarly, create a DataFrame called `not_earned`, indexed by `'Season'` and `'Baker'`, that has a row for each baker who **never** received a handshake 🤝, and a single column called `'Handshake'` containing all zeros. 
# 
# *Note*: There are several bakers by the same name, but never in the same season.
# 
# *Hint*: Check out the functions [`np.ones`](https://numpy.org/doc/stable/reference/generated/numpy.ones.html) and [`np.zeros`](https://numpy.org/doc/stable/reference/generated/numpy.zeros.html).

# In[75]:


handshake = perc.groupby(['Season', 'Baker']).sum()
shake = handshake[(handshake.get('Signature Handshake') !=0) | (handshake.get('Showstopper Handshake') !=0)]


# In[76]:


earned = shake.assign(Handshake=np.ones(shake.shape[0])).get(['Handshake'])
earned


# In[77]:


grader.check("q4_5_a")


# In[78]:


noshake = handshake[(handshake.get('Signature Handshake')==0) & (handshake.get('Showstopper Handshake')==0)]
noshake


# In[79]:


not_earned = noshake.assign(Handshake=np.zeros(noshake.shape[0])).get(['Handshake'])
not_earned


# In[80]:


grader.check("q4_5_b")


# Our `earned` and `not_earned` DataFrames contain the information we need to determine who falls into which group for our permutation test, but we need to combine this data with the Technical Challenge percentiles in `perc`. 
# 
# The first step is to combine the rows of `earned` and with those of `not_earned`. We'll do this using the `babypandas` DataFrame method `.append`, which is similar to the familiar `np.append`, but for DataFrames instead of arrays. The cell below puts the rows of `not_earned` onto the end of `earned` and saves the result as `shakes`. Don't worry if you see a warning; ignore it.

# In[81]:


shakes = earned.append(not_earned)
shakes


# Now we need to merge `shakes` with `perc` to get the handshake 🤝 data and the percentile data in one DataFrame. Since there are multiple bakers that share a name, we need to merge by *both* `'Season'` and `'Baker'`, which we can do by merging on a `list` containing both column names. Run the cell below to complete the merge and save the result as `perc_shakes`.

# In[82]:


perc_shakes = perc.merge(shakes, left_on=['Season', 'Baker'], right_index=True)
perc_shakes


# **Question 4.6.** Now perform a permutation test, mimicking the procedure of Question 4.3, to help you analyze the following claim. 
# 
# The contestants who have gotten a handshake 🤝 claim that they are more technically skilled than the other contestants. Based on your permutation test, using a p-value of cutoff of 0.01, do you think this claim is likely accurate? Set `handshake_claim` to True or False.

# In[83]:


def difference_shake(data):
    mean = data.groupby('Handshake').mean()
    return mean.get('Percentile').loc[1] - mean.get('Percentile').loc[0]

simulated_shakes = np.array([])
for i in np.arange(1000):
    shuffled = perc_shakes.assign(Winner = np.random.permutation(perc_shakes.get('Handshake')))
    stat = difference(shuffled)
    simulated_shakes = np.append(simulated_shakes, stat)
    
simulated_shakes
observed_shakes = difference_shake(perc_shakes)

# Plot your visualization here.
bpd.DataFrame().assign(difference=simulated_shakes).plot(kind='hist', density=True, ec='w')
plt.axvline(observed_shakes, color='black', linewidth=4, label='observed difference in means')
plt.legend();


# In[84]:


np.count_nonzero(simulated_shakes >= observed_shakes) / 1000


# In[85]:


handshake_claim = False
handshake_claim


# In[86]:


grader.check("q4_6")


# <a id='section5'></a>
# ## Section 5: Devilishly Difficult Challenges 😈
# 
# Contestants on the Great British Bake Off sometimes groan when the hosts announce that the upcoming Technical Challenge was chosen by judge Paul Hollywood. Paul has a reputation for posing exceptionally difficult challenges and most bakers believe that his recipes are much harder than those of his co-judges, Mary Berry and Prue Leith. We want to examine whether this theory is justified by the data. 
# 
# The `technical_challenge_recipes` DataFrame contains 83 Technical Challenge recipes from seasons 1 through 9. Each Technical Challenge is posed by one particular judge, and comes from their personal collection of recipes. In the first nine seasons, Mary posed 32 Technical Challenges, Paul posed 41, and Prue posed 10. The `technical_challenge_recipes` DataFrame includes a `'DifficultyScore'` for each recipe, with more challenging recipes having higher scores.

# **Question 5.1.** Create a DataFrame `mean_by_judge` with the judge's name as the index and just one column called `'mean_difficulty_score'` that contains the mean difficulty score for each judge's Technical Challenges. 

# In[87]:


mean_by_judge = technical_challenge_recipes.assign(mean_difficulty_score = technical_challenge_recipes.get('DifficultyScore')).groupby('Whose').mean().get(['mean_difficulty_score'])
mean_by_judge


# In[88]:


grader.check("q5_1")


# If you solved this problem correctly, you will notice that Mary and Paul both have an average difficulty of less than 5, whereas Prue has a mean difficulty greater than 7. Does it mean that Prue, in fact, is the devil when it comes to Technical Challenges? In other words, does Prue have a much more challenging recipe collection than the other judges? Or is this all by chance?
# 
# Suppose each judge has an extensive personal recipe collection with recipes of varying difficulty, and the Technical Challenges for each episode are drawn randomly from this collection. We want to estimate the average difficulty of all recipes in each judge's collection. Unfortunately, we don't have access to a judge's entire recipe collection, we only have access to the sample of recipes they've used for Technical Challenges in the Great British Bake Off. Thus, we will tackle this problem using **bootstrapping**. 
# 
# **Question 5.2.** Below, write a function called `simulate_estimates`. It should take 3 arguments:
# - `sample_df`: A DataFrame with a row for each element of the original sample. In this case, it will consist of Technical Challenges posed by a particular judge.
# - `variable`: The column name of the relevant variable, whose mean we want to estimate.
# - `repetitions`: The number of repetitions to perform (i.e., the number of resamples to create).
# 
# It should take `repetitions` resamples with replacement from the given DataFrame. For each of those resamples, it should compute the mean of the relevant variable for that resample. Then it should return an array containing the value of those means for each resample.

# In[89]:


technical_challenge_recipes


# In[90]:


def simulate_estimates(sample_df, variable, repetitions):
    '''Returns an array of length repetitions, containing bootstrapped means of the variable from sample_df. '''
    means = np.array([])
    
    for i in np.arange(repetitions):
        sample = sample_df.sample(sample_df.shape[0], replace=True)
        resample = sample.get(variable).mean()
        means = np.append(means, resample)
    return means


# In[91]:


grader.check("q5_2")


# **Question 5.3.** Use your function `simulate_estimates` to estimate the mean difficulty score of three judges' recipe collections. Use `repetitions = 5000`, and save your arrays of bootstrapped means for each judge in the variables `mary_boot_means`, `paul_boot_means`, and `prue_boot_means`.  
# 
# Then, plot the distributions of all three of these arrays in one overlaid histogram. Use `bins=np.arange(2,10,0.2)` and set `alpha=0.5` (this changes the opacity to see the distribution more clearly).
# 
# *Hint:* Create a DataFrame with one column for each judge's bootstrapped means, and use this to plot the histogram.

# In[92]:


mary_boot_means = simulate_estimates(technical_challenge_recipes[technical_challenge_recipes.get('Whose')=='Mary'], 
                                    'DifficultyScore', 5000)
paul_boot_means = simulate_estimates(technical_challenge_recipes[technical_challenge_recipes.get('Whose')=='Paul'], 
                                    'DifficultyScore', 5000)
prue_boot_means = simulate_estimates(technical_challenge_recipes[technical_challenge_recipes.get('Whose')=='Prue'], 
                                    'DifficultyScore', 5000)

# Plot your overlaid histogram here.
plt.hist([mary_boot_means, paul_boot_means, prue_boot_means], density=True,alpha=0.5, ec='w', color=['purple', 'green', 'blue'], label=['Mary', 'Paul', 'Prue'])
    
plt.legend()


# In[93]:


grader.check("q5_3")


# **Question 5.4.** Now we want to calculate three 95% confidence intervals for the mean difficulty score of recipes from each of the three judges. To do this, create a function `confidence_interval_95`, which takes in an array of bootstrapped statistics `boot_stats` and returns a list of length two, containing the left endpoint and the right endpoint of the 95% confidence interval. 

# In[94]:


def confidence_interval_95(boot_stats):
    '''Returns a list of the endpoints of a 95% confidence interval based on boot_stats.'''
    return [np.percentile(boot_stats, 2.5), np.percentile(boot_stats, 97.5)]

print("Mary 95% CI:", confidence_interval_95(mary_boot_means))
print("Paul 95% CI:", confidence_interval_95(paul_boot_means))
print("Prue 95% CI:", confidence_interval_95(prue_boot_means))


# In[95]:


grader.check("q5_4")


# **Question 5.5.** Based on your results, which of the following statements are correct? Assign `true_statements` to a list containing **all** the true statements. 
# 
# 1. Paul's recipes are generally harder than those of his co-judges.
# 2. Prue's recipes are generally harder than those of her co-judges.
# 3. Prue and Mary's confidence intervals overlap.
# 4. Mary and Paul's confidence intervals overlap.
# 5. Mary's confidence interval is wider than Paul's.
# 6. Prue's confidence interval is wider than Mary's.

# In[96]:


true_statements = [2, 4, 5, 6]
true_statements


# In[97]:


grader.check("q5_5")


# **Question 5.6.** If your calculation is correct, you will see that Prue's confidence interval is almost twice as wide as the other two judges' confidence intervals. Why is Prue's confidence interval wider? 
# 
# Assign either 1, 2, or 3 to the variable `why_wider` below.
# 
# 1. She has more challenging recipes in her collection.
# 2. She has posed fewer Technical Challenges.
# 3. She has posed Technical Challenges with a wider range of difficulty levels.

# In[98]:


why_wider = 3
why_wider


# In[99]:


grader.check("q5_6")


# From what we've done so far, it's clear that Prue's recipes have a very different difficulty level than the recipes of the other two judges. Now let's address a different question: how does the average difficulty of Paul's recipes compare to the average difficulty of Mary's recipes? 
# 
# **Question 5.7.** Create a DataFrame called `mary_only` containing only the recipes in our original `technical_challenge_recipes` sample from Mary's collection. Then, create another DataFrame called `paul_only` containing only the recipes in our original sample from Paul's collection. Then, set `observed_diff_mean` to the difference in mean difficulty score between Mary's recipes and Paul's recipes in our sample (subtract in the order Mary minus Paul).

# In[100]:


mary_only = technical_challenge_recipes[technical_challenge_recipes.get('Whose')=='Mary']
paul_only = technical_challenge_recipes[technical_challenge_recipes.get('Whose')=='Paul']
observed_diff_mean = mary_only.get('DifficultyScore').mean() - paul_only.get('DifficultyScore').mean()
observed_diff_mean


# In[101]:


grader.check("q5_7")


# So there is definitely a difference in mean difficulty scores between Mary's and Paul's Technical Challenge recipes, within our sample of recipes that have appeared as Technical Challenges in the show. But does this reflect a difference in mean recipe difficulty scores in the population (the judges' recipe collections), or was it by chance that our sample's difficulty displayed this difference? Let's do a hypothesis test to find out. We'll state our hypotheses as follows:
# 
# - **Null Hypothesis:** The mean difficulty of Mary's recipe collection equals the mean difficulty of Paul's recipe collection. Equivalently, the difference in the mean difficulty for Mary's and Paul's recipes equals 0.
# - **Alternative Hypothesis:** The mean difficulty of Mary's recipe collection does not equal the mean difficulty of Paul's recipe collection. Equivalently, the difference in the mean difficulty for Mary's and Paul's recipe does not equal 0.
# 
# Since we were able to set up our hypothesis test as a question of whether our population parameter – the difference in mean difficulty scores for Mary's and Paul's recipe collections – is equal to a certain value, we can **test our hypotheses by constructing a confidence interval for the parameter**. This is the method we used in Lecture 19 to test whether the median salary of Fire-Rescue Department workers was the same as the median salary of all San Diego city employees. We also did a similar example in Homework 5 Question 3 (Live Crystal Scoops 🔮). For a refresher on this method, you can read more about conducting a hypothesis test with a confidence interval in [CIT 13.4](https://inferentialthinking.com/chapters/13/4/Using_Confidence_Intervals.html#).

# **Question 5.8.** Compute 1000 bootstrapped estimates for the difference in the mean difficulty for Mary's recipes and Paul's recipes (subtract in the order Mary minus Paul). Store your 1000 estimates in the `difference_means` array.
# 
# You should generate your resamples of Mary's recipes by sampling from `mary_only`, and similarly for Paul, by sampling from `paul_only`. You should not use `technical_challenge_recipes` at all.

# In[102]:


np.random.seed(57) # Don't change this. This is for the autograder.

difference_means = simulate_estimates(mary_only, 'DifficultyScore', 1000) - simulate_estimates(paul_only, 'DifficultyScore', 1000)

# Just display the first ten differences.
difference_means[:10]


# In[103]:


grader.check("q5_8")


# Let's visualize your estimates:

# In[104]:


(bpd.DataFrame().assign(DifferenceMeans = difference_means)
 .plot(kind='hist', density=True, ec='w', figsize=(10, 5)));


# **Question 5.9.** Use the function `confidence_interval_95` you created before to compute a 95% confidence interval for the difference in the mean difficulty of Mary's and Paul's recipes (as before, Mary's minus Paul's). Assign to `mary_paul_difference_CI` a list containing the endpoints of this confidence interval.

# In[105]:


mary_paul_difference_CI = confidence_interval_95(difference_means)
mary_paul_difference_CI


# In[106]:


grader.check("q5_9")


# Recall the hypotheses we were testing:
# - **Null Hypothesis:** The mean difficulty of Mary's recipe collection equals the mean difficulty of Paul's recipe collection. Equivalently, the difference in the mean difficulty for Mary's and Paul's recipes equals 0.
# - **Alternative Hypothesis:** The mean difficulty of Mary's recipe collection does not equal the mean difficulty of Paul's recipe collection. Equivalently, the difference in the mean difficulty for Mary's and Paul's recipe does not equal 0.
# 
# **Question 5.10.** Based on the confidence interval you've created, would you reject the null hypothesis at the 0.05 significance level? Set `reject_null_mary_paul` to True if you would reject the null hypothesis, and False if you would not.

# In[107]:


reject_null_mary_paul = False
reject_null_mary_paul


# In[108]:


grader.check("q5_10")


# We have now uncovered some interesting facts about the difficulty levels of the different judges' recipe collections. However, we also want to know whether the judges' recipe collections have other differences. For example, do certain judges have recipes with more ingredients, more components, or longer instructions?
# 
# To do this, we want to generalize our simulation code so that we can create a confidence interval for any variable.
# 
# **Question 5.11.** Create a function called `bootstrap_estimation`, which takes in 4 inputs:
# - `sample_df`, A DataFrame with a row for each element of the original sample (Technical Challenge recipes posed by one or more judges)
# - `judges`, a list of judge's names we want to compare, which can include any number of "Paul", "Mary", and "Prue"
# - `variable`, the column name of the relevant variable, whose mean we want to estimate 
# - `repetitions`, the number of repetitions to perform (i.e., the number of resamples to create)
# 
# The function should adhere to these specifications:
# 1. The function should generate an overlaid histogram showing each of the specified judges' simulated means of the given variable. Make sure to give your histogram a descriptive title and to use appropriate labels. Use `bins=20`, `alpha=0.5`, and `figsize=(10,5)`.
# 2. The function should print a statement with the 95% confidence interval for the mean value of the given variable for each of the specified judges. See the example below for the type of statement to print, but the exact formatting is up to you.
# 3. The function should return nothing.
# 
# *Hints:* 
# - This is designed to be a challenging question, but remember that you can use any of the functions you've already created. 
# - Our solution uses an `if`-statement to assign columns named `'Mary_mean_estimate'`, `'Paul_mean_estimate'`, or `'Prue_mean_estimate'`.
# 
# Here is an example output that shows a comparison of estimates for the mean number of dirty dishes produced by recipes in each of the three judges' collections.
# <img src="images/desired_output.jpg" width = 700>

# In[109]:


def bootstrap_estimation(sample_df, judges, variable, repetitions):
    '''Generates a histogram and for each judge, a confidence interval for the mean value of the variable from sample_df.'''
    if 'Mary'in judges:
        #Mary = simulate_estimates(sample_df[sample_df.get('Whose')=='Mary'], variable, repetitions)
        plt.hist([simulate_estimates(sample_df[sample_df.get('Whose')=='Mary'], variable, repetitions)], density=True,alpha=0.5, ec='w', color=['red'], label=['Mary_mean_estimate'])
        print('Mary\'s 95% CI for '+ variable + ':', confidence_interval_95(simulate_estimates(sample_df[sample_df.get('Whose')=='Mary'], variable, repetitions)))
    if 'Paul'in judges:
        #Paul = simulate_estimates(sample_df[sample_df.get('Whose')=='Paul'], variable, repetitions)
        plt.hist([simulate_estimates(sample_df[sample_df.get('Whose')=='Paul'], variable, repetitions)], density=True,alpha=0.5, ec='w', color=['blue'], label=['Paul_mean_estimate'])
        print('Paul\'s 95% CI for '+ variable + ':', confidence_interval_95(simulate_estimates(sample_df[sample_df.get('Whose')=='Paul'], variable, repetitions)))
    if 'Prue'in judges:
        #Prue = simulate_estimates(sample_df[sample_df.get('Whose')=='Prue'], variable, repetitions)
        plt.hist([simulate_estimates(sample_df[sample_df.get('Whose')=='Prue'], variable, repetitions)], density=True,alpha=0.5, ec='w', color=['purple'], label=['Prue_mean_estimate'])
        print('Pure\'s 95% CI for '+ variable + ':', confidence_interval_95(simulate_estimates(sample_df[sample_df.get('Whose')=='Prue'], variable, repetitions)))
        
    #if 'Mary' and 'Paul' and 'Prue' in judges:  
     #   plt.hist([Mary, Paul, Prue], density=True,alpha=0.5, ec='w', color=['purple', 'green', 'blue'], label=['Mary', 'Paul', 'Prue'])
    #if 'Paul' and 'Prue'in judges:
     #   plt.hist([Paul, Prue], density=True,alpha=0.5, ec='w', color=['purple', 'green', 'blue'], label=['Mary', 'Paul', 'Prue'])
    plt.legend()
    plt.ylabel('Frequency')
    plt.title(variable)
    return 
# Try to replicate the graph shown in the example.
bootstrap_estimation(technical_challenge_recipes, ['Mary','Paul', 'Prue'], 'Dishes', 1000)


# **Question 5.12.** Using the `bootstrap_estimation` function you just wrote, create histograms and confidence intervals that would help you answer each of the following questions. Use `repetitions=1000`. 
# 
# 1. Whose recipes have more sentences on average, Mary's or Paul's?
# 2. Of the three judges, how do their average counts of recipe ingredients compare?
# 
# For each part, all you need to do is make one call to `bootstrap_estimation` with the appropriate inputs.

# <!-- BEGIN QUESTION -->
# 
# <!--
# BEGIN QUESTION
# name: q5_12
# manual: True
# points: 2
# 
# -->

# In[110]:


technical_challenge_recipes


# In[111]:


# For question 1, make your function call here.

bootstrap_estimation(technical_challenge_recipes, ['Mary','Paul'], 'RecipeSentences', 1000)


# <!-- END QUESTION -->

# In[112]:


# For question 2, make your function call here.

bootstrap_estimation(technical_challenge_recipes, ['Mary','Paul', 'Prue'], 'IngredCount', 1000)


# Feel free to explore other questions with other variables as you wish. But at this point, the devilish judge should be clear!  😈

# <a id='section6'></a>
# ## Section 6: Piece of Cake? 🍰
# 
# In this section of the project, we'll focus on probability.

# **Question 6.1.** You wonder if it takes a lot of skill to win the bake off. If we randomly select a (series) winner from the first ten seasons of the show, what is the probability that they came first in one of the Technical Challenges? Use the `bakers` DataFrame to calculate this probability and assign your answer to the variable `p_tech_given_win`.
# 
# 

# In[113]:


series_win = bakers[bakers.get('series_winner')!=0]
series_win[series_win.get('technical_winner')!=0].shape[0]


# In[114]:


p_tech_given_win = series_win[series_win.get('technical_winner')!=0].shape[0]/series_win.shape[0]
p_tech_given_win


# In[115]:


grader.check("q6_1")


# **Question 6.2.** You wonder how frequently winners are recognized with the special designation of Star Baker ⭐. If we randomly select a winner from the first ten seasons of the show, what is the probability that they won Star Baker ⭐ at some point? Assign your answer to the variable `p_star_given_win`.

# In[116]:


series_win[series_win.get('star_baker')!=0]


# In[117]:


p_star_given_win = series_win[series_win.get('star_baker')!=0].shape[0]/series_win.shape[0]
p_star_given_win


# In[118]:


grader.check("q6_2")


# Notice that in both of the previous questions, you calculated a conditional probability. Among bakers who satisfy one condition (winning), what is the probability they satisfy another condition (placing first in a technical, or earning Star Baker ⭐). Let's generalize the code for these calculations so that we can more easily compute conditional probabilities with other conditions.
# 
# **Question 6.3.** Your job is to implement the function `conditional_probability`. It has two arguments, `find` and `given`, both of which are lists. Let's walk through how it works, using an example – suppose we want to use it to compute the probability that a randomly selected contestant from `bakers` was a Star Baker ⭐, given that they won (the same probability that you computed in the previous question.)
# 
# - `find` is a list of two elements:
#     - The first element in `find` is the column in `bakers` that contains the event that we are trying to find the probability of. This can be any column in `baker`; in our example, this is `'star_baker'`. 
#     - The second element in `find` is the value in the aforementioned column that we're trying to find; in our example, this is `1`.
# - `given` is a list of two elements:
#     - The first element in `given` is the column in `bakers` that contains the event that we are given to be true. This can also be any column in `baker`; in our example, this is `'series_winner'`. 
#     - The second element in `given` is the value in the aforementioned column; in our example, this is `1`.
# 
# Putting this all together, this means that `conditional_probability(['star_baker', 1], ['series_winner', 1])` should evaluate to your answer from the previous part (but the `conditional_probability` function should work for any example, not just this one).
# 

# In[119]:


series_win = bakers[bakers.get('series_winner')!=0]
series_win[series_win.get('technical_winner')!=0].shape[0]
series_win[series_win.get('star_baker')!=0]


# In[120]:


def conditional_probability(find, given):
    '''Returns the conditional probability of an event given a known condition.'''
    series = bakers[bakers.get(given[0])==given[1]]
    return series[series.get(find[0])==find[1]].shape[0]/series.shape[0]
    
# This should evalaute to your answer to Question 6.2
conditional_probability(['star_baker', 1], ['series_winner', 1])


# In[121]:


grader.check("q6_3")


# ### **Question 6.4.** Now use the function `conditional_probability` to calculate the following probabilities:
# - `p_female_given_young`: The probability that a randomly chosen young contestant is female. 👧🏽
# - `p_female_given_elderly`: The probability that a randomly chosen elderly contestant is female. 👵

# In[122]:


p_female_given_young = conditional_probability(['gender','F'],['age_category','Young'])
p_female_given_elderly = conditional_probability(['gender','F'],['age_category','Elderly'])

# Don't change the code below.
print(f'''P(female given young) = {p_female_given_young}
P(female given elderly) = {p_female_given_elderly}''')


# In[123]:


grader.check("q6_4")


# **Question 6.5.** Suppose the producers of the show want to do a special episode bringing back past contestants, as they often do for the holidays 🎄🕎. They decide to choose one contestant at random from each of the first ten seasons. What is the probability that there is at least one winner selected? Assign your answer to `p_include_winner`.
# 
# *Hint:* The function `np.prod` might be helpful. Here is a [link to its documentation](https://numpy.org/doc/stable/reference/generated/numpy.prod.html).

# In[124]:


bakers.groupby('series').count()
prob = np.array([])
for i in np.arange(10):
    one = (bakers.groupby('series').count().get('baker').iloc[i]-1)/bakers.groupby('series').count().get('baker').iloc[i]
    prob = np.append(prob, one)
prob


# In[125]:


p_include_winner = 1- np.prod(prob)
p_include_winner


# In[126]:


grader.check("q6_5")


# **Question 6.6.** You have dreams 💭 of being on the bake off yourself, and to practice, you decide to bake 10 Technical Challenge recipes, chosen at random **with replacement** from the `technical_challenge_recipes` DataFrame. What is the probability that all 10 of them have a `'DifficultyScore'` greater than 5? Assign your answer to `p_all_above_5`.
# 
# *Note:* Like all other questions in this section, this is a probability question. It does not require a simulation.

# In[127]:


p_all_above_5 = (technical_challenge_recipes[technical_challenge_recipes.get('DifficultyScore')>5].shape[0]/
                 technical_challenge_recipes.shape[0])**10
p_all_above_5


# In[128]:


grader.check("q6_6")


# **Question 6.7.** After putting in a lot of time practicing the Technical Challenge recipes, you feel that you need to get some advice from a former participant. You originally had all their names and phone numbers ☎️ written down in your notebook 📓, but your dog 🐶 ate the portion of the notebook with their names, leaving you with a list of just phone numbers. You are left with no choice but to call one of them at random.
# 
# But you also want quality advice, which in your mind are participants who have:
# - remained in the show for at least half a season, and
# - placed in the top 3 in at least one of the Technical Challenges.
# 
# What is the probability that you will get quality advice from calling a random number from the list in your notebook? Assign your answer to `p_quality_advice`.

# In[129]:


week_num = np.array([])
for i in np.arange(1, 12): 
    week = (baker_weeks.groupby(['Season']).max().reset_index()[baker_weeks.groupby(['Season']).max().reset_index().get('Season')==i].get('Week Number')).iloc[0]
    week_num = np.append(week_num, week)
week_num


# In[130]:


eliminate = baker_weeks[baker_weeks.get('Eliminated')==1]
season1 = baker_weeks[baker_weeks.get('Season')==1]
season2 = baker_weeks[baker_weeks.get('Season')==2]
season_rest = baker_weeks[baker_weeks.get('Season')>2]

top3 = np.array(bakers[bakers.get('technical_top3')>=1].get('baker'))
sea1 = season1.groupby('Baker').sum()[season1.groupby('Baker').sum().get('Competed')>=3].reset_index().get('Baker')
sea2 = season2.groupby('Baker').sum()[season2.groupby('Baker').sum().get('Competed')>=4].reset_index().get('Baker')
sea3 = season_rest.groupby('Baker').sum()[season_rest.groupby('Baker').sum().get('Competed')>=5].reset_index().get('Baker')
fitted = np.array([])
for char in np.array(sea1):
    if char in top3:
        fitted = np.append(fitted, char)
for i in np.array(sea2):
    if i in top3:
        fitted = np.append(fitted, i)
for i in np.array(sea3):
    if i in top3:
        fitted = np.append(fitted, i)
fitted


# In[131]:


p_quality_advice = len(fitted)/len(np.unique(bakers.get('baker_full')))
p_quality_advice


# In[132]:


grader.check("q6_7")


# <a id='section7'></a>
# ## Section 7: Recipe Name Generator 👩‍🍳🖨️
# 
# After seeing the creative bakes featured in the Signature and Showstopper Challenges, you're feeling inspired to invent some new recipes yourself. Instead of letting your tastebuds and your better judgment guide you, you decide to generate recipe titles *randomly* in a systematic way. 
# 
# All of your recipe titles will consist words chosen randomly from a limited set of options. There are **three categories of words**:
# 1. *Ingredients* 
#     - For example, "Chocolate", "Pumpkin", and "Mint".
#     
# 2. *Items* 
#     - For example, "Cupcakes", "Croissants", and "Biscuits".
#     
# 3. *Extras* 
#     - For example, "Meringue", "Swirl", and "Ganache".
# 
# To generate a recipe title, you'll first randomly select a template for your recipe title. There are **four recipe templates**:
# 1. *Ingredient Ingredient Item with Ingredient Extra* 
#     - For example, "Chocolate Mint Cupcakes with Pumpkin Swirl".
# 2. *Item with Ingredient Extra*  
#     - For example "Croissants with Mint Ganache".
# 3. *Ingredient, Ingredient, and Ingredient Item* 
#     - For example, "Mint, Chocolate, and Pumpkin Biscuits".
# 4. *Ingredient Ingredient Item* 
#     - For example, "Pumpkin Chocolate Croissants".
# 
# Once you have determined the template, you will randomly select *Ingredients*, *Items*, and *Extras* in the appropriate quantities. Each category of words has an associated probability distribution that describes the likelihood of each word in the category being chosen. 
# 
# Run the next three cells to see the possible words in each category, as well as the probability of choosing each word.

# In[133]:


ingredient_df = bpd.read_csv('data/ingredients.csv')
ingredient_df


# In[134]:


item_df = bpd.read_csv('data/items.csv')
item_df


# In[135]:


extra_df = bpd.read_csv('data/extras.csv')
extra_df


# **Question 7.1.** Write a function called `one_recipe` that generates a random recipe title using the process described above. Start by choosing one of the four possible templates at random, such that each has an equal probability of being selected. Once you have your template, select words from `ingredient_df`, `item_df`, and `extra_df` as required. 
# 
# If you need to select multiple ingredients, make sure to choose them **without replacement** because each ingredient should only occur once in a recipe title. For example, you should not generate "Pumpkin Pumpkin Cupcakes". 
# 
# Your function `one_recipe` should return the title of one randomly generated recipe.
# 
# *Hint*: Use `np.random.choice` and take advantage of the option to specify the probability of each item being selected. See the [documentation](https://numpy.org/doc/stable/reference/random/generated/numpy.random.choice.html).
# 

# In[136]:


n = np.random.choice([1, 2, 3, 4], 1, p=[0.25, 0.25, 0.25, 0.25])
n


# In[137]:


''.join(np.random.choice(ingredient_df.get('ingredients'),1, p = ingredient_df.get('probabilities')) +' ' + np.random.choice(ingredient_df.get('ingredients'),1, p = ingredient_df.get('probabilities')))


# In[138]:


# Templates:
# 1. Ingredient Ingredient Item with Ingredient Extra 
# 2. Item with Ingredient Extra  
# 3. Ingredient, Ingredient, and Ingredient Item 
# 4. Ingredient Ingredient Item

def one_recipe():
    template = np.random.choice([1, 2, 3, 4], 1, p=[0.25, 0.25, 0.25, 0.25])
    ingredient = np.random.choice(ingredient_df.get('ingredients'),3,replace=True, p = ingredient_df.get('probabilities'))
    item = np.random.choice(item_df.get('items'), 2,replace=True, p = item_df.get('probabilities'))
    extra = np.random.choice(extra_df.get('extras'), 1, p = extra_df.get('probabilities'))
    if template == 1:
        return ''.join(ingredient[0] + ' ' + ingredient[1] + ' ' + item[0] + ' with '+ ingredient[2] +  ' ' + item[1])
    if template == 2:
        return ''.join(item[0] + ' with '+ ingredient[0] +  ' ' + extra)
    if template == 3:
        return ''.join(ingredient[0] + ', ' + ingredient[1] + ', and ' + ingredient[2] +  ' ' + item[0])
    if template == 4:
        return ''.join(ingredient[0] + ' ' + ingredient[1] + ' ' + item[0])

one_recipe()


# In[139]:


grader.check("q7_1")


# **Question 7.2.** Generate 10,000 recipe titles and store them in an array called `recipe_titles`. 

# In[140]:


recipe_titles = np.array([])
for i in np.arange(10000):
    recipe_titles = np.append(recipe_titles, one_recipe())
recipe_titles


# In[141]:


grader.check("q7_2")


# **Question 7.3.** You firmly believe that chocolate makes everything better. 🍫 🩹 Use the 10,000 recipe titles that you generated to estimate the probability that a randomly generated recipe title includes the word "Chocolate". Store your estimate in the variable `prob_chocolate`.

# In[142]:


recipe_list = list(recipe_titles)
num_chocolate = sum('Chocolate' in char for char in recipe_list)
num_chocolate / len(recipe_list)


# In[143]:


prob_chocolate = num_chocolate / len(recipe_list)
prob_chocolate


# In[144]:


grader.check("q7_3")


# **Question 7.4.** You're also a big fan of cupcakes. 🧁 Use the 10,000 recipe titles that you generated to estimate the probability that a randomly generated recipe title includes the word "Cupcakes". Store your estimate in the variable `prob_cupcakes`.

# In[145]:


num_cupcakes = sum('Cupcakes' in char for char in recipe_list)
prob_cupcakes = num_cupcakes/len(recipe_list)
prob_cupcakes


# In[146]:


grader.check("q7_4")


# You should have found that your estimate for the probability of a randomly generated recipe containing the word "Chocolate" is significantly higher than the probability associated with the word "Chocolate" in `ingredient_df`. Yet, you also should have found that your estimate the probability of a randomly generated recipe containing the word "Cupcakes" is about the same as the probability associated with the word "Cupcakes" in `item_df`. 
# 
# Compare these values by running the cell below.

# In[147]:


print("The probability associated with Chocolate in the DataFrame is "+
      str(ingredient_df.get('probabilities').iloc[0])+
      " and your estimated probability of Chocolate is "+
      str(prob_chocolate)+".\n")

print("The probability associated with Cupcakes in the DataFrame is "+
      str(item_df.get('probabilities').iloc[0])+
      " and your estimated probability of Cupcakes is "+
      str(prob_cupcakes)+".")


# **Question 7.5** Why is the probability for "Cupcakes" so similar to the value in the DataFrame but the probability for "Chocolate" so different? How can you explain this phenomenon?

# <!-- BEGIN QUESTION -->
# 
# <!--
# BEGIN QUESTION
# name: q7_5
# points: 1
# manual: True
# -->

# It is because that there are four different templates. The average number of ingredients in a template is far more than 1, while the average number of items in a template is about the same as 1.

# <!-- END QUESTION -->
# 
# 
# 
# <a id='section8'></a>
# ## Section 8: Dishwashing 🧼🍽️
# 
# In this section, we will explore whether the difficulty of a recipe is correlated with the number of dirty dishes it produces. Regression is helpful when we want to use one numerical value to predict another numerical value.
# 
# Let's start by visualizing the data with a scatter plot to see if linear regression would make sense for this dataset.

# In[148]:


technical_challenge_recipes.plot(kind='scatter', x='DifficultyScore', y='Dishes');


# Based on the scatter plot, it seems like linear regression would be an appropriate tool. Let's proceed!

# **Question 8.1.** Complete the function `standard_units` which takes in an array or Series and returns an array with the values in standard units. Then use your function to standardize the `'DifficultyScore'` and `'Dishes'` columns from `technical_challenge_recipes`. Store the standardized arrays in the variables `difficulty_standard` and `dishes_standard`. 
# 
# *Note*: Since the inputs to the `standard_units` function might be arrays or Series with some missing values, use [`np.nanmean`](https://numpy.org/doc/stable/reference/generated/numpy.nanmean.html) and [`np.nanstd`](https://numpy.org/doc/stable/reference/generated/numpy.nanstd.html) to compute means and standard deviations. These will ignore the missing values in the computation of the mean and standard deviation.
# 

# In[149]:


def standard_units(sequence):
    '''Returns the input sequence as an array in standard units.'''
    # Convert the input to an array, if it is not already.
    sequence = np.array(sequence)
    return (sequence - sequence.mean())/np.std(sequence)

difficulty_standard = standard_units(technical_challenge_recipes.get('DifficultyScore'))
dishes_standard = standard_units(technical_challenge_recipes.get('Dishes'))


# In[150]:


grader.check("q8_1")


# **Question 8.2.** Complete the function `correlation`, which should take in:
# 1. `df`, a DataFrame,
# 2. `independent`, the column label of the independent ($x$) variable, as a string, and 
# 3. `dependent`, the column label of the dependent ($y$) variable, as a string.
# 
# The function should output the correlation between the two variables. As before, your function needs to work even if there are missing values in the DataFrame.
# 
# Then, use your function to compute the correlation between `'DifficultyScore'` and `'Dishes'` and store your result in the variable `corr`.

# In[151]:


def correlation(df, independent, dependent):
    '''Returns the correlation between the independent and dependent variables in the given DataFrame.'''
    return (standard_units(df.get(independent)) * standard_units(df.get(dependent))).mean()

corr = correlation(technical_challenge_recipes, 'DifficultyScore', 'Dishes')
corr


# In[152]:


grader.check("q8_2")


# **Question 8.3.** Now construct two functions, `reg_slope` and `reg_intercept`, which each take in the same three inputs as `correlation`. `reg_slope` should return the slope of the regression line and `reg_intercept` should return the intercept of the regression line, in original units. As before, your function needs to work even if there are missing values in the DataFrame.
# 
# Use your function to store the slope and intercept of the regression line for  `'DifficultyScore'` and `'Dishes'` in the variables `slope` and `intercept`.

# In[153]:


def reg_slope(df, independent, dependent):
    '''Returns the slope of the regression line in original units.'''
    r = correlation(df, independent, dependent)
    return r * np.std(df.get(dependent)) / np.std(df.get(independent))

def reg_intercept(df, independent, dependent):
    '''Return the intercept of the regression line in original units.'''
    return df.get(dependent).mean() - reg_slope(df, independent, dependent) * df.get(independent).mean()

slope = reg_slope(technical_challenge_recipes, 'DifficultyScore', 'Dishes')
intercept = reg_intercept(technical_challenge_recipes, 'DifficultyScore', 'Dishes')
slope, intercept


# In[154]:


grader.check("q8_3")


# **Question 8.4.** Create a function called `predict` that takes in the same three inputs as `correlation`. `predict` should return an array of predicted values of the dependent variable calculated from the regression line. 
# 
# Use your function to create an array of the predicted number of dirty dishes for each recipe in the `technical_challenge_recipes` DataFrame, based on the recipe's difficulty. Save your answer as `predicted_dishes`. Note that the predicted number of dirty dishes need not be a whole number.
# 

# In[155]:


def predict(df, independent, dependent):
    '''Returns an array of predicted values of the dependent variable calculated from the regression line.'''
    return reg_slope(df, independent, dependent) * df.get(independent) + reg_intercept(df, independent, dependent)

predicted_dishes = predict(technical_challenge_recipes, 'DifficultyScore', 'Dishes')
predicted_dishes


# In[156]:


grader.check("q8_4")


# **Question 8.5.** Use the strategy for overlaying scatter plots described in [this tutorial](https://www.statology.org/pandas-scatter-plot-multiple-columns/) to create an overlaid scatter plot with:
# - a blue dot 🔵 for each recipe showing the difficulty on the $x$-axis and the number of dirty dishes on the $y$-axis (as in the scatter plot at the beginning of this section), and
# - a red dot 🔴 for each recipe showing the difficulty on the $x$-axis and the **predicted** number of dirty dishes on the $y$-axis.
# 
# The red dots should form a straight line - that's the regression line!
# 
# *Note:* This is the first time you've been asked to make an overlaid scatter plot, so you'll need to learn something new to answer this question. Read the linked tutorial carefully and try to mimic their example; you can learn how to do a lot of things this way!

# <!-- BEGIN QUESTION -->
# 
# <!--
# BEGIN QUESTION
# name: q8_5
# points: 1
# manual: True
# -->

# In[157]:


# Create your overlaid scatter plot here.
ax1 = technical_challenge_recipes.plot(kind='scatter', x='DifficultyScore', y='Dishes', color='blue', label='A')
ax2 = technical_challenge_recipes.assign(pre = predicted_dishes).plot(kind='scatter', x='DifficultyScore', y='pre', color='red', label='B', ax=ax1)
ax1.set_xlabel('Difficulty')
ax1.set_ylabel('Dishes')


# <!-- END QUESTION -->
# 
# 
# 
# **Question 8.6.** Use the equation of the regression line to answer the following questions. Check that your answers are reasonable using the scatter plot above. Note that the predicted number of dirty dishes need not be a whole number.
# 
# 1.  A recipe for crème caramel 🍮 has a difficulty score of 7.5. What is the predicted number of dirty dishes for this recipe? Save your answer as `creme_caramel`.
# 2. A basic recipe for chocolate chip cookies 🍪 has a difficulty score of $d$ and an advanced recipe for gourmet chocolate chip cookies 🍪 has a difficulty score of $d+2$. How many additional dirty dishes would we predict the advanced recipe to create, as compared to the basic one? Save your answer as `cookies`.
# 3. A recipe for pretzels 🥨 is predicted to create 6 dirty dishes. What is the difficulty of this recipe? Round to the nearest whole number and save your answer as `pretzels`.

# In[158]:


creme_caramel = slope*7.5+intercept
cookies = 2*slope
pretzels = round((6-intercept)/slope)
print("creme caramel: "+str(creme_caramel))
print("cookies: "+str(cookies))
print("pretzels: "+str(pretzels))


# In[159]:


grader.check("q8_6")


# **Question 8.7.** Now that we have general code to calculate the regression line between any pair of variables in any DataFrame, let's generalize our code for the overlaid scatter plot so we can visualize relationships between other pairs of variables.
# 
# Complete the function `display_predictions` below. This function should take in the same three inputs as the `correlation` function, create an overlaid scatter plot similar to the one in Question 8.5, and return a string describing the correlation between the variables and the slope and intercept of the regression line.

# In[160]:


def display_predictions(df, independent, dependent):
    '''Generates an overlaid scatter plot showing the relationship between the independent and dependent variables in df.
    Returns a string describing the correlation and the slope and intercept of the regression line.'''
    # Create your overlaid scatter plot here.
    ax1 = df.plot(kind='scatter', x=independent, y=dependent, color='blue', label='A')
    ax2 = df.assign(pre = predict(df, independent, dependent)).plot(kind='scatter', x=independent, y='pre', color='red', label='B', ax=ax1)

    # We've provided the code for the return statement.
    return ("The correlation between {0} and {1} is {2}. " +\
           " The slope of the regression line is {3}." + \
           " The intercept of the regression line is {4}.")\
                .format(independent, 
                        dependent, 
                        str(round(correlation(df, independent, dependent), 2)),
                        str(round(reg_slope(df, independent, dependent), 2)), 
                        str(round(reg_intercept(df, independent, dependent), 2)))

# Your function should produce the same scatter plot as in Question 8.5 on the inputs below. 
# Make sure to test it out on other inputs too.
display_predictions(technical_challenge_recipes, 'DifficultyScore', 'Dishes')


# **Question 8.8.** Using the `display_predictions` function you just wrote, create scatter plots and calculate regression lines that would help you answer each of the following questions. 
# 
# 1. Do longer recipes with more sentences require more ingredients?
#     - Store the output of your call to `display_predictions` in the variable `sentences_ingredients`.
# 2. Are recipes with more ingredients more difficult? 
#     - Store the output of your call to `display_predictions` in the variable `ingredients_diff`.

# In[161]:


technical_challenge_recipes


# In[162]:


sentences_ingredients = display_predictions(technical_challenge_recipes, 'RecipeSentences', 'IngredCount')
sentences_ingredients


# In[163]:


grader.check("q8_8_a")


# In[164]:


ingredients_diff = display_predictions(technical_challenge_recipes, 'IngredCount', 'DifficultyScore')
ingredients_diff


# In[165]:


grader.check("q8_8_b")


# <a id='sources'></a>
# ## Data Sources 📖
# 
# - Hill A, Ismay C, Iannone R (2022). bakeoff: Data from "The Great British Bake Off". https://bakeoff.netlify.app/, https://github.com/apreshill/bakeoff.
# 
# - Davis, Erin (2019). Are Great British Bake Off Technical Challenges getting harder? https://erdavis.com/2019/06/08/are-great-british-bake-off-technical-challenges-getting-harder/, https://gist.github.com/erdavis1/09fd4a3aa424c5425a88d47f572ec20a.
# 
# - Ahamed, Nick (2019). Analyzing the Great British Bake Off. https://medium.com/analytics-vidhya/analyzing-the-great-british-bake-off-part-1-ffcdf3791bf3, https://medium.com/@nickahamed/analyzing-the-great-british-bake-off-part-2-1695ff95a0c9, https://docs.google.com/spreadsheets/d/1cvouOik_01QqtFQSq78xODIjcZZ8A-02VXa6IBvdG3I/edit#gid=0.
