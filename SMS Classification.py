#!/usr/bin/env python
# coding: utf-8

# In[12]:


import pandas as pd
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# In[13]:


import chardet
import pandas as pd
with open('spam.csv', 'rb') as f:
    result = chardet.detect(f.read())
data = pd.read_csv('spam.csv', encoding=result['encoding'])


# In[14]:


def preprocess_text(text):
  # Implement your desired text cleaning steps here (lowercase, remove punctuation, etc.)
  text = text.lower()
  text = ''.join([char for char in text if char not in string.punctuation])
  return text


# In[15]:


data["v2"] = data["v2"].apply(preprocess_text)


# In[16]:


# Feature Extraction
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data["v2"])


# In[17]:


# Target variable
y = data["v1"]


# In[18]:


# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[19]:


# Train Model (Naive Bayes)
model = MultinomialNB()
model.fit(X_train, y_train)


# In[20]:


# Prediction and Evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[22]:


# Test on a new message (optional)
new_message = "Oh k...i'm watching here:)"
new_message_vector = vectorizer.transform([new_message])
prediction = model.predict(new_message_vector)[0]
print(f"Predicted label for '{new_message}': {prediction}")


# In[ ]:




