## Real vs Fake News Classifier

### Introduction
The increasing spread of misinformation and fake news has posed significant challenges in the digital age. With the advent of social media and online news platforms, it has become imperative to develop reliable methods to differentiate between real and fake news. In this task, we implemented a deep learning-based classifier using an LSTM (Long Short-Term Memory) and a CNN netowrk to tackle this issue. The classifier aims to identify whether a given news article is real or fake based on its textual content and few other feaures including its type and language.

### Dataset Used
The dataset used for this task consists of news articles labeled as either "Real" or "Fake." The dataset contains various columns such as author, published date, title, text, language, site URL, main image URL, type of news, and the label indicating whether the news is real or fake. Here are some statistics about the dataset:

- **Total Articles:** 2045
- **Languages:** English, German, French, Spanish, and others
- **Labels:** 1291 Fake, 754 Real
- **News Types:** bs, conspiracy, bias, hate, satire, state, junksci, fake

### Data Preprocessing
Data preprocessing is a crucial step to ensure the data is in the right format for training the machine learning model. The following steps were taken to preprocess the data:

1. **Handling Missing Values:** Dropped rows with missing/nan values.
2. **Text Cleaning:** Removed HTML tags, converted text to lowercase, removed punctuation, tokenized the text, and lemmatized tokens using NLTK's WordNetLemmatizer.
3. **Dropping Unnecessary Columns:** Dropped columns like title, site URL, and main image URL as well as repeated columns which were not directly used in training the model.
4. **Tokenization and Padding:** Tokenized the text data and converted it into sequences. The sequences were then padded to ensure uniform input length for the model. vocabulary size and max length of sequence has also been treated as hyperparameters to assess their effects on the model performance

### Deep Learning Model Used
We used a deep learning model with the following architecture:

1. **Embedding Layer:** Converts the input text into dense vectors of fixed size.
2. **LSTM Layer:** Captures long-term dependencies and patterns in the text data.
3. **Dense Layer:** Outputs a single value indicating the probability of the news being real or fake.

It takes encoded text of the news as input and assesses whether it is real or fake news. 
The model was compiled with the binary cross-entropy loss function and Adam optimizer.

### Results
The model was trained for 10 epochs with a batch size of 32. Here are the performance metrics:

- **Training Accuracy:** Increased significantly across epochs.
- **Validation Accuracy:** Started to plateau and slightly decrease, indicating potential overfitting.
- **Test Accuracy:** 70.52%

### Second Model
In the 2nd scenario we have used news language and its type encoded as features to a DL model with two deense layers and an output layer. In one example, language and news types are converted into features using onehot encoding  and in 2nd example we have used tokenizer to convert these features to numbers to feed them as input to the DL model.
test Accuracy score is 64.54% for both scenarios.
### Future Recommendations
Based on the results and observations, here are some recommendations for future improvements:

1. **Data Augmentation:** Increase the dataset size by augmenting the text data to improve model generalization.
2. **Regularization:** Implement dropout layers and L2 regularization to reduce overfitting.
3. **Advanced Architectures:** Experiment with more complex models like Bidirectional LSTM.
4. **Hyperparameter Tuning:** Perform grid search or random search to find the optimal hyperparameters.
5. **Ensemble Methods:** Combine both models to improve overall performance by incorporating textual as well as other features like language and news type.
By addressing these aspects, the performance of the fake news classifier can be significantly enhanced, leading to more reliable and accurate results.
