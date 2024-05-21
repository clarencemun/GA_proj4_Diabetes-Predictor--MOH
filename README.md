<img src="http://imgur.com/1ZcRyrc.png" style="float: left; margin: 20px; height: 55px">

# DSI-SG-42
## Project 4b: Managing the Prevalence of Diabetes in Singapore
> Authors: Clarence Mun, Syahiran Rafi, Conrad Aw
---

## Contents
---
- [Executive Summary](#executive-summary)
- [Problem Statement](#problem-statement)
- [Data Collection](#data-collection)
- [Data Dictionary](#data-dictionary)
- [Data Cleaning, Preprocessing and EDA](#Data-Cleaning,-Preprocessing-and-EDA)
- [Modelling](#Modelling)
- [Conclusion](#Conclusion)
- [Recommendations](#Recommendations)

## Executive Summary
---
Suppose you're part of a data science team at the Ministry of Health (MOH), ministry of the Government of Singapore responsible for managing the public healthcare system in Singapore. In recent discussions with healthcare providers, you've noticed growing health-consciousness around diabetes.

With an emphasis on career-building in recent years, long working hours, high stress and irregular meals are the norm for the typical working adult in Singapore.

This scenario, while simplified, illustrates the growing health-consciousness regarding diabetes among young adults. In Singapore, with its unique population demographics and lifestyle patterns, the challenge is not only diagnosing but also predicting and managing diabetes effectively to reduce long-term complications and healthcare burdens.

**Real-world problem**: Enhance the early detection and predictive management of diabetes among Singaporeans to improve patient outcomes and reduce the healthcare system's strain.

**Data science problem**: Develop a predictive model that accurately identifies individuals at high risk of developing diabetes, utilizing healthcare data to minimize false negatives (missed diagnoses) and false positives (unnecessary interventions), thereby enabling targeted and timely healthcare interventions.

## Problem Statement
---
According to the Ministry of Health, about one in three Singaporeans has a lifetime risk of developing diabetes. To address this challenge, we propose developing a data-driven solution that utilises healthcare data and predictive analytics to identify individuals at high risk of developing diabetes.

By leveraging classification algorithms and population health data, our solution aims to provide a risk assessment of diabetes for individuals to enable early detection and targeted intervention. Additionally, our solution also aims to equip individuals with the ability to make more informed nutritional choices by providing healthier suggestions for everyday food products. 

## Data Collection
---
Source: https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset?resource=download

We have selected the following datasets from the above source:

### 1. diabetes_binary_5050split_health_indicators_BRFSS2015.csv
- A clean dataset of 70,692 survey responses to the CDC's BRFSS2015.
- It has an equal 50-50 split of respondents with no diabetes and with either prediabetes or diabetes.
- The target variable Diabetes_binary has 2 classes. 0 is for no diabetes, and 1 is for prediabetes or diabetes.
- This dataset has 21 feature variables and is balanced.

### 2. diabetes_binary_health_indicators_BRFSS2015.csv
- A clean dataset of 253,680 survey responses to the CDC's BRFSS2015.
- The target variable Diabetes_binary has 2 classes. 0 is for no diabetes, and 1 is for prediabetes or diabetes.
- This dataset has 21 feature variables and is not balanced.

In our initial exploration, we used both datasets #1 and #2 to evaluate model performance and runtime complexity. Due to dataset #2's significant class imbalance, majority of the model evaluation metrics are below 0.7 even after tuning. Additionally, since dataset #3 is more than 3x the size of dataset #1, the runtime is also approximately 2-3x longer.

With these considerations, our team decided to proceed with dataset #1, as seen in this Part 2A notebook. The exploration using dataset #2 can be found in the part 2B notebook.

## Data Dictionary
---
### For raw data file 'diabetes_binary_5050split_health_indicators_BRFSS2015.csv':

|Feature|Type|Description|
|---|---|---|
|Diabetes_binary|int64|Diabetic or non-diabetic with a value of 1 else 0|
|HighBP|int64|Adults who have been told they have high blood pressure by a doctor, nurse, or other health professional with a value of 1 else 0|
|HighChol|int64|Adults who have been told they have high cholestrol by a doctor, nurse, or other health professional with a value of 1 else 0|
|CholCheck|int64|Cholesterol check within past five years with a value of 1 else 0|
|BMI|int64|Body Mass Index (BMI)|
|Smoker|int64|Smoked atleast 100 cigarettes in life with a value of 1 else 0|
|Stroke|int64|Ever told they had a stroke with a value of 1 else 0|
|HeartDiseaseorAttack|int64|Coronary heart diease or myocardial infarction with a value of 1 else 0|
|Fruits|int64|Consume Fruit once or more times per day with a value of 1 else 0|
|Veggies|int64|Consume Vegetables once or more times per day with a value of 1 else 0|
|HvyAlcoholConsump|int64|Heavy drinkers with a value of 1 else 0 (adult men having more than 14 drinks per week and adult women having more than 7 drinks per week)|
|AnyHealthcare|int64|Have any kind of health care coverage with a value of 1 else 0|
|NoDocbcCost|int64| Unable to see a doctor when needed to in the past 12 months because of cost with a value of 1 else 0|
|GenHlth|int64|Individuals rating their own general health on a scale of 1-5 (1 = excellent 2 = very good 3 = good 4 = fair 5 = poor)|
|MentHlth|int64|Individuals rating their own mental health on a scale of 1 - 30 days based on how many days they experienced stress, depression, and problems with emotions during the past 30 days (scale 1-30 days)|
|PhysHlth|int64|Individuals rating their own physical health on a scale of 1 -30 days based on how many days they were physically ill or injured during the past 30 days (scale 1-30 days)|
|DiffWalk|int64|Serious difficulty walking or climbing stairs with a value of 1 else 0|
|Sex|int64|Gender (0 for female, 1 for male)|
|Age|int64|13-level age category from 18 years old to 80+ years|
|Education|int64|Education level on a scale of scale 1-6 (1 = Never attended school or only kindergarten 2 = Grades 1 through 8 (Elementary) 3 = Grades 9 through 11 (Some high school) 4 = Grade 12 or GED (High school graduate) 5 = College 1 year to 3 years (Some college or technical school) 6 = College 4 years or more (College graduate))|
|Income|int64|Income on a scale of 1-8 (1 = less than $10,000 5 = less than $35,000 8 = $75,000 or more)|

## Data Cleaning, Preprocessing and EDA
---
The following variable labels were adjusted based on the detailed data dictionary for better clarity in the plots:

|Feature|Type|Description|
|---|---|---|
|GenHlth_Label|int64|1: 'Excellent', 2: 'Very Good', 3: 'Good', 4: 'Fair', 5: 'Poor'|
|Age_Label|int64|1: '18-24', 2: '25-29', 3: '30-34', 4: '35-39', 5: '40-44', 6: '45-49', 7: '50-54', 8: '55-59', 9: '60-64', 10: '65-69', 11: '70-74', 12: '75-79', 13: '80+', 14: 'Unknown'|
|Income_Label|int64|1: '<$15,000', 2: '$15,000-$24,999', 3: '$25,000-$34,999', 4: '$35,000-$49,999', 5: '≥$50,000', 9: 'Unknown'|
|Education_Label|int64|1: 'Did not graduate High School', 2: 'Graduated High School', 3: 'Attended College/Technical School', 4: 'Graduated College/Technical School', 9: 'Unknown'|

The following interaction terms were created:

|Feature|Type|Description|
|---|---|---|
|BMI_HighBP_interaction|int64|Individuals with a high BMI might experience a more significant increase in blood pressure compared to those with a normal BMI.|
|Age_PhysActivity_interaction|int64|Older individuals might respond differently to physical activity due to age-related physiological changes.|

## Modelling
---
Models used for tuning and selection
- Logistics Regression
- Random Forest
- XGBoost
- Decision Tree
- Gradient Boost
- Support Vector Machine
- Neural Network
    
### Model scores
---
|Model                     |Train Score    |Test Score    |Cross-Validation Score|
-----------------------|--------------|-------------|------------------------|
|Logistic Regression          |0.747476      |0.745385         |0.74767|
|Random Forest                |0.965625      |0.715751         |0.72242|
|XGBoost                      |0.774123      |0.744749         |0.745318|
|Decision Tree                |0.965643      |0.659382         |0.662688|
|Gradient Boost               |0.753735      |0.74779          |0.751101|
|Support Vector Machine       |0.746079      |0.741566         |0.745283|
|Neural Network               |0.751384      |0.746941          |0.745814|

|Model                     |Accuracy    |Precision    |Sensitivity    |Specificity    |F1-Score|
|----------------------  |----------  |-----------  |-------------  |-------------  |----------|
|Logistic Regression       |0.745385     |0.733477       |0.77083        |0.719943    |0.75169|
|Random Forest             |0.715751     |0.704369       |0.743528       |0.687977    |0.723419|
|XGBoost                   |0.744749     |0.726618       |0.784694       |0.704809    |0.75454|
|Decision Tree             |0.659382     |0.66647        |0.637997       |0.680764    |0.651923|
|Gradient Boost            |0.74779      |0.729284       |0.788089       |0.707496    |0.757547|
|Support Vector Machine    |0.741566     |0.71666        |0.798981       |0.684158    |0.755585|
|Neural Network            |0.746941     |0.709871       |0.835196       |0.658699    |0.767451|

### Model Selection: Neural Network
---
By comparing Train Scores with Cross-Validation Scores, we can eliminate "Random Forest" and "Decision Tree" as they both show evidence of overfitting (Train Score is much greater than CV Score).

For Sensitivity, we can consider the top two models "Support Vector Machine" and "Neural Network" -- both of which have Sensitivity scores greater than 0.79.

"Support Vector Machine" performs slightly better than "Neural Network" in Precision and Specificity. "Neural Network" performs slightly better than "Support Vector Machine" in Accuracy and F1-Score. NN is significantly better than SVM in Sensitivity.

With the above analysis, we have selected Neural Network as our model before Hyperparameter turning

### Hyperparameter turning for Neural Network
---
We performed both `GridSearchCV` and `RandomSearchCV`.
- `GridSearchCV` performs an exhaustive search over all combinations of hyperparameters specified in a grid.
- `RandomSearchCV` randomly samples hyperparameter combinations from specified distributions or ranges.

Both `GridSearchCV` and `RandomSearchCV` gave a conclusive result for best parameters and model:
- `Best Parameters: {'hidden_layer_sizes': (50,), 'alpha': 0.1, 'activation': 'tanh'}`
- `Best Model: MLPClassifier(activation='tanh', alpha=0.1, hidden_layer_sizes=(50,),random_state=42)`
- `Sensitivity` score before tuning: 0.8351
- Best `Sensitivity` score after tuning with `GridSearchCV`: 0.8356
- Best `Sensitivity` score after tuning with `RandomSearchCV`: 0.8002

We will use the best model from from `GridSearchCV` for model evaluation due to the higher `Sensitivity` score after tuning.

### Model Evaluation

Neural Network Pre- vs. Post-Tuning Evaluation

| Metric           |   Pre-Tuning Score |   Post-Tuning Score |   Percentage Change |
|------------------|--------------------|---------------------|---------------------|
| Cross-Validation |           0.745814 |            0.747847 |            0.272659 |
| Accuracy         |           0.746941 |            0.748073 |            0.151501 |
| Precision        |           0.709871 |            0.722497 |            1.7786   |
| Sensitivity      |           0.835196 |            0.805489 |           -3.55691  |
| Specificity      |           0.658699 |            0.690665 |            4.85291  |
| F1-Score         |           0.767451 |            0.761739 |           -0.744256 |

While there is a drop in `Sensitivity` after tuning, the post-tuning `Sensitivity` score of  0.805 still exceeds the other models we considered previously. The drop in `F1-Score` of 0.74% is marginal and should not affect the model's performance by much.

On the other hand, three other performance metrics -- `Accuracy`, `Precision` and `Specificity` -- increased, making the model more well-balanced overall.

## Conclusion
---
While `Sensitivity` has remained largely constant at 0.805, `Precision` has suffered significantly, dropping from 0.722 to 0.295. This means that our model is highly sensitive to False Positives.

In the context of our problem, individuals who may have lower to no risk of diabetes may still be flagged as being of higher risk. For a disease detection model, having low precision may not be ideal due to ethical concerns related to false diagnoses.

While our model does not claim to formally diagnose diabetes, a balance of `Sensitivity` and `Precision` should ultimately be sought for the model to be serviceable to the general public.

## Recommendations
---
For future consideration, here are some ways in which the precision of a binary classification model could be improved.

1. **Feature Selection and Engineering:** To build a more robust model, we could gather more relevant data from local participants/patients to train the model. (The original dataeset is from the US.) This may allow us to go deeper with feature engineering like creating new features (e.g., interaction terms, polynomial features) or transforming existing features to provide more discriminatory power to the model, potentially improving precision.

2. **Address Class Imbalance:** It is worth exploring further how we could use the imbalanced dataset to train the model (as seen in Part 2B). Apart from stratification, resampling techniques and adjusting class weights -- which we have attempted -- we could also explore other techniques such as bootstrapping and cost-sensitive learning to see if model performance could be improved.

Implementing these strategies while monitoring the impact on precision can help build a more effective diabetes detection model. It's important to experiment and iteratively refine the model based on performance feedback to achieve the desired balance of sensitivity and precision levels.

---

### Files

**Code**
- 01_Cleaning_and_EDA.ipynb   
- 02A_Modelling_Evaluation_and_Conclusion.ipynb
- 02B_Modelling_ImBalanced_Trial.ipynb
- 02C_Modelling_Tensorflow_Trial.ipynb

**Datasets**
- diabetes_binary_5050split_health_indicators_BRFSS2015.csv
- diabetes_binary_health_indicators_BRFSS2015.csv

Group 2 - Project 4.pdf

README.md

---