# Red_Wine_Predicition
Predicting Red Wine Rating as Bad(0), Good (1) or Exceptional (2) Given a Number of Characteristics

## Data Collection:  
Utilized a Data set available from Kaggle.

## Data Cleaning:  
Rounded some of the variables to clean up the data.

## EDA:  
In order to better understand some of the data, utilized some charts to visualize the data and distributions of the variables.
* Most of the variables followed a normal distribution with a few exceptions:  
    * Volatile Acidity
    * Citric Acid
    * Alcohol
* To better understand the relationship between the different variables, we can take a look at the a heatmap:  
   <img src= "/heatmap.png" height=540 width=540>
* Checking the Box Plots for the different variables could help to give an understanding of the different outliers contained in the data:  
    * Residual sugars stood out having quite a number of outliers in regards to quality:  
      ![](/sugarsboxplot.png "Sugars Box Plot")  
        * This is somewhat explained by the large variance in residual sugars: min-0.9 and max- 15.5  
    * Chlorides also had a substantial amount of outliers:  
      ![](/chloridesboxplot.png "Chlorides Box Plot")  
        * Again this could be explained by the large variance in chloride values: min- 0.012 and max- 0.611
        
## Modeling:  
### Ran a number of models, and created confusion matrices for each model. After testing, the top three models without tuning were:  
**Random Forest Accuracy: 69.79**  
   <img src="/RandomForestConfusionMatrix.png" width=426 height=320>  
The Random Forest outperformed all the other models without tuning and correctly labeled 165 Bad wines as Bad, misrepreseted 46 Bad wines as Good and 2 Bad wine as Exceptional. It correctly attributed 131 Good wines in their respective category, mislabeled 51 Good wines as bad and 18 Good wines as Exceptional. The Exceptional Wines had 39 correctly categorized, 27 incorrectly categorized as Good and 1 categorized as Bad.
   <br>
**Linear SVC Accuracy: 65.83**  
<img src="/LinearSVCConfusionMatrix.png" width=426 height=320> 
The Linear SVC Model correctly labeled under half of the Bad wines with only 81 being correctly categorized. 101 of the Bad wines were miscategorized under Good and 31 under Exceptional. The model correctly attributed 121 Good wines to its respective category, with 40 being miscategorized under Bad and 39 under Exceptional. In addition the model correctly categorized 18 Exceptional wines as Exceptional, 48 incorrectly under Good and 1 under Bad.
<br>
**Logistic Regression Accuracy: 58.13**  
   <img src="/LogisticRegressionConfusionMatrix.png" width=426 height=320>
The Logistic Regression Classifier Model placed 160 Bad wines in their correct category, 52 Bad wines were missclassified as Good and 1 Bad wine was misclassified as Exceptional. The model correctly assigned 115 Good wines to their correct category with 4 being misclassified as Exceptional and 81 misclassified as Bad. The Exceptional wines were mostly missclassified as only 4 Exceptional wines were correctly categorized and 59 were classified under Good with 4 classified under Bad.
   <br>

## Resources:  
Red Wine Data: https://www.kaggle.com/uciml/red-wine-quality-cortez-et-al-2009  
GridSearchCV: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html  
Hypertuning: https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74  
