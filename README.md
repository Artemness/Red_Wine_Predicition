# Red_Wine_Predicition
Predicting Red Wine Rating Given a Number of Characteristics

## Resources:  
Red Wine Data: https://www.kaggle.com/uciml/red-wine-quality-cortez-et-al-2009  

## Data Collection:  
Utilized a Data set available from kaggle.

## Data Cleaning:  
Rounded some of the variables to clean up the data.

## EDA:  
In order to better understand some of the data, utilized some charts to visualize the data and distributions of the variables.
* Most of the variables followed a normal distribution with a few exceptions:  
    * Volatile Acidity
    * Citric Acid
    * Alcohol
* Checking the Box Plots for the different variables could help to give an understanding of the different outliers contained in the data:  
    * Residual sugars stood out having quite a number of outliers in regards to quality:  
        * This is somewhat explained by the large variance in residual sugars: min-0.9 and max- 15.5  
    * Chlorides also had a substantial amount of outliers:  
        * Again this could be explained by the large variance in chloride values: min- 0.012 and max- 0.611
