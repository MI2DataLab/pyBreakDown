
# pyBreakDown

Python implementation of breakDown package (https://github.com/pbiecek/breakDown).

Currently under construction, stable alpha version delivery date - June, 2018.

## Example usage for decision tree regressor


```python
from pyBreakDown import explainer as e
```

### Load and prepare data


```python
from sklearn import datasets
x = datasets.load_boston()
data = x.data
```

Train any sklearn model


```python
from sklearn import tree
clf = tree.DecisionTreeRegressor()
clf = clf.fit(data, x.target)
```

### Create explainer object


```python
explainer = e.Explainer(clf, x.feature_names)
```

### Explain using "go up" method:


```python
exp = explainer.explain(observation=data[111,:], data=data, direction="up", baseline=0)
```

### Show text results:


```python
exp.text()
```

    Feature                  Contribution   Cumulative     
    Intercept = 1            22.53          22.53          
    LSTAT = 10.16            1.73           24.26          
    RM = 6.715               1.71           25.97          
    B = 395.59               0.26           26.23          
    CRIM = 0.10084           0.23           26.46          
    PTRATIO = 17.8           0.05           26.51          
    ZN = 0.0                 0.0            26.51          
    INDUS = 10.01            0.0            26.51          
    CHAS = 0.0               0.0            26.51          
    RAD = 6.0                0.0            26.51          
    AGE = 81.6               -0.21          26.3           
    DIS = 2.6775             -0.34          25.96          
    TAX = 432.0              -1.22          24.74          
    NOX = 0.547              -1.94          22.8           
    Final prediction                        22.8           
    Baseline = 0


### Text results can be customized:


```python
exp.text(fwidth=30, contwidth=20, cumulwidth=20, digits=4)
```

    Feature                       Contribution        Cumulative          
    Intercept = 1                 22.5328             22.5328             
    LSTAT = 10.16                 1.7267              24.2595             
    RM = 6.715                    1.7107              25.9702             
    B = 395.59                    0.2621              26.2322             
    CRIM = 0.10084                0.2316              26.4638             
    PTRATIO = 17.8                0.0498              26.5136             
    ZN = 0.0                      0.0                 26.5136             
    INDUS = 10.01                 0.0                 26.5136             
    CHAS = 0.0                    0.0                 26.5136             
    RAD = 6.0                     0.0                 26.5136             
    AGE = 81.6                    -0.2146             26.299              
    DIS = 2.6775                  -0.3358             25.9632             
    TAX = 432.0                   -1.2249             24.7383             
    NOX = 0.547                   -1.9383             22.8                
    Final prediction                                  22.8                
    Baseline = 0


### Method "go down" is also supported:


```python
explainer.explain(observation=data[100,:],data=data, direction="down", baseline=0).text()
```

    Feature                  Contribution   Cumulative     
    Intercept = 1            22.53          22.53          
    LSTAT = 9.42             2.66           25.19          
    RM = 6.727               0.57           25.76          
    NOX = 0.52               1.86           27.62          
    DIS = 2.7778             0.02           27.63          
    CRIM = 0.14866           0.2            27.84          
    TAX = 384.0              -0.34          27.5           
    B = 394.76               0.0            27.5           
    PTRATIO = 20.9           0.0            27.5           
    RAD = 5.0                0.0            27.5           
    AGE = 79.9               0.0            27.5           
    CHAS = 0.0               0.0            27.5           
    INDUS = 8.56             0.0            27.5           
    ZN = 0.0                 0.0            27.5           
    Final prediction                        27.5           
    Baseline = 0


### Use intercept as baseline:


```python
explainer.explain(observation=data[100,:],data=data, direction="up", useIntercept=True).text()
```

    Feature                  Contribution   Cumulative     
    Intercept = 1            0              0.0            
    LSTAT = 9.42             2.66           2.66           
    RM = 6.727               0.57           3.22           
    NOX = 0.52               1.86           5.08           
    CRIM = 0.14866           0.3            5.38           
    B = 394.76               0.16           5.55           
    ZN = 0.0                 0.0            5.55           
    INDUS = 8.56             0.0            5.55           
    CHAS = 0.0               0.0            5.55           
    RAD = 5.0                0.0            5.55           
    AGE = 79.9               -0.0           5.55           
    DIS = 2.7778             -0.08          5.46           
    PTRATIO = 20.9           -0.37          5.09           
    TAX = 384.0              -0.13          4.97           
    Final prediction                        4.97           
    Baseline = 22.53


## Example for Random Forest classifier

### Load and prepare data


```python
iris = datasets.load_iris()
```

### Prepare model


```python
from sklearn import ensemble
clf = ensemble.RandomForestClassifier()
clf = clf.fit(iris.data, iris.target)
```

### Make explainer object and explain


```python
explainer = e.Explainer(clf, iris.feature_names)
exp = explainer.explain(observation=iris.data[99,:],data=iris.data,direction="up")
```


```python
exp.text()
```

    Feature                  Contribution   Cumulative     
    Intercept = 1            1.01           1.01           
    petal length (cm) = 4.1  0.22           1.23           
    sepal width (cm) = 2.8   -0.03          1.2            
    sepal length (cm) = 5.7  -0.2           1.0            
    petal width (cm) = 1.3   0.0            1.0            
    Final prediction                        1.0            
    Baseline = 0

