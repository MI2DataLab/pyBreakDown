Performance
===========

This document summarizes performance of R package breakDown and python package breakDown on the same data.

Model-agnostic method (up and down approach) was executed on boston data for regression using standard benchmarking libraries from each language.

R breakDown test code
~~~~~~~~~~~~~~~~~~~~~

Following commands were executed:

.. code:: python

    library(breakDown) 
    library(rpart)
    library(MASS) 
    library(microbenchmark)
    train_data = MASS::Boston[1:300,] 
    model = rpart(medv~.,data=train_data)

**Following commands were executed:**

.. code:: python

    microbenchmark(breakDown::broken(model,MASS::Boston[400,],train_data,direction="up",baseline="Intercept"),times=100)

.. code:: python

    r_results_up = [
      151496945, 179019040, 174285649, 171657389, 186736577, 176873811, 149477996, 156078363, 153822111, 157110590, 164812549, 149597564, 161371930, 150536604, 159660424,
      151099397, 164005548, 160877359, 155673724, 158594509, 156552341, 200853530, 160604062, 151016360, 172159991, 157366522, 147924708, 162123947, 147203023, 162070644,
      151646715, 153851925, 156636659, 153249238, 156439382, 156615324, 157338560, 161933728, 149347225, 178186792, 151804799, 212896701, 153531788, 150779583, 154386491,
      156761934, 188869215, 159124536, 152582601, 160098227, 155172859, 224314278, 210030058, 158369641, 155658547, 157556676, 159644216, 163577044, 160073830, 167877435,
      162871403, 210618121, 152126449, 151551847, 154210243, 162995112, 158628385, 158397297, 156230989, 155259541, 210152892, 167261196, 157579803, 163454795, 210172135,
      207497217, 217937407, 268007391, 238880808, 200768319, 298817194, 187716953, 332831036, 288971446, 310214308, 279432796, 314266951, 240378471, 275743596, 269309196,
      222933544, 147644030, 236694466, 237237474, 166425943, 176675193, 162013906, 176361342, 171035583, 156164433]

.. code:: python

    r_results_up_s = list(map((lambda x:x/1000000000.0), r_results_up)) #nanoseconds to seconds (float)

**Results for following command**

.. code:: python

    microbenchmark(breakDown::broken(model,MASS::Boston[400,],train_data,direction="down",baseline="Intercept"),times=100)

.. code:: python

    r_results_down = [
      157636586, 176977826, 152566930, 176370770, 180238813, 173332343, 155944278, 169256885, 154035595, 182050786, 163405722, 161102422, 150142820, 153812325, 167083885,
      168320711, 172771259, 149702551, 150772637, 158442582, 164156678, 153659233, 171969098, 171280606, 170234515, 161486393, 165544387, 341738681, 166618779, 153261220,
      267645521, 269206271, 364408941, 160329436, 161647353, 271923368, 283175933, 288241510, 210309194, 157228111, 206084831, 232462591, 151892180, 236498265, 256872159,
      183687155, 156095554, 149535499, 270376952, 291976544, 285317108, 231151520, 151745569, 151015639, 193988674, 249337290, 230854189, 159672939, 165214270, 152861032,
      186642382, 184334376, 168095304, 168336995, 169595482, 195182804, 168409596, 340181327, 252239990, 160054458, 222313336, 215507379, 256581788, 250826165, 185914535,
      260640237, 157249833, 179897500, 183111127, 151472923, 167080830, 155694615, 141172815, 165356275, 154660090, 164839180, 150481817, 174504689, 158367491, 244537238,
      200776265, 143306303, 174211859, 143917859, 152706348, 160942923, 169771885, 148698114, 148858718, 189644579
    ]

.. code:: python

    r_results_down_s = list(map((lambda x:x/1000000000.0), r_results_down)) #nanoseconds to seconds (float)

Python pyBreakDown test code
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Following code was executed:

.. code:: python

    import timeit
    setup="""from pyBreakDown import explainer
    from pyBreakDown import explanation
    from sklearn import datasets, tree
    model = tree.DecisionTreeRegressor()
    boston = datasets.load_boston()
    train_data = boston.data[0:300,:]
    model = model.fit(X=train_data,y=boston.target[0:300])
    exp = explainer.Explainer(clf=model, data=train_data, colnames=boston.feature_names)"""

**Similar commands for up and down method**

.. code:: python

    t = timeit.Timer(stmt='exp.explain(observation=boston.data[399,:],direction=\"up", useIntercept=True)', setup=setup)
    p_results_up = t.repeat(number=1,repeat=100)

.. code:: python

    t = timeit.Timer(stmt='exp.explain(observation=boston.data[399,:],direction=\"down", useIntercept=True)', setup=setup)
    p_results_down = t.repeat(number=1,repeat=100)

.. code:: python

    import numpy as np
    def describe (arr):
        print ("Min".ljust(10)+str(np.min(arr)))
        print ("1Q".ljust(10)+str(np.percentile(arr,q=25)))
        print ("Median".ljust(10)+str(np.median(arr)))
        print ("Mean".ljust(10)+str(np.mean(arr)))
        print ("3Q".ljust(10)+str(np.percentile(arr,q=75)))
        print ("Max".ljust(10)+str(np.max(arr)))

Basic statistics for breakDown
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    describe(r_results_down_s)


.. parsed-literal::

    Min       0.141172815
    1Q        0.15694497175
    Median    0.16883324049999998
    Mean      0.19000790343999996
    3Q        0.20714092175
    Max       0.364408941


.. code:: python

    describe(r_results_up_s)


.. parsed-literal::

    Min       0.147203023
    1Q        0.15566992975000002
    Median    0.1611246445
    Mean      0.18094488425
    3Q        0.18800501849999998
    Max       0.332831036


Basic statitics for pyBreakDown
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    describe(p_results_down)


.. parsed-literal::

    Min       0.007466723000106867
    1Q        0.007695871750911465
    Median    0.007944501499878243
    Mean      0.008533558790659299
    3Q        0.008452101750663132
    Max       0.015690394000557717


.. code:: python

    describe(p_results_up)


.. parsed-literal::

    Min       0.007126873002562206
    1Q        0.007325664251766284
    Median    0.007430911500705406
    Mean      0.007852593970528687
    3Q        0.007539984750110307
    Max       0.015298425998480525

