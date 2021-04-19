# AI Workflow - Capstone
* * * 
* * *
## Data Ingestion: Assimilate the business scenario and articulate testable hypotheses.
* * *
The client have asked me to create a service that, at any point in time, will predict the revenue for the following 
month. 
Further, the service should have the ability to project revenue for a specific country.  

The end users currently feel like they are spending too much time using their own methods to predict revenue. 
Additionally, their lack of data science expertise means their predictions are not as accurate as they would like.
The production of this service will save the management team time and increase the accuracy of revenue predictions.
The knock-on effect of this, is that well-projected numbers will help stabilize staffing and budget projections.  

Objectives of the service produced for the client:  
* Save managers time creating their own prediction models.
* Increase accuracy of revenue projections.

From the two objectives we derive the following testable hypothesis for our model. 
> **Hypothesis:** The new model projects monthly revenue more accurately than the managers self produced models.

If the model we create satisfies the above hypothesis, managers will have no need to create their own models as they
would not be better than the one we provided. 
Therefore, the model would satisfy both objectives needed for the service we produce for the client.



## Data Ingestion:State the ideal data to address the business opportunity and clarify the rationale for needing specific data
* * *

|Data |Rationale |  
|:----|:----|
|Accuracy of Managers Models| Without this data the hypothesis cannot be tested, as there will be no benchmark to test my model against|
|The transaction-level data the managers tested their models on|If I test my model on different data than the managers used, the accuracy of the two cannot be compared. It would be the equivalent of saying saying a green pear weighs more than a red apple, when I want to see if a green apple weighs more than a red apple.|
|The actual revenue for each month|To find the accuracy of my model, I need to know what actually occurred.

## Data Modelling: State the different modelling approaches that you will compare to address the business opportunity
* * *
The two modelling approaches I implemented are
* Random Forests
* XGBoost regression

## Articulate your findings with visualisations
* * *
(https://github.com/sommers98/aavail-ai-workflow-capstone/blob/main/monthly_revenue.png)








Usage notes
===============

All commands are from this directory.


To test app.py
---------------------

.. code-block:: bash

    ~$ python app.py

or to start the flask app in debug mode

.. code-block:: bash

    ~$ python app.py -d

Go to http://0.0.0.0:8080/ and you will see a basic website that can be customtized for a project.
    
To test the model directly
----------------------------

see the code at the bottom of `model.py`

.. code-block:: bash

    ~$ python model.py

To build the docker container
--------------------------------

.. code-block:: bash

    ~$ docker build -t iris-ml .

Check that the image is there.

.. code-block:: bash

    ~$ docker image ls
    
You may notice images that you no longer use. You may delete them with

.. code-block:: bash

    ~$ docker image rm IMAGE_ID_OR_NAME

And every once and a while if you want clean up you can

.. code-block:: bash

    ~$ docker system prune


Run the unittests
-------------------

Before running the unit tests launch the `app.py`.

To run only the api tests

.. code-block:: bash

    ~$ python unittests/ApiTests.py

To run only the model tests

.. code-block:: bash

    ~$ python unittests/ModelTests.py


To run all of the tests

.. code-block:: bash

    ~$ python run-tests.py

Run the container to test that it is working
----------------------------------------------    

.. code-block:: bash

    ~$ docker run -p 4000:8080 iris-ml

Go to http://0.0.0.0:4000/ and you will see a basic website that can be customtized for a project.



