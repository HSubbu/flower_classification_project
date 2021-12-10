# FLOWER CLASSIFICATION PROJECT
**A deep learning model which can classify images of a flower into 5 names**
![image](https://user-images.githubusercontent.com/30765337/145608887-ee56500a-91e9-4640-be5f-d7df482edac6.png)


### Project Description
The problem has been attempted from folllowing practice datathon https://dphi.tech/challenges/data-sprint-25-flower-recognition/61/overview/about
The problem statement in concise words is to to develop and deploy a ML model which can identify name of the flower from the image and classify into one of the five classes.The classes are - daisy, dandelion, roses, sunflowers and tulips. 

### Brief Description of Project Lifecycle and Project Repository
The dataset contains raw jpeg images of five types of flowers.The dataset can be downloaded from the given link: https://drive.google.com/file/d/1H0rJmSBmYQoWM2w2tqy-jmX0Y2Wg6k2v/view?usp=sharing. 

The colab notebook comprises of code in prepreocessing the data , developing a model using python Tensorflow and saving the best model .
*The Data preprocessing* involved unzipping the main data folder,  splitting the train data into a validation dataset following similar structure of train data (images are in 5 files with folder names as respective flower names) using 20% of train data for validation . The datset also provided test data and model was tested using the test data and then evaluating on dphi website(based on accuracy score).Transfer Learning methodology for model building was used and a MobileNetV2 was used as shortlisted model. The final model gave an accuracy of 90% on test data. The final model was saved for further development process. A train.py was also prepared from the colab notebook and is available in the repository for further training in case of upgrade (due to model drift or retraining with additional data).

As the basic Tensorflow model is heavy , the model was converted to a light weight tensorflow lite model using convert.py script. *It is noted that i created a virtual environment in my local machine using conda (called tensorlfow-cpu) and all dependencies were installed inside the VE*. 

To deploy the model , a python library *streamlit* was used . A complete python script streamlit_app.py which included the GUI for the web service and prediction of flower name using the model.predict() method on the user image. The same was tested on local machine using the command *$streamlit run streamlit_app.py*. The web service was designed to give the user option of loading the image from file or provide image url for prediction. 

Subsequently, the web service was containerized using Docker and tested on local machine. The Dockerfile contains the requisite code for creating the image.
The Docker container was deployed on Heroku ( a limited free service!) and is available  at https://project2-mlzoomcamp.herokuapp.com/ . 
The repository also contains text document containing screen shots important aspects of model development and deployment.

### Testing

(a) The webservice can be tested at https://project2-mlzoomcamp.herokuapp.com/ . The user can upload and image or provide link (sample images and a text doc with URL links has been provided in repo for easy testing ) . 

(b) for testing the code following steps may be taken: 

    (i) clone the project repo to your local machine reference https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository
    
    (ii) open the Dockerfile and make changes to PORT setting as indicated in the comments of Dockerfile
    
    (iii) build the image , $docker build -t streamlit_app .
    
    (iv) once the image is build sucessfully , run $docker run -it --rm -p 8501:8501 streamlit_app
    
    (v) open localhost:/8501 and this will open the application and can be tested using the images and URL provided in the folder
    

### Conclusion

The design and deployment of webservice **flower_classification_project** was undertaken as a part of Capstone Project for Course in ML with DataTalksClub https://datatalks.club/courses/2021-winter-ml-zoomcamp.html . The model documentation to be uploaded with webs service also needs revision. It is envisaged that further improvement to model performance and presentation of application can be undertaken in due course for next version of application.
