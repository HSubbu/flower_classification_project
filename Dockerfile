FROM python:3.8-slim

COPY . .
RUN pip3 install --upgrade pip

RUN pip install -r requirements.txt
#RUN pip3 install https://raw.githubusercontent.com/alexeygrigorev/serverless-deep-learning/master/tflite/tflite_runtime-2.2.0-cp37-cp37m-linux_x86_64.whl --no-cache-dir
RUN pip3 install --extra-index-url https://google-coral.github.io/py-repo/ tflite_runtime

COPY mobilet_flower_v3.tflite mobilet_flower_v3.tflite
COPY streamlit_app.py streamlit_app.py
#EXPOSE 8501
EXPOSE $PORT
#for loca testing
#ENTRYPOINT ["streamlit", "run"]
#CMD ["streamlit_app.py"]
#
# for dynamic port assigned by Heroku server
CMD streamlit run streamlit_app.py --server.port $PORT
