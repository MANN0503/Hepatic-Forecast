from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np

# Load ML model
model = pickle.load(open('model.sav', 'rb')) 

# Create application
app = Flask(__name__)
# Bind home function to URL
@app.route('/home')
def index():
    return render_template('index.html')

# Bind predict function to URL
@app.route('/predict', methods =['POST','GET'])
def predict():
    int_features =[int(x) for x in request.form.values()]
    final = [np.array(int_features)]
    prediction = model.predict(final)
    output = '{0:.{1}f}'.format(prediction[0][1],2)

    if output>= 1:
        return render_template('index,html', pred="You have liver disease!")
    else:
        return render_template('index.html', pred="You don't have liver disease!")    

 

if __name__ == "__main__":
#Run the application
    
   
