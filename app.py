#backend.py
from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Collect form data
    data = {
        'A1': request.form.get('A1'),
        'A2': request.form.get('A2'),
        'A3': request.form.get('A3'),
        'A4': request.form.get('A4'),
        'A5': request.form.get('A5'),
        'A6': request.form.get('A6'),
        'A7': request.form.get('A7'),
        'A8': request.form.get('A8'),
        'A9': request.form.get('A9'),
        'A10': request.form.get('A10'),
        'AGE_MONS': request.form.get('AGE_MONS'),
        'QCHAT_10_SCORE': request.form.get('QCHAT_10_SCORE'),
        'SEX': request.form.get('SEX'),
        'ETHNICITY': request.form.get('ETHNICITY'),
        'JAUNDICE': request.form.get('JAUNDICE'),
        'FAMILY_MEM_WITH_ASD': request.form.get('FAMILY_MEM_WITH_ASD'),
        'WHO_COMPLETED_THE_TEST': request.form.get('WHO_COMPLETED_THE_TEST')
    }

    # Convert QCHAT_10_SCORE to integer
    qchat_score = int(data['QCHAT_10_SCORE'])
    
    # Initialize prediction result
    prediction_result = ""

    if qchat_score >= 4:
        # Define the percentage mapping
        percentage = 10 + (qchat_score - 1) * 10 + (1 if qchat_score == 10 else 0)
        prediction_result = (
            f"Predicted: Based on the analysis, there is a {percentage}% probability that the child is having Autism Spectrum Disorder (ASD)."
        )
        
        recommendations = ("Recommendations:\n"
        
            "We recommend consulting with the following types of specialists for further evaluation and support:"
            "1. *Pediatrician*: A general pediatrician can provide initial assessments and refer you to specialists."
            "2. *Child Psychologist*: They can conduct thorough evaluations and provide therapy."
            "3. *Developmental Pediatrician*: Specializes in developmental disorders and can offer detailed assessments."
            "4. *Speech-Language Pathologist*: If there are concerns about speech and communication."
            "5. *Occupational Therapist*: For help with sensory issues and daily living skills."
        )
        prediction_result += "\n" + recommendations
    else:
        prediction_result = "Predicted: Based on the analysis, there is a low probability that the child is having Autism Spectrum Disorder (ASD)."

    return render_template('predict.html', result=prediction_result)

if __name__ == '__main__':
    app.run(debug=True)



    


