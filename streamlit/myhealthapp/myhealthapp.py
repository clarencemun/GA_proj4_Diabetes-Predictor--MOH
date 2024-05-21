import streamlit as st
import pickle
import numpy as np
import pytesseract
from PIL import Image
import cv2
import re
import os
import io

# Example of adding custom styles
st.markdown("""
<style>
.custom-font {
    font-family: Arial, sans-serif;
    color: #333;
}
.custom-header {
    color: #4CAF50;
}
</style>
""", unsafe_allow_html=True)

st.title('My Health App')

# Example usage of tabs with enhanced appearance
tab1, tab2 = st.tabs(["🔍 Diabetes Risk Assessment", "🍎 Sweet-Spot"])

with tab1:
    st.image('/Users/clarencemun/GA/personal/project_4/streamlit/myhealthapp/diabetes_banner.jpg', width=700)  # Example banner - ensure you have an image named 'diabetes_banner.jpg'
    st.markdown("""
<style>
.safe-text-color {
    color: #787878;  /* Medium gray for good contrast in both light and dark themes */
}
</style>
<div class="safe-text-color">
    Welcome to the <strong>Diabetes Risk Assessment</strong> tool. This quick assessment aims to give you insight into your diabetes risk based on lifestyle and health factors. Let's get started on creating a healthier future together.
</div>
""", unsafe_allow_html=True)


    st.header("👤 User Profile")
    # Initialize session state for BMI if it does not exist
    if 'bmi' not in st.session_state:
        st.session_state['bmi'] = 22.5  # Default BMI value, adjust as needed

    # Function to load the model
    def load_model():
        with open('/Users/clarencemun/GA/personal/project_4/streamlit/myhealthapp/tuned_models_gs.pkl', 'rb') as file:
            model = pickle.load(file)
        return model

    # Load your model
    model_dict = load_model()
    model = model_dict['Neural Network']

    age = st.selectbox(
        'Select Age Range',
        ('Age 18 to 24', 'Age 25 to 29', 'Age 30 to 34', 'Age 35 to 39', 
        'Age 40 to 44', 'Age 45 to 49', 'Age 50 to 54', 'Age 55 to 59',
        'Age 60 to 64', 'Age 65 to 69', 'Age 70 to 74', 'Age 75 to 79', 
        'Age 80 or older')
    )

    # Mapping of age ranges to a representative value or handling logic
    # You can adjust the representative values based on how your model was trained
    age_mapping = {
        'Age 18 to 24': 21,
        'Age 25 to 29': 27,
        'Age 30 to 34': 32,
        'Age 35 to 39': 37,
        'Age 40 to 44': 42,
        'Age 45 to 49': 47,
        'Age 50 to 54': 52,
        'Age 55 to 59': 57,
        'Age 60 to 64': 62,
        'Age 65 to 69': 67,
        'Age 70 to 74': 72,
        'Age 75 to 79': 77,
        'Age 80 or older': 85, # Assuming a representative value for 80 or older
    }

    # Convert selected age range to its representative value
    age_value = age_mapping[age]

    # Education level dropdown
    education_options = [
        'Never attended school or only kindergarten',
        'Primary School (Grades 1 to 6)',
        'Secondary School (Grades 7 to 10 or 11)',
        "GCE 'O' Level or 'N' Level (Grade 11 or 12)",
        "Polytechnic Diploma or GCE 'A' Level",
        "University Degree (Bachelor’s or higher)"
    ]
    education = st.selectbox('Highest Level of Education', options=education_options)

    # Mapping of education options to numeric values
    education_mapping = {
        'Never attended school or only kindergarten': 1,
        'Primary School (Grades 1 to 6)': 2,
        'Secondary School (Grades 7 to 10 or 11)': 3,
        "GCE 'O' Level or 'N' Level (Grade 11 or 12)": 4,
        "Polytechnic Diploma or GCE 'A' Level": 5,
        "University Degree (Bachelor’s or higher)": 6
    }

    # Convert selected education level to its numeric value
    education_value = education_mapping[education]
    # Use education_value for your model input

    # Income level dropdown
    income_options = {
        'Less than SGD 20,000': 1,
        'SGD 20,000 to less than SGD 35,000': 2,
        'SGD 35,000 to less than SGD 50,000': 3,
        'SGD 50,000 to less than SGD 70,000': 4,
        'SGD 70,000 or more': 5,
        'Prefer not to say': 9
    }
    income_selection = st.selectbox('Annual Household Income', options=list(income_options.keys()))

    # Convert selected income level to its numeric value
    income_value = income_options[income_selection]
    # Use income_value for your model input


    st.header('BMI Calculator')

    # User inputs for weight and height
    weight = st.number_input('Weight in kilograms', value=70.0, step=0.1)
    height = st.number_input('Height in meters', value=1.75, step=0.01)

    def calculate_bmi():
        bmi_val = weight / (height ** 2)
        st.session_state.bmi = bmi_val  # Update the session state
        return bmi_val

    if st.button('Calculate BMI'):
        bmi = weight / (height ** 2)
        st.metric(label="Your BMI", value=f"{bmi:.2f}")
        
        # BMI Categories and tactful messaging
        if bmi < 18.5:
            category = "under the recommended range"
            advice = "It's advisable to see a healthcare provider for assessment and possibly dietary adjustments."
        elif bmi < 23:
            category = "within the recommended range"
            advice = "Maintaining a balanced diet and regular physical activity are key to staying healthy."
        elif bmi < 30:
            category = "above the recommended range"
            advice = "Consider evaluating your diet and physical activity level. Seeking advice from a healthcare provider might be helpful."
        else:
            category = "significantly above the recommended range"
            advice = "It's a good idea to discuss your health with a healthcare provider to explore personalized advice and options."
        
        # Display the category and advice
        st.subheader(f"Your BMI suggests that you're {category}.")
        st.write(advice)

    # Streamlit app layout adjustments for horizontal radio buttons

    st.header('Health Status')
    # Custom CSS to style the radio buttons
    st.markdown("""
    <style>
        div.row-widget.stRadio > div{flex-direction:row;}
    </style>
    """, unsafe_allow_html=True)

    # Function to display Yes/No radio buttons side by side with custom CSS
    def yes_no_radio(label, key):
        return st.radio(label, ['Yes', 'No'], index=1, key=key, horizontal=True)

    highbp = yes_no_radio('Have you been diagnosed with high blood pressure?', 'highbp')
    highchol = yes_no_radio('Have you been diagnosed with high cholesterol?', 'highchol')
    cholcheck = yes_no_radio('Have you had your cholesterol levels checked in the last 5 years?', 'cholcheck')
    stroke = yes_no_radio('Have you ever had a stroke?', 'stroke')
    heartdiseaseorattack = yes_no_radio('Have you ever been diagnosed with heart disease or had a heart attack?', 'heartdiseaseorattack')


    st.header('Your Physical Health Status')

    # General Health dropdown
    genhlth_options = ['Poor', 'Fair', 'Good', 'Very Good', 'Excellent']
    genhlth = st.selectbox('How would you rate your overall health?', options=genhlth_options, index=genhlth_options.index('Good'))

    # Mapping selections to numeric values
    genhlth_mapping = {
        'Poor': 5,
        'Fair': 4,
        'Good': 3,
        'Very Good': 2,
        'Excellent': 1
    }
    genhlth_value = genhlth_mapping[genhlth]


    st.markdown("""
    **Physical Health:** Reflect on the last month. How many days were impacted by health issues, such as illness or pain?
    """, unsafe_allow_html=True)

    # Slider for input
    physhlth = st.slider('Number of days', min_value=0, max_value=30, value=0, step=1, format="%d days")

    # You can adjust the default value and step as needed. "%d days" formats the display value.

    physactivity = yes_no_radio('In the past month, have you engaged in physical activity?', 'physactivity')
    diffwalk = yes_no_radio('Have you experienced any difficulty walking recently?', 'diffwalk')

    # Convert Yes/No answers to binary (Yes: 1, No: 0)
    binary_answers = {
        'Yes': 1,
        'No': 0
    }

    # Calculated interaction terms, converting Yes/No to binary
    bmi_highbp_diffwalk_interaction = st.session_state.bmi * binary_answers[highbp] * binary_answers[diffwalk]
    age_highchol_heartdiseaseorattack_interaction = age_value * binary_answers[highchol] * binary_answers[heartdiseaseorattack]
    genhlth_physhlth_interaction = genhlth_value * physhlth

    # Collect all feature inputs into an array, converting Yes/No to binary for relevant features
    input_features = np.array([
        binary_answers[highbp], binary_answers[highchol], binary_answers[cholcheck], st.session_state.bmi, binary_answers[stroke],
        binary_answers[heartdiseaseorattack], binary_answers[physactivity], genhlth_value, physhlth, binary_answers[diffwalk],
        age_value, education_value, income_value, bmi_highbp_diffwalk_interaction, age_highchol_heartdiseaseorattack_interaction,
        genhlth_physhlth_interaction
    ]).reshape(1, -1)


    # Button to make prediction
    if st.button('Assess'):
        prediction = model.predict(input_features)
        if prediction[0] == 0:
            st.success('Based on the information provided, your risk level for diabetes is lower. Maintaining a healthy lifestyle and regular check-ups are good ways to continue managing your health effectively.')
        else:
            st.warning('Based on the information provided, you may have a higher risk level for diabetes. It is advisable to consult with a healthcare provider for a comprehensive assessment and personalized advice.')

# Tab 2: Another Feature

# Set the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'/opt/homebrew/bin/tesseract'  # Update this path

with tab2:
    st.image("/Users/clarencemun/GA/personal/project_4/streamlit/myhealthapp/nutrigrade_banner.jpg", width=700)  # Example banner - ensure you have an image named 'diabetes_banner.jpg'
    st.markdown("""
<style>
.safe-text-color {
    color: #787878;  /* Medium gray for good contrast in both light and dark themes */
}
</style>
<div class="safe-text-color">
    Welcome to <strong>Sweet-Spot</strong>. This simple analysis tool is designed to empower you with knowledge about the nutritional quality of your food choices. Let’s embark on a journey towards optimal nutrition together.
</div>
""", unsafe_allow_html=True)
    
    def extract_sugar_content(image_path):
        # Convert the uploaded file to bytes and then to an image
        # Open the image file
        image = Image.open(image_path)

        # Perform OCR using pytesseract
        text = pytesseract.image_to_string(image)
        
        # Define regex pattern to extract the number before "mL" or "ml"
        serving_size_pattern = r'(\d+)\s*m[lL]'  # Pattern to match digits before "mL" or "ml"
        # Use regex to find the match in the text
        serving_size_match = re.search(serving_size_pattern, text)

        # Extract the number if match is found
        if serving_size_match:
            serving_size = float(serving_size_match.group(1))  # Extract the number and convert to float
        else:
            st.write("Serving size not found")
            return # Return None if no match is found
        sugar_pattern = r"(?:Total )?(?:Sugar|Sugars) (d{1,1}|\d+)"  # Updated pattern to handle variations
    
        sugar_matches = re.findall(sugar_pattern, text)
        if sugar_matches:
            sugar_value = float(sugar_matches[0])  # Convert extracted value to float
            sugar_per_100ml = sugar_value / serving_size * 100

            # Determine warning message based on sugar content
            if sugar_per_100ml == 0:
                st.write("NO SUGAR CONTENT")
            elif sugar_per_100ml > 10:
                st.write("WARNING: HIGH SUGAR CONTENT")
            elif sugar_per_100ml > 5:
                st.write("MODERATE SUGAR CONTENT")
            else:
                st.write("LOW SUGAR CONTENT")

            st.write(f"Total sugar content per 100 ml: {sugar_per_100ml:.1f} g")
        else:
            st.write("No sugar information found")
        # else:
        #     st.write("Serving size not found")

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
        
        # Save the uploaded image to a temporary file
        temp_image_path = 'temp_image.jpg'
        with open(temp_image_path, 'wb') as f:
            f.write(uploaded_file.getvalue())

        # Extract "Total Sugar" and its value
        # total_sugar_text = extract_sugar_content(temp_image_path)

        # Display the extracted "Total Sugar" text
        # if total_sugar_text:
        #     st.write(f"Extracted Text: {total_sugar_text}")
        # else:
        #     st.write("Total Sugar not found")

    # Process the extracted text and display the sugar content classification
        extract_sugar_content(uploaded_file)

        # Clean up temporary file
        os.remove(temp_image_path)