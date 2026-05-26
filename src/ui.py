import requests
import streamlit as st

st.set_page_config(
    page_title='Employee Attrition Prediction',
    page_icon='📊',
    layout='wide'
)

st.title('📊 Employee Attrition Prediction')
st.markdown(
    'Predict employee attrition probability using the trained machine learning model.'
)

with st.form('prediction_form'):

    st.subheader('Employee Information')

    col1, col2, col3 = st.columns(3)

    with col1:

        Age = st.slider(
            'Age',
            min_value=18,
            max_value=60,
            value=30
        )

        BusinessTravel = st.selectbox(
            'Business Travel',
            options=[
                'Non-Travel',
                'Travel_Rarely',
                'Travel_Frequently'
            ]
        )

        DailyRate = st.slider(
            'Daily Rate',
            min_value=102,
            max_value=1499,
            value=800
        )

        Department = st.selectbox(
            'Department',
            options=[
                'Sales',
                'Research & Development',
                'Human Resources'
            ]
        )

        DistanceFromHome = st.slider(
            'Distance From Home',
            min_value=1,
            max_value=29,
            value=10
        )

        Education = st.selectbox(
            'Education',
            options=[
                'Below College',
                'College',
                'Bachelor',
                'Master',
                'Doctor'
            ]
        )

        EducationField = st.selectbox(
            'Education Field',
            options=[
                'Life Sciences',
                'Medical',
                'Marketing',
                'Technical Degree',
                'Human Resources',
                'Other'
            ]
        )

        EnvironmentSatisfaction = st.slider(
            'Environment Satisfaction',
            min_value=1,
            max_value=4,
            value=3
        )

        Gender = st.selectbox(
            'Gender',
            options=['Male', 'Female']
        )

        HourlyRate = st.slider(
            'Hourly Rate',
            min_value=30,
            max_value=100,
            value=60
        )

    with col2:
        JobInvolvement = st.slider(
            'Job Involvement',
            min_value=1,
            max_value=4,
            value=3
        )

        JobLevel = st.slider(
            'Job Level',
            min_value=1,
            max_value=5,
            value=2
        )

        JobRole = st.selectbox(
            'Job Role',
            options=[
                'Sales Executive',
                'Research Scientist',
                'Laboratory Technician',
                'Manufacturing Director',
                'Healthcare Representative',
                'Manager',
                'Sales Representative',
                'Research Director',
                'Human Resources'
            ]
        )

        JobSatisfaction = st.slider(
            'Job Satisfaction',
            min_value=1,
            max_value=4,
            value=3
        )

        MaritalStatus = st.selectbox(
            'Marital Status',
            options=[
                'Single',
                'Married',
                'Divorced'
            ]
        )

        MonthlyIncome = st.slider(
            'Monthly Income',
            min_value=1009,
            max_value=19999,
            value=5000
        )

        MonthlyRate = st.slider(
            'Monthly Rate',
            min_value=2094,
            max_value=26999,
            value=12000
        )

        NumCompaniesWorked = st.slider(
            'Number of Companies Worked',
            min_value=0,
            max_value=9,
            value=2
        )

        OverTime = st.selectbox(
            'Over Time',
            options=['Yes', 'No']
        )

        PercentSalaryHike = st.slider(
            'Percent Salary Hike',
            min_value=11,
            max_value=25,
            value=15
        )

    with col3:

        PerformanceRating = st.slider(
            'Performance Rating',
            min_value=3,
            max_value=4,
            value=3
        )

        RelationshipSatisfaction = st.slider(
            'Relationship Satisfaction',
            min_value=1,
            max_value=4,
            value=3
        )

        StockOptionLevel = st.slider(
            'Stock Option Level',
            min_value=0,
            max_value=3,
            value=1
        )

        TotalWorkingYears = st.slider(
            'Total Working Years',
            min_value=0,
            max_value=40,
            value=10
        )

        TrainingTimesLastYear = st.slider(
            'Training Times Last Year',
            min_value=0,
            max_value=6,
            value=2
        )

        WorkLifeBalance = st.slider(
            'Work Life Balance',
            min_value=1,
            max_value=4,
            value=3
        )

        YearsAtCompany = st.slider(
            'Years At Company',
            min_value=0,
            max_value=40,
            value=5
        )

        YearsInCurrentRole = st.slider(
            'Years In Current Role',
            min_value=0,
            max_value=18,
            value=3
        )

        YearsSinceLastPromotion = st.slider(
            'Years Since Last Promotion',
            min_value=0,
            max_value=15,
            value=1
        )

        YearsWithCurrManager = st.slider(
            'Years With Current Manager',
            min_value=0,
            max_value=17,
            value=3
        )

    submitted = st.form_submit_button('Predict Attrition')


if submitted:

    payload = {
        'Age': Age,
        'BusinessTravel': BusinessTravel,
        'DailyRate': DailyRate,
        'Department': Department,
        'DistanceFromHome': DistanceFromHome,
        'Education': Education,
        'EducationField': EducationField,
        'EnvironmentSatisfaction': EnvironmentSatisfaction,
        'Gender': Gender,
        'HourlyRate': HourlyRate,
        'JobInvolvement': JobInvolvement,
        'JobLevel': JobLevel,
        'JobRole': JobRole,
        'JobSatisfaction': JobSatisfaction,
        'MaritalStatus': MaritalStatus,
        'MonthlyIncome': MonthlyIncome,
        'MonthlyRate': MonthlyRate,
        'NumCompaniesWorked': NumCompaniesWorked,
        'OverTime': OverTime,
        'PercentSalaryHike': PercentSalaryHike,
        'PerformanceRating': PerformanceRating,
        'RelationshipSatisfaction': RelationshipSatisfaction,
        'StockOptionLevel': StockOptionLevel,
        'TotalWorkingYears': TotalWorkingYears,
        'TrainingTimesLastYear': TrainingTimesLastYear,
        'WorkLifeBalance': WorkLifeBalance,
        'YearsAtCompany': YearsAtCompany,
        'YearsInCurrentRole': YearsInCurrentRole,
        'YearsSinceLastPromotion': YearsSinceLastPromotion,
        'YearsWithCurrManager': YearsWithCurrManager
    }

    try:
        response = requests.post("http://localhost:8080/predict", json=payload).json()
        st.divider()
        st.subheader('Prediction Result')

        prediction = response['prediction']
        probability = response['probability']

        if prediction == 1:
            st.error(
                f'Employee predicted to attrite with probability: {probability:.2%}'
            )
        else:
            st.success(
                f'Employee predicted to stay with probability: {(1-probability):.2%}'
            )

        st.write(response)

    except Exception as e:
        st.error(f'An error occured: {e}')