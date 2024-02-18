import streamlit as st
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
from io import BytesIO
from reportlab.lib.pagesizes import letter
import sweetviz as sv
from reportlab.pdfgen import canvas
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# Load the trained model
model = pickle.load(open('pipe_model.sav', 'rb'))

with st.sidebar:
    selected = option_menu('Navigation',['About','Predict','Model','Analysis','Feedback','Report'],icons=["bi-info-circle","bi-bullseye","bi-bar-chart-fill","bi-clipboard-data","bi-textarea-resize","bi-file-pdf"] , default_index = 0)

df = pd.read_csv('Churn_Modelling.csv')
X = df.drop(columns=['Exited'])
Y = df['Exited']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

analysis_results = []

def generate_pdf_report(analysis_results, graphs):
    pdf_filename = "churn_analysis_report.pdf"
    with open(pdf_filename, "w") as pdf_file:
        pdf_file.write("Churn Analysis Report\n\n")
        
        for result in analysis_results:
            pdf_file.write(f"{result}\n")

        pdf_file.write("\nGraphs:\n")
        for graph in graphs:
            pdf_file.write(f"{graph}\n")

    st.success(f"PDF Report generated successfully: [{pdf_filename}]")
    st.download_button(label="Download PDF Report", data=pdf_filename, file_name=pdf_filename, mime="application/pdf")

def model_evaluation(X_test, Y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(Y_test, y_pred)
    classification_rep = classification_report(Y_test, y_pred)
    confusion_mat = confusion_matrix(Y_test, y_pred)

    st.subheader("Model Performance Metrics:")
    st.write(f"Accuracy: {accuracy:.2%}")
    st.write("Classification Report:")
    st.text(classification_rep)
    st.write("Confusion Matrix:")
    st.table(confusion_mat)

    # ROC Curve
    fpr, tpr, thresholds = roc_curve(Y_test, model.predict_proba(X_test)[:, 1])
    roc_auc = auc(fpr, tpr)
    st.subheader("Receiver Operating Characteristic (ROC) Curve:")
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel('False Positive Rate (FPR)')
    ax.set_ylabel('True Positive Rate (TPR)')
    ax.set_title('ROC Curve')
    st.pyplot(fig)

if selected  == "About":
    st.title("Customer Churn Prediction App")
    st.write("Customer churn prediction is a process in which businesses use data analysis and machine learning techniques to identify and forecast customers who are likely to stop using their products or services.")
    st.text("")
    st.text("")
    st.image('cc.png', width=600)
    st.write("WHY PROJECT IS BEING MADE ?")
    st.write("Identify At-Risk Customers: Predicting customer churn helps businesses identify customers who are likely to discontinue using their products or services.Recognizing these at-risk customers early on allows companies to implement targeted retention strategies")
    st.write("Customer Satisfaction: Improving customer satisfaction is a key goal. Predicting and addressing factors that contribute to churn allows organizations to enhance their products, services, or customer support, leading to increased satisfaction and loyalty.")
    st.write("Prevent Revenue Loss: Losing customers can result in a decline in revenue. By predicting churn, businesses can take proactive measures to retain valuable customers, thereby minimizing revenue loss.")

elif selected == "Predict":

    def churn_prediction(input_data):
 
        input_data_as_numpy_array = np.asarray(input_data)
        input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
        prediction = model.predict(input_data_reshaped)

        if prediction[0] == 0:
            st.success('CUSTOMER IS PREDICTED TO STAY.')
        else:
            st.error('CUSTOMER IS PREDICTED TO CHURN.')

    st.title('Customer Churn Predictor App')

    credit_score = st.number_input('Credit Score')
    geography = st.selectbox('Geography', options=df['Geography'].unique())
    gender = st.selectbox('Gender', options=df['Gender'].unique())
    age = st.number_input('Age')
    tenure = st.selectbox('Tenure', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    balance = st.number_input("Enter your balance : ")
    num_of_products = st.selectbox('Number of Products', [1, 2, 3, 4], index=0)
    has_credit_card = st.selectbox('Has Credit Card', ['Yes', 'No'])
    has_credit_card = 1 if has_credit_card == 'Yes' else 0
    is_active_member = st.radio('Is Active Member', ['Yes', 'No'])
    is_active_member = 1 if is_active_member == 'Yes' else 0
    estimated_salary = st.number_input('Estimated Salary')


    if st.button('Predict Churn'):
            input_data = [credit_score, geography, gender, age, tenure, balance, num_of_products, has_credit_card, is_active_member, estimated_salary]
            churn_prediction(input_data)

elif selected == "Model":
    model_evaluation(X_test, Y_test)

elif selected == "Analysis":
    st.title("Analysis Dashboard")

    # Dropdowns for selecting columns
    bar_column = st.selectbox('Select Column for Bar Graph', ['Geography', 'Gender', 'HasCrCard', 'IsActiveMember'])
    pie_column = st.selectbox('Select Column for Pie Chart', ['Gender', 'Exited'])
    hist_column = st.selectbox('Select Column for Histogram', ['CreditScore','Age', 'Balance', 'EstimatedSalary'])
    box_column = st.selectbox('Select Column for Box Plot', ['CreditScore', 'Age', 'Balance'])
    violin_column = st.selectbox('Select Column for violin Plot', ['CreditScore', 'Balance'])


    # bar_column, pie_column, hist_column, box_column, violin_column = st.columns(5)
    bar_color = st.color_picker('Choose Bar Color', '#ff0000')
    pie_color = st.color_picker('Choose pie Color', '#00FFE0')
    hist_color = st.color_picker('Choose Histogram Color', '#1f77b4')
    box_color = st.color_picker('Choose box Color', '#E0FF00')
    violin_color = st.color_picker('Choose violin Color', '#006BFF')

    # Bar graph
    bar_fig = px.bar(df, x=bar_column,color_discrete_sequence=[bar_color], title=f'Bar Graph for {bar_column}')
    st.plotly_chart(bar_fig)

    # Pie chart
    pie_fig = px.pie(df, names=pie_column,color_discrete_sequence=[pie_color], title=f'Pie Chart for {pie_column}')
    st.plotly_chart(pie_fig)

    # Histogram
    hist_fig = px.histogram(df, x=hist_column,color_discrete_sequence=[hist_color], title=f'Histogram for {hist_column}')
    st.plotly_chart(hist_fig)

    # Box plot
    box_fig = px.box(df, x='Exited', y=box_column,color_discrete_sequence=[box_color], title=f'Box Plot for {box_column}')
    st.plotly_chart(box_fig)

    # Violin plot
    violin_fig = px.violin(df, x='Exited', y=violin_column,color_discrete_sequence=[violin_color], title=f'Violin Plot for {violin_column}')
    st.plotly_chart(violin_fig)

    probas = model.predict_proba(X_test)[:, 1]
    df_probs = pd.DataFrame({'Probability': probas, 'Actual': Y_test})

    prob_dist_fig = px.histogram(df_probs, x='Probability', color='Actual',
                                title='Probability Distribution Plot', marginal='box')
    st.plotly_chart(prob_dist_fig)

    analysis_results.append(f"Bar Graph Analysis: {bar_column}")
    analysis_results.append(f"Pie Chart Analysis: {pie_column}")
    analysis_results.append(f"Histogram Analysis: {hist_column}")
    analysis_results.append(f"Box Plot Analysis: {box_column}")
    analysis_results.append(f"Violin Plot Analysis: {violin_column}")

    report = sv.analyze(df)
    report_filename = "customer churn report.html"
    report.show_html(report_filename)
    analysis_results.append(f"CUSTOMER CHURN Analysis: {report_filename}")

elif selected == "Feedback":
    st.subheader("Feedback Section")
    user_feedback = st.text_area("Share your feedback here:")
    if st.button("Submit Feedback"):
        if not user_feedback.strip():
            st.error("Please provide feedback before submitting.")
        else:
            st.success("Thank you for your feedback! We appreciate it.")

elif selected == "Report":
    generate_pdf_report(analysis_results, graphs=[])
    st.markdown("Report Generated: Dhruv Patel")
