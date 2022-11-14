import profile
from matplotlib.pyplot import title
import streamlit as st
import pandas as pd
import numpy as np
import requests
import inspect
from streamlit_lottie import st_lottie
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
from numerize import numerize
from itertools import chain
import plotly.graph_objects as go
import plotly.express as px
import joblib
import sklearn
import statsmodels.api as sm
# Display lottie animations
def load_lottieurl(url):

    # get the url
    r = requests.get(url)
    # if error 200 raised return Nothing
    if r.status_code !=200:
        return None
    return r.json()
# Extract Lottie Animations

lottie_home = load_lottieurl("https://assets3.lottiefiles.com/packages/lf20_tijmpky4.json")
lottie_dataset = load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_grpcjnlf.json")
lottie_prediction= load_lottieurl("https://assets4.lottiefiles.com/private_files/lf30_ghysqmiq.json")

#Title
st.set_page_config(page_title='Mental Health at workplace',  layout='wide')

#header
t1, t2 = st.columns((0.4,1)) 
t2.title("Mental Health & Well-being")

#Hydralit Navbar
import hydralit_components as hc
from streamlit_option_menu import option_menu
# define what option labels and icons to display
Menu = option_menu(None, ["Home",  "EDA", "Prediction"], icons=['house',"bar-chart-line","clipboard-check"],
menu_icon="cast", default_index=0, orientation="horizontal", 
styles={"container": {"padding": "0!important", "background-color": "#fafafa"},"icon": {"color": "black", "font-size": "25px"}, "nav-link": {"font-size": "15px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},"nav-link-selected": {"background-color": "#4F6272"},})
# Home Page
if Menu == "Home":
      # Display Introduction
#Upload Image
    image=Image.open('mental health.jpg')
    from PIL import Image
    title_container = st.container()
    col1, mid, col2 = st.columns([1,4,1])
    col3, mid1, col4 = st.columns([1,4,1])
    with title_container:
      with mid:
       st.image(image,caption='')
    with title_container:
      with mid1:
        st.write('Without effective support, mental disorders and other mental health conditions can affect a person confidence and identity at work, capacity to work productively, absences and the ease with which to retain or gain work.')

# EDA page
df=pd.read_csv("output.csv")
if Menu == "EDA":
  st.header("Visualizations")
  m1, m2, m3, m4, m5, m6 = st.columns((1,1,1,1,1,1))
  m1.metric(label="Number of employees", value ="1237")
  m2.metric(label="Treatment", value ="624")
  m3.metric(label="Family History", value ="483")
  m4.metric(label="Care Options", value ="436")
  m5.metric(label="Welness Program", value ="228")
  m6.metric(label="Benefits", value ="467")
#Visualization
  g1,g2= st.columns((1,2))
  k1,k2=st.columns((3,1))
  import plotly.express as px
  import plotly.graph_objects as go
  import chart_studio.plotly as py
  import seaborn as sns
  from plotly.subplots import make_subplots
  import matplotlib.pyplot as plt

#visualization of categorical variables

  df_ = df.drop(['Age', 'Country'], axis=1)

  buttons = []
  i = 0
  vis = [False] * 24

  for col in df_.columns:
      vis[i] = True
      buttons.append({'label' : col,
               'method' : 'update',
               'args'   : [{'visible' : vis},
               {'title'  : col}] })
      i+=1
      vis = [False] * 24

  fig = go.Figure()

  for col in df_.columns:
      fig.add_trace(go.Pie(
               values = df_[col].value_counts(),
               labels = df_[col].value_counts().index,
               title = dict(text = 'Distribution of {}'.format(col),
                            font = dict(size=18, family = 'monospace'),
                            ),
               hole = 0.5,
               hoverinfo='label+percent',))

  fig.update_traces(hoverinfo='label+percent',
                    textinfo='label+percent',
                    textfont_size=15,
                    opacity = 0.8,
                    showlegend = False,
                    marker = dict(colors = ['#4F6272', '#B7C3F3','#A7C3F6'] ,
                                line=dict(color='#4F6272', width=1)))
              

  fig.update_layout(margin=dict(t=0, b=0, l=0, r=0),
                    updatemenus = [dict(
                          type = 'dropdown',
                          x = 1.15,
                          y = 0.9,
                          showactive = True,
                          active = 0,
                          buttons = buttons)],
                   annotations=[
                               dict(text = "<b>Choose<br>Column<b> : ",
                               showarrow=False,
                               x = 1.02, y = 0.99, yref = "paper", align = "left")])
  for i in range(1,22):
      fig.data[i].visible = False
  g1.plotly_chart(fig, use_container_width=True) 

# visualization of non-categorical variables
  fig = make_subplots(rows = 1, cols=2)
  fig.append_trace(go.Bar(
                          y = df['Country'].value_counts(),
                          x = df['Country'].value_counts().index,
                          name = 'Observations from Countries ',
                          text = df['Country'].value_counts(),
                          textfont = dict(size = 12,
                                          family = 'monospace'),
                          textposition = 'outside',
                          marker=dict(color="#4F6272")
                          ), row=1, col=1)

  fig.append_trace(go.Histogram(
                          x = df['Age'],
                          nbinsx = 8,
                          text = ['16', '500', '562', '149', '26', '5', '1'],
                          marker =  dict(color="#B7C3F3")),
                          row=1, col=2)
# For Subplot : 1

  #fig.update_xaxes(
          #row=1, col=1,
          #tickfont = dict(size=10, family = 'monospace'),
          #tickmode = 'array',
          #ticktext = df['Country'].value_counts().index,
          #tickangle = 50,
          #ticklen = 8,
          #showline = False,
          #showgrid = False,
          #ticks = 'outside')

     #fig.update_yaxes(type = 'log',
         # row=1, col=1,
          #tickfont = dict(size=15, family = 'monospace'),
          #tickmode = 'array',
          #showline = False,
          #showgrid = False,
          #ticks = 'outside')
  #fig.update_traces(
                    #marker_line_color='black',
                    #marker_line_width= 1.2,
                    #opacity=0,
                    # row = 1, col = 1)

  #fig.update_xaxes(range=[-1,10], row = 1, col = 1)

# For Subplot : 2

  fig.update_xaxes(        
          title = dict(text = 'Age',
                       font = dict(size = 13,
                                   family = 'monospace')),
          row=1, col=2,
          tickfont = dict(size=15, family = 'monospace', color = 'black'),
          tickmode = 'array',
          ticktext = ['10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79'],
          ticklen = 6,
          showline = False,
          showgrid = False,
          ticks = 'outside')

  fig.update_yaxes(
          row=1, col=2,
          tickfont = dict(size=15, family = 'monospace'),
          tickmode = 'array',
          showline = False,
          showgrid = False,
          ticks = 'outside')

  fig.update_traces(
                    marker_line_color='black',
                    marker_line_width = 2,
                    opacity = 0.6,
                    row = 1, col = 2)


  fig.update_layout(height=500, width=900,
                    title = dict(text = 'Visualization of non-categorical variables<br>1. Observation from Countries<br>2. Ages Count',
                                 x = 0.5,
                                 font = dict(size = 16, color ='#27302a',
                                 family = 'monospace')),
                    showlegend = False)
  g2.plotly_chart(fig, use_container_width=True) 


#Results
  cw1, cw2 = st.columns((5, 0.5))
  seek = df[df.treatment == 'Yes'].drop(['treatment', 'Country', 'Age'], axis=1)
  dont = df[df.treatment == 'No'].drop(['treatment', 'Country', 'Age'], axis=1)
  buttons = []
  i = 0
  vis = [False] * 21

  for col in seek.columns:
      vis[i] = True
      buttons.append({'label' : col,
               'method' : 'update',
               'args'   : [{'visible' : vis},
               {'title'  : col}] })
      i+=1
      vis = [False] * 21

  fig = make_subplots(rows=1, cols=2,
                specs=[[{'type':'domain'}, {'type':'domain'}]])

  for col in dont.columns:
      fig.add_trace(go.Pie(
               values = dont[col].value_counts(),
               labels = dont[col].value_counts().index,
               title = dict(text = 'No Treatment: <br>Distribution<br>of {}'.format(col),
                            font = dict(size=18, family = 'monospace'),
                            ),
               hole = 0.5,
               hoverinfo='label+percent',),1,1)
  for col in seek.columns:
      fig.add_trace(go.Pie(
               values = seek[col].value_counts(),
               labels = seek[col].value_counts().index,
               title = dict(text = 'Seek Treatment: <br>Distribution<br>of {}'.format(col),
                            font = dict(size=18, family = 'monospace'),
                            ),
               hole = 0.5,
               hoverinfo='label+percent',),1,2)

  fig.update_traces(hoverinfo='label+percent',
                    textinfo='label+percent',
                    textfont_size=15,
                    opacity = 0.8,
                    showlegend = False,
                    marker = dict(colors = ['#4F6272', '#B7C3F3', '#DD7596', '#8EB897', '#A7C3F6'] ,
                                line=dict(color='#4F6272', width=1)))

  fig.update_traces(row=1, col=2, hoverinfo='label+percent',
                    textinfo='label+percent',
                    textfont_size=15,
                    opacity = 0.8,
                    showlegend = False,
                    marker = dict(colors = ['#4F6272', '#B7C3F3', '#DD7596', '#8EB897', '#A7C3F6'] ,
                                line=dict(color='#4F6272', width=1)))

  fig.update_layout(margin=dict(t=0, b=0, l=0, r=0),
                  font_family   = 'monospace',
                  height=300, width=600,
                  updatemenus = [dict(
                          type = 'dropdown',
                          x = 0.60,
                          y = 0.70,
                          showactive = True,
                          active = 0,
                          buttons = buttons)],
                  annotations=[
                               dict(text = "<b>Choose<br>Column<b> : ",
                                    font = dict(size = 14),
                               showarrow=False,
                               x = 0.5, y = 0.90, yref = "paper", align = "left")])
  fig.update_layout(height=400, width=800,
                    title = dict(text = '<br>How did employees respond to questions whether<br>they received treatment or not?',
                                 x = 0.5,
                                 font = dict(size = 16, color ='#27302a',
                                 family = 'monospace')),
                    showlegend = False)
  for i in range(1,42): 
      fig.data[i].visible = False
  fig.data[21].visible = True
  cw1.plotly_chart(fig, use_container_width=True)  


# Machine Learning Application
# Import necessary librariries
import joblib
import sklearn
import statsmodels.api as sm
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import MinMaxScaler
# Machine Learning Application
if Menu == "Prediction":

    col = st.columns(2)

    # Article
    with col[0]:
        st.markdown("""
        <h3 class="f2 f1-m f-headline-l measure-narrow lh-title mv0">
        Know The Risks
         </h3>
         <p class="f5 f4-ns lh-copy measure mb4" style="text-align: justify;font-family: Sans Serif">
        It's time to assess if any employee may seek medical attention. 
        To see the outcome, fill out the employee details.
         </p> 
            """,unsafe_allow_html = True)

    # Lottie Animation
    with col[1]:
        st_lottie(lottie_prediction, key = "churn", height = 300, width = 800)
     

    # Get Numerical features
    # Monthly Charges are dropped since have high multicollinearity with total Charges
    numerical_features = ['Age']

    # Get Categorical Features
    categorical_features = ['treatment','Gender','self_employed','family_history','work_interfere','no_employees','remote_work','tech_company','benefits','care_options',
                            'wellness_program','seek_help','anonymity','leave','mental_health_consequence','phys_health_consequence',
                            'coworkers','supervisor','mental_health_interview','phys_health_interview','mental_vs_physical','obs_consequence']

    # Subset Data based on numerical features
    df_numerical = df[numerical_features]
    df_categorical = df[categorical_features]
    X = pd.concat([df_categorical,df_numerical], axis = 1)
    # Target Variable

    y = df['treatment']

    # Define a Function of several inputs for the user to fill

    def user_features():

        # Employee Information

        st.title('Employee Information')

        cols1 = st.columns(3)

        with cols1[0]:
            Age = st.number_input("Age", value = 18.00, min_value =18.00, max_value=70.00)

        with cols1[1]:
            Gender = st.selectbox("Gender",('Female','Male','Other'))

        with cols1[2]:
            self_employed = st.selectbox("self_employed",("Yes","No"))
        

        cols2 = st.columns(3)
        with cols2[0]:
            family_history = st.selectbox("family_history",('Yes','No'))

        with cols2[1]:
            work_interfere = st.selectbox("work_interfere",("Sometimes","Often","Rarely","Never","no answer"))

        with cols2[2]:
            no_employees = st.selectbox("no_employees",("1-5","6-25","26-100","100-500","500-1000","More than 1000"))


        st.write("------")

        # Company Information
        st.title("Company Information")
        cols3 = st.columns(4)
        with cols3[0]:
            remote_work = st.selectbox("remote_work",("Yes","No"))

        with cols3[1]:
            tech_company = st.selectbox("tech_company",("Yes","No"))

        with cols3[2]:
            benefits = st.selectbox("benefits",("Yes","No","Don't Know"))

        with cols3[3]:
            care_options  = st.selectbox("care_options ",("Yes","No","Not sure"))

        cols4 = st.columns(4)

        with cols4[0]:
            wellness_program= st.selectbox("wellness_program ",("Yes","No","Don't Know"))

        with cols4[1]:
            seek_help = st.selectbox("seek_help",("Yes","No","Don't Know"))

        with cols4[2]:
            anonymity = st.selectbox("anonymity",("Yes","No","Dont't Know"))

        with cols4[3]:
            leave = st.selectbox("leave",("Very Easy","Somewhat Easy","Somewhat Difficult","Very Difficult","Don't Know"))


        cols5 = st.columns(4)
        with cols5[0]:
            mental_health_consequence = st.selectbox("mental_health_consequence",("Yes","No","Maybe"))

        with cols5[1]:
            phys_health_consequence = st.selectbox("phys_health_consequence ",("Yes","No","Maybe"))

        with cols5[2]:
            coworkers = st.selectbox("coworkers",("Yes","No","Some of them"))

        with cols5[3]:
            supervisor = st.selectbox("supervisor",("Yes","No","Some of them"))

        cols6 = st.columns(4)
        
        with cols6[0]:
            mental_health_interview = st.selectbox("mental_health_interview",("Yes","No","Maybe"))
        
        with cols6[1]:
            phys_health_interview = st.selectbox("phys_health_interview",("Yes","No","Maybe"))
        
        with cols6[2]:
            mental_vs_physical = st.selectbox("mental_vs_physical",("Yes","No","Don't Know"))
        
        with cols6[3]:
            obs_consequence = st.selectbox("obs_consequence",("Yes","No"))

        # Transform the inputs into a dataframe shape of size 1 so the model predict the outcome
        dataframe = {'Age':Age,
                    'Gender':Gender,
                    'self_employed':self_employed,
                    'family_history':family_history,
                    'work_interfere':work_interfere,
                    'no_employees':no_employees,
                    'remote_work':remote_work,
                    'tech_company':tech_company,
                    'benefits':benefits,
                    'care_options':care_options,
                    'wellness_program':wellness_program,
                    'seek_help':seek_help,
                    'anonymity':anonymity,
                    'leave': leave,
                    'mental_health_consequence':mental_health_consequence,
                    'phys_health_consequence':phys_health_consequence,
                    'coworkers': coworkers,
                    'supervisor': supervisor,
                    'mental_health_interview': mental_health_interview,
                    'phys_health_interview': phys_health_interview,
                    'mental_vs_physical': mental_vs_physical,
                    'obs_consequence':obs_consequence}
        features= pd.DataFrame(dataframe, index=[0])
        return features
    df_input = user_features()
    
 # Load fitted model
    model = joblib.load("pipe.joblib")

    st.write("")
    st.write("")
    st.write("")
    #Button style
    button = st.markdown("""
        <style>
        div.stButton > button{
        background-color: #0178e4;
        color:#ffffff;
        box-shadow: #094c66 4px 4px 0px;
        border-radius:8px 8px 8px 8px;
        transition : transform 200ms,
        box-shadow 200ms;
        }
         div.stButton > button:focus{
        background-color: #0178e4;
        color:#ffffff;
        box-shadow: #094c66 4px 4px 0px;
        border-radius:8px 8px 8px 8px;
        transition : transform 200ms,
        box-shadow 200ms;
        }
        div.stButton > button:active {
                transform : translateY(4px) translateX(4px);
                box-shadow : #0178e4 0px 0px 0px;
            }
        </style>""", unsafe_allow_html=True)
    predict = st.button('Predict')


    # Show the Outcome of the Model
   # Show the Outcome of the Model
    if predict:
        res = model.predict(df_input)


        if res == 0:
            st.write("")
            st.write("")

            col1,col2 = st.columns([0.1,1])
            # Show Check Sign
            with col1:
                st.image("https://cdn-icons-png.flaticon.com/512/709/709510.png",width =80)
                st.write('''
               <style>
                   img, svg {
                    'vertical-align': 'center';
                                }
               </style>
                ''', unsafe_allow_html=True)
            # Customer Unlikely To Churn
            with col2:
                st.markdown("""<h3 style="color:#0178e4;font-size:35px;">
                   No need for treatment
                    </h3>""",unsafe_allow_html = True)
        else:
                 st.write("")
                 st.write("")

                 col3,col4 = st.columns([0.1,1])
                 # Show Warning Message
                 with col4:
                     st.markdown("""<h3 style="color:#00284c;font-size:35px;">
                        Need a treatment
                         </h3>""",unsafe_allow_html = True)
