import pandas as pd
import numpy  as np 
import pickle
import streamlit as st 
from streamlit_option_menu import option_menu
from sklearn import preprocessing
import warnings
warnings.filterwarnings("ignore")
from scipy.stats import skew
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# streamlit part
st.set_page_config(layout="wide")
st.title(":violet[INDUSTRIAL COPPER MODEL]")
selected = option_menu(
    menu_title=None,
    options= ["ABOUT", "APPLICATION"],
    menu_icon=None,
    icons=None,
    orientation="horizontal"
    #styles={}
)
# About a project description using streamlit
if selected=="ABOUT":
    st.header("PROJECT DESCRIPTION")
    st.text(" The Industrial Copper Modeling project focuses on predicting the selling price and status (won or lost) in the industrial copper market using machine learning regression and classification algorithms.")
    st.text("NumPy: A library for numerical computations in Python.")
    st.text("Pandas: A library for data manipulation and analysis.")
    st.text("Scikit-learn: A machine learning library that provides various regression and classification algorithms.")
    st.text('Matplotlib: A plotting library for creating visualizations.')
    st.text("Seaborn: A data visualization library built on top of Matplotlib.")
    
# selling price prediction
if selected=="APPLICATION":
    select=st.radio("Select one ",(":green[SELLING PRIICE PREDICTION]",":green[STATUS PREDICTION]"))
    if select==":green[SELLING PRIICE PREDICTION]":
        col1,col2=st.columns(2)
        with col1:

            status=st.selectbox(":rainbow[SELECT STATUS]",("0","1","2","3","4","5","6","7","8"))
            st.write('Draft:0','Lost:1','Not lost for AM:2','Offerable:3','Offered:4','Revised:5','To be approved:6','Won:7','Wonderful:8')
            status=int(status)

            country=st.selectbox(":rainbow[SELECT COUNTRY]",("28","25","30","32","38","78","27","77","113","79","26","39","40","84","80","107","89"))
            country=float(country)

            item_type=st.selectbox(":rainbow[SELECT ITEM_TYPE]",("0","1","2","3","4","5","6"))
            st.write('IPL:0','Others:1','PL:2','S:3','SLAWR:4','W:5','WI:6')
            item_type=int(item_type)

            application=st.selectbox(":rainbow[SELECT APPLICATION]",("10","41","28","59","15","4","38","56","42","26","27","19","20","66","29","22","40","25","67","79","3","99","2","5","39","69","70","65","58","68"))
            application=float(application)

            product_ref=st.text_input(":rainbow[SELECT PRODUCT_REF] (MIN:611728 & MAX:1722207579)")
            
            item_date_year=st.selectbox(":rainbow[SELECT ITEM_DATE_YEAR]",("2021","2020"))
            item_date_year=int(item_date_year)

            item_date_month=st.selectbox(":rainbow[SELECT ITEM_DATE_MONTH]",("1",'2','3','4','5','6','7','8','9','10','11','12'))
            item_date_month=int(item_date_month)

            item_date_day=st.selectbox(":rainbow[SELECT ITEM_DATE_DAY]",('1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31'))
            item_date_day=int(item_date_day)

            delivery_date_year=st.selectbox(":rainbow[SELECT DELIVERY_DATE_YEAR]",("2022","2021","2020",'2019'))
            delivery_date_year=int(delivery_date_year)

            delivery_date_month=st.selectbox(":rainbow[SELECT DELIVERY_DATE_MONTH]",("1",'2','3','4','5','6','7','8','9','10','11','12'))
            delivery_date_month=int(delivery_date_month)

            delivery_date_day=st.selectbox(":rainbow[SELECT DELIVERY_DATE_DAY]",('1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31'))
            delivery_date_day=int(delivery_date_day)

        with col2:
            
            quantity_tons=st.text_input(":rainbow[SELECT QUANTITY_TONS](min:611728 & max: 1722207579)")
            customer=st.text_input(":rainbow[SELECT CUSTOMER](min:12458.0  & max:2147483647.0 )")
            thickness=st.text_input(":rainbow[SELECT THICKNESS](MIN:0.18 & MAX:2500.0)")
            width=st.text_input(":rainbow[SELECT WIDTH](MIN:1.0 & MAX:2990.0)")
            width=float(width) if width else None
            button1=st.button(":rainbow[PREDICT SELLING PRICE]")
            if button1:
                data={"country":country,"status": status,"item type":item_type,"application":application,"width": width,"quantity tons_log":quantity_tons,"customer_log":customer,"thickness_log": thickness,"product_ref_log":product_ref,
                      "item_date_year":item_date_year,"item_date_month": item_date_month,"item_date_day":item_date_day,"delivery_date_year":delivery_date_year,"delivery_date_month":delivery_date_month,"delivery_date_day":delivery_date_day}

                df=pd.DataFrame(data,index=[1])

                #convert log values

                df['quantity tons_log']=np.log(float(df["quantity tons_log"]))
                df['customer_log']=np.log(float(df["customer_log"]))
                df['thickness_log']=np.log(float(df["thickness_log"]))if thickness else None
                df['product_ref_log']=np.log(float(df["product_ref_log"]))


                # deserializing the model
                with open("c:/Users/ADMIN/Desktop/projects_coding/industrial/Regression model.pk1","rb") as f3:
                    R_model=pickle.load(f3)
                with open("c:/Users/ADMIN/Desktop/projects_coding/industrial/scalar.pkl","rb") as f4:
                    scalar=pickle.load(f4)
                new_data=scalar.transform(df)
                y_pred=R_model.predict(new_data)
                st.write('### :violet[Selling price is]')
                st.write(np.exp(y_pred))

            #status prediction
    if select==":green[STATUS PREDICTION]":
        col3,col4=st.columns(2)
        with col3:

            #Get input from user
            country=st.selectbox(":rainbow[SELECT COUNTRY]",("28","25","30","32","38","78","27","77","113","79","26","39","40","84","80","107","89"))
            country=float(country)

            item_type=st.selectbox(":rainbow[SELECT ITEM_TYPE]",("0","1","2","3","4","5","6"))
            st.write('IPL:0','Others:1','PL:2','S:3','SLAWR:4','W:5','WI:6')
            item_type=int(item_type)

            application=st.selectbox(":rainbow[SELECT APPLICATION]",("10","41","28","59","15","4","38","56","42","26","27","19","20","66","29","22","40","25","67","79","3","99","2","5","39","69","70","65","58","68"))
            application=float(application)

            product_ref=st.text_input(":rainbow[SELECT PRODUCT_REF] (MIN:611728 & MAX:1722207579)")
            
            item_date_year=st.selectbox(":rainbow[SELECT ITEM_DATE_YEAR]",("2021","2020"))
            item_date_year=int(item_date_year)

            item_date_month=st.selectbox(":rainbow[SELECT ITEM_DATE_MONTH]",("1",'2','3','4','5','6','7','8','9','10','11','12'))
            item_date_month=int(item_date_month)

            item_date_day=st.selectbox(":rainbow[SELECT ITEM_DATE_DAY]",('1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31'))
            item_date_day=int(item_date_day)

            delivery_date_year=st.selectbox(":rainbow[SELECT DELIVERY_DATE_YEAR]",("2022","2021","2020",'2019'))
            delivery_date_year=int(delivery_date_year)

            delivery_date_month=st.selectbox(":rainbow[SELECT DELIVERY_DATE_MONTH]",("1",'2','3','4','5','6','7','8','9','10','11','12'))
            delivery_date_month=int(delivery_date_month)

            delivery_date_day=st.selectbox(":rainbow[SELECT DELIVERY_DATE_DAY]",('1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31'))
            delivery_date_day=int(delivery_date_day)

        with col4:
            selling_price=st.text_input(":rainbow[ENTER SELLING PRICE](Min: 1.0  & max: 100001015.0)")
            quantity_tons=st.text_input(":rainbow[SELECT QUANTITY_TONS](min:611728 & max: 1722207579)")
            customer=st.text_input(":rainbow[SELECT CUSTOMER](min:12458.0  & max:2147483647.0 )")
            thickness=st.text_input(":rainbow[SELECT THICKNESS](MIN:0.18 & MAX:2500.0)")
            width=st.text_input(":rainbow[SELECT WIDTH](MIN:1.0 & MAX:2990.0)")
            width=float(width) if width else None
            button2=st.button(":rainbow[PREDICT STATUS]")
            if button2:
                data1={"country":country,"item type":item_type,"application":application,"width": width,"quantity tons_log":quantity_tons,"customer_log":customer,"thickness_log": thickness,"selling_price_log":selling_price,"product_ref_log":product_ref,
                      "item_date_year":item_date_year,"item_date_month": item_date_month,"item_date_day":item_date_day,"delivery_date_year":delivery_date_year,"delivery_date_month":delivery_date_month,"delivery_date_day":delivery_date_day}

                df1=pd.DataFrame(data1,index=[1])

                #convert log values
                df1['selling_price_log']=np.log(float(df1['selling_price_log']))
                df1['quantity tons_log']=np.log(float(df1["quantity tons_log"]))
                df1['customer_log']=np.log(float(df1["customer_log"]))
                df1['thickness_log']=np.log(float(df1["thickness_log"]))if thickness else None
                df1['product_ref_log']=np.log(float(df1["product_ref_log"]))if product_ref else None

                #Deserializing the model
                with open("c:/Users/ADMIN/Desktop/projects_coding/industrial/classification.pkl","rb") as f7:
                    c_model=pickle.load(f7)
                with open("c:/Users/ADMIN/Desktop/projects_coding/industrial/scalar2.pkl","rb")as f8:
                    scalar_=pickle.load(f8)
                new_data1=scalar_.transform(df1)
                y_pred1=c_model.predict(new_data1)
                if y_pred1==1:
                    st.write('### :violet[STATUS IS WON]')
                else:
                    st.write('### :violet[STATUS IS LOST]')
               



                
