# app_profit.py
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import sklearn

st.set_page_config(page_title="پیش‌بینی سود سفارش", layout="centered")
st.title("💰 پیش‌بینی سود سفارش")

@st.cache_resource
def load_models():
    model = joblib.load('final_model.pkl')
    encoders = joblib.load('final_encoders.pkl')
    return model, encoders

try:
    model, encoders = load_models()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"خطا در بارگذاری مدل: {e}")

st.markdown("---")

if model_loaded:
    col1, col2 = st.columns(2)
    
    with col1:
        sales = st.number_input("💰 فروش (دلار)", min_value=0, value=500)
        quantity = st.number_input("📦 تعداد", min_value=1, value=5)
        discount = st.number_input("🎯 تخفیف (0.1 = 10%)", min_value=0.0, max_value=1.0, value=0.1, step=0.05)
        shipping = st.number_input("🚚 هزینه ارسال (دلار)", min_value=0, value=50)
    
    with col2:
        category = st.selectbox("📂 دسته محصول", ['Technology', 'Furniture', 'Office Supplies'])
        sub_category = st.selectbox("📁 زیردسته", ['Phones', 'Chairs', 'Tables', 'Storage', 'Accessories', 
                                                   'Binders', 'Paper', 'Art', 'Machines', 'Supplies'])
        segment = st.selectbox("👥 بخش مشتری", ['Consumer', 'Corporate', 'Home Office'])
        ship_mode = st.selectbox("✈️ روش ارسال", ['Standard Class', 'Second Class', 'First Class', 'Same Day'])
    
    if st.button("🔮 پیش‌بینی سود"):
        try:
            cat_enc = encoders['Category'].transform([category])[0]
            sub_enc = encoders['Sub-Category'].transform([sub_category])[0]
            seg_enc = encoders['Segment'].transform([segment])[0]
            ship_enc = encoders['Ship Mode'].transform([ship_mode])[0]
            
            log_sales = np.log1p(sales)
            input_data = np.array([[log_sales, quantity, discount, shipping,
                                   cat_enc, sub_enc, seg_enc, ship_enc]])
            margin = model.predict(input_data)[0]
            profit = (margin / 100) * sales
            
            st.markdown("---")
            if profit < 0:
                st.error(f"💰 سود پیش‌بینی شده: **{profit:,.0f} دلار**")
                st.warning("⚠️ هشدار: این سفارش پیش‌بینی می‌شود ضررده باشد!")
            else:
                st.success(f"💰 سود پیش‌بینی شده: **{profit:,.0f} دلار**")
                st.balloons()
            
            st.info(f"📈 حاشیه سود: {margin:.1f}%")
        except Exception as e:
            st.error(f"خطا در پیش‌بینی: {e}")
