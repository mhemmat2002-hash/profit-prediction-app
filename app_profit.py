# app_profit_final.py
import streamlit as st
import numpy as np
import pandas as pd
import sys
import pickle
import joblib

st.set_page_config(page_title="پیش‌بینی سود سفارش", layout="centered")
st.title("💰 پیش‌بینی سود سفارش")

def load_model_encoders():
    """تلاش برای بارگذاری مدل با روش‌های مختلف"""
    
    # روش 1: با pickle
    try:
        with open('final_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('final_encoders.pkl', 'rb') as f:
            encoders = pickle.load(f)
        print("مدل با pickle بارگذاری شد")
        return model, encoders
    except Exception as e:
        print(f"pickle failed: {e}")
    
    # روش 2: با joblib
    try:
        model = joblib.load('final_model_joblib.pkl')
        encoders = joblib.load('final_encoders_joblib.pkl')
        print("مدل با joblib بارگذاری شد")
        return model, encoders
    except Exception as e:
        print(f"joblib failed: {e}")
    
    return None, None

model, encoders = load_model_encoders()
model_loaded = model is not None

if not model_loaded:
    st.error("""
    ❌ مدل بارگذاری نشد.
    
    لطفاً ابتدا فایل `save_model_onnx.py` را در محیط محلی اجرا کنید.
    """)
    st.stop()

st.success("✅ مدل با موفقیت بارگذاری شد")

# بقیه کد مانند قبل
st.markdown("---")

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
        
        st.info(f"📈 حاشیه سود تخمینی: {margin:.1f}%")
        
    except Exception as e:
        st.error(f"خطا در پیش‌بینی: {e}")