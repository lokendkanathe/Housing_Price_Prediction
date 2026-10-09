import streamlit as st 
import joblib
import numpy as np 
import os

st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏡",
    layout="centered"
)

@st.cache_resource
def load_model():
    model_path = os.path.join(
        "models",
        "house_price_model.pkl"
    )
    return joblib.load(model_path)

model = load_model()

city_encoder = joblib.load(
    os.path.join(
        "models", 
        "city_encoder.pkl"
    ))

st.sidebar.title("📊 Model Information")
st.sidebar.success("Model Loaded Successfully")

st.sidebar.write("""
### Features Used

- Bedrooms
- Bathrooms
- Living Area
- Lot Area
- Floors
- Waterfront
- View
- Condition
- Sqft Above
- Basement Sqft
- Year Built
""")

st.sidebar.info("""
Algorithm:
Linear Regression / Random Forest

Dataset:
USA Housing Dataset
""")

st.title("🏡 House Price Prediction")
st.markdown(
    "Enter house details below and click **Predict Price**."
)

bedrooms = st.number_input(
    "Bedrooms",
    min_value=1,
    max_value=10,
    value=3
)

bathrooms = st.number_input(
    "Bathrooms",
    min_value=1.0,
    max_value=10.0,
    value=2.0
)

sqft_living = st.number_input(
    "Living Area (sqft)",
    min_value=500,
    max_value=10000,
    value=2000
)

sqft_lot = st.number_input(
    "Lot Area (sqft)",
    min_value=500,
    max_value=50000,
    value=5000
)

floors = st.number_input(
    "Floors",
    min_value=1,
    max_value=5,
    value=2
)

waterfront = st.selectbox(
    "Waterfront",
    [0, 1]
)

view = st.slider(
    "View Rating",
    0,
    4,
    2
)

condition = st.slider(
    "Condition",
    1,
    5,
    3
)

sqft_above = st.number_input(
    "Sqft Above",
    min_value=500,
    max_value=10000,
    value=1500
)

sqft_basement = st.number_input(
    "Basement Sqft",
    min_value=0,
    max_value=5000,
    value=500
)

yr_built = st.number_input(
    "Year Built",
    min_value=1900,
    max_value=2025,
    value=2010
)

yr_renovated = st.number_input(
    "Year Renovated",
    min_value=0,
    max_value=2025,
    value=0
)

city = st.selectbox(
    "City",
    list(city_encoder.classes_)
)

if st.button("🔮 Predict Price"):

    city_encoded = city_encoder.transform(
    [city]
    )[0]

    house_age = max(0, 2026 - yr_built)

    features = np.array([[
    bedrooms,
    bathrooms,
    sqft_living,
    sqft_lot,
    floors,
    waterfront,
    view,
    condition,
    sqft_above,
    sqft_basement,
    yr_built,
    yr_renovated,
    house_age,
    city_encoded
    ]])

    prediction = model.predict(features)

    price = prediction[0]

    st.success(
        f"💰 Predicted Price: ${price:,.2f}"
    )

    if price < 300000:
        st.warning(
            "🏚️ Budget House"
        )

    elif price < 700000:
        st.info(
            "🏠 Mid-Range House"
        )

    else:
        st.error(
            "🏢 Luxury House"
        )
    
    st.subheader("Input Summary:")

    st.write(
    {
        "Bedrooms": bedrooms,
        "Bathrooms": bathrooms,
        "Living Area": sqft_living,
        "Lot Area": sqft_lot,
        "Floors": floors,
        "Waterfront": waterfront,
        "View": view,
        "Condition": condition,
        "Sqft Above": sqft_above,
        "Basement": sqft_basement,
        "Year Built": yr_built,
        "Year Renovated": yr_renovated,
        "House Age": house_age,
        "City": city
    }
)

# Footer 
st.markdown("___")
st.caption(
    "Bulit with python, Scikit-Learn and Streamlit"
)