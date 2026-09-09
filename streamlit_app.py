import streamlit as st

from backend import (
    CONSTRUCTION_RATES_INR,
    LAND_APPRECIATION_RATES,
    calculate_construction_cost,
    load_training_data,
    predict_price_breakdown,
    train_model,
)


USD_TO_INR = 91.0


def format_inr(amount: float) -> str:
    rounded_amount = int(round(amount))
    amount_text = str(rounded_amount)
    if len(amount_text) <= 3:
        return f"₹{amount_text}"
    last_three = amount_text[-3:]
    remaining = amount_text[:-3]
    groups = []
    while remaining:
        groups.insert(0, remaining[-2:])
        remaining = remaining[:-2]
    return f"₹{','.join(groups)},{last_three}"


st.set_page_config(
    page_title="House price predictor",
    page_icon=":material/home:",
    layout="wide",
)


@st.cache_resource
def get_model():
    return train_model()


st.title("House price predictor")
st.write("Estimate the sale price of an Ames, Iowa home using the supplied dataset.")

with st.spinner("Preparing the prediction model..."):
    model, feature_columns, defaults = get_model()

training_data = load_training_data()
st.caption(
    f"Model trained on {len(training_data):,} homes and "
    f"{len(feature_columns.columns)} property features."
)

with st.form("house_details"):
    st.subheader("Property details")
    first, second, third = st.columns(3)

    with first:
        overall_quality = st.slider(
            "Overall quality",
            min_value=1,
            max_value=10,
            value=int(defaults["OverallQual"]),
            help="Overall material and finish quality, from 1 (poor) to 10 (excellent).",
        )
        living_area = st.number_input(
            "Above-ground living area (sq ft)",
            min_value=200,
            max_value=6000,
            value=int(defaults["GrLivArea"]),
            step=50,
        )
        neighborhood_options = sorted(training_data["Neighborhood"].dropna().unique())
        neighborhood = st.selectbox(
            "Neighborhood",
            neighborhood_options,
            index=neighborhood_options.index(defaults["Neighborhood"]),
        )
        land_type = st.selectbox(
            "Land type",
            list(LAND_APPRECIATION_RATES),
            help="Land appreciates while the house structure depreciates.",
        )
        room_count = st.number_input(
            "Total rooms",
            min_value=1,
            max_value=30,
            value=max(2, int(defaults["BedroomAbvGr"]) + 2),
            step=1,
        )

    with second:
        year_built = st.number_input(
            "Year built",
            min_value=1872,
            max_value=2026,
            value=int(defaults["YearBuilt"]),
            step=1,
        )
        basement_area = st.number_input(
            "Total basement area (sq ft)",
            min_value=0,
            max_value=4000,
            value=int(defaults["TotalBsmtSF"]),
            step=50,
        )
        first_floor_area = st.number_input(
            "First-floor area (sq ft)",
            min_value=200,
            max_value=4000,
            value=int(defaults["1stFlrSF"]),
            step=50,
        )
        land_area = st.number_input(
            "Land area (sq ft)",
            min_value=100,
            max_value=500000,
            value=int(defaults["LotArea"]),
            step=100,
        )
        room_area = st.number_input(
            "Average room area (sq ft)",
            min_value=80,
            max_value=500,
            value=120,
            step=12,
            help="A standard 10 ft x 12 ft room is 120 sq ft; 12 ft x 12 ft is 144 sq ft.",
        )

    with third:
        bedrooms = st.number_input(
            "Bedrooms",
            min_value=0,
            max_value=10,
            value=int(defaults["BedroomAbvGr"]),
            step=1,
        )
        bathrooms = st.number_input(
            "Full bathrooms",
            min_value=0,
            max_value=6,
            value=int(defaults["FullBath"]),
            step=1,
        )
        garage_cars = st.number_input(
            "Garage capacity (cars)",
            min_value=0,
            max_value=6,
            value=int(defaults["GarageCars"]),
            step=1,
        )
        floors = st.number_input(
            "Number of floors",
            min_value=1,
            max_value=10,
            value=1,
            step=1,
            help="Used to estimate room-wise construction area and select the construction rate.",
        )

    submitted = st.form_submit_button(
        "Predict price",
        type="primary",
        icon=":material/price_check:",
        width="stretch",
    )

if submitted:
    values = {
        "OverallQual": overall_quality,
        "GrLivArea": living_area,
        "Neighborhood": neighborhood,
        "LandType": land_type,
        "LotArea": land_area,
        "Floors": floors,
        "YearBuilt": year_built,
        "TotalBsmtSF": basement_area,
        "1stFlrSF": first_floor_area,
        "BedroomAbvGr": bedrooms,
        "FullBath": bathrooms,
        "GarageCars": garage_cars,
    }
    breakdown = predict_price_breakdown(model, feature_columns, defaults, values)
    construction = calculate_construction_cost(
        living_area, room_count, room_area, floors
    )
    estimated_inr = breakdown["total_price"] * USD_TO_INR
    house_inr = breakdown["house_value"] * USD_TO_INR
    land_inr = breakdown["land_value"] * USD_TO_INR
    st.success(f"Estimated total property value: {format_inr(estimated_inr)}")
    house_column, land_column = st.columns(2)
    house_column.metric("House value after depreciation", format_inr(house_inr))
    land_column.metric("Land value after appreciation", format_inr(land_inr))
    st.metric(
        "Room-wise construction estimate",
        format_inr(float(construction["cost_inr"])),
    )
    st.caption(
        f"Approximate conversion at ₹{USD_TO_INR:.0f} per US dollar. "
        f"House age: {int(breakdown['house_age'])} years. "
        f"Land growth: {breakdown['land_rate'] * 100:.1f}% annually. "
        f"Construction: {int(construction['area']):,} sq ft at "
        f"₹{int(construction['rate']):,}/sq ft ({construction['rate_type']} rate). "
        "This estimate is not a professional appraisal."
    )