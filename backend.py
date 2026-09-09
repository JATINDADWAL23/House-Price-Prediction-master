from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "ML_Model" / "data_set"
TRAIN_PATH = DATA_DIR / "train.csv"
CURRENT_YEAR = 2026
LAND_REFERENCE_YEAR = 2000
HOUSE_DEPRECIATION_RATE = 0.05
LAND_APPRECIATION_RATES = {
    "Urban / suburban": 0.085,
    "Rural / interior": 0.045,
    "High-growth belt": 0.125,
}
CONSTRUCTION_RATES_INR = {
    "minimum": 1500.0,
    "average": 1950.0,
    "maximum": 2400.0,
}
HOUSE_SHARE = 0.80
LAND_SHARE = 0.20

MONOTONIC_EFFECTS = {
    "OverallQual": 18000.0,
    "GrLivArea": 95.0,
    "YearBuilt": 900.0,
    "TotalBsmtSF": 45.0,
    "1stFlrSF": 55.0,
    "BedroomAbvGr": 9000.0,
    "FullBath": 14000.0,
    "GarageCars": 16000.0,
}


def load_training_data() -> pd.DataFrame:
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(f"Training data not found at {TRAIN_PATH}")
    return pd.read_csv(TRAIN_PATH)


def _build_pipeline(features: pd.DataFrame) -> Pipeline:
    numeric_features = features.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = features.select_dtypes(exclude=["number"]).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                GradientBoostingRegressor(
                    n_estimators=300,
                    learning_rate=0.04,
                    max_depth=3,
                    random_state=42,
                ),
            ),
        ]
    )


def train_model() -> tuple[Pipeline, pd.DataFrame, dict[str, Any]]:
    data = load_training_data()
    features = data.drop(columns=["SalePrice", "Id"], errors="ignore")
    target = data["SalePrice"]
    pipeline = _build_pipeline(features)
    pipeline.fit(features, target)

    defaults: dict[str, Any] = {}
    for column in features.columns:
        if pd.api.types.is_numeric_dtype(features[column]):
            defaults[column] = float(features[column].median())
        else:
            mode = features[column].mode(dropna=True)
            defaults[column] = mode.iloc[0] if not mode.empty else "NA"

    return pipeline, features, defaults


def predict_price(
    pipeline: Pipeline,
    feature_columns: pd.DataFrame,
    defaults: dict[str, Any],
    values: dict[str, Any],
) -> float:
    breakdown = predict_price_breakdown(
        pipeline, feature_columns, defaults, values
    )
    return breakdown["total_price"]


def predict_price_breakdown(
    pipeline: Pipeline,
    feature_columns: pd.DataFrame,
    defaults: dict[str, Any],
    values: dict[str, Any],
) -> dict[str, float]:
    baseline_row = defaults.copy()
    baseline_frame = pd.DataFrame([baseline_row], columns=feature_columns.columns)
    baseline_price = float(pipeline.predict(baseline_frame)[0])

    neighborhood_row = baseline_row.copy()
    if "Neighborhood" in values:
        neighborhood_row["Neighborhood"] = values["Neighborhood"]
    neighborhood_frame = pd.DataFrame(
        [neighborhood_row], columns=feature_columns.columns
    )
    neighborhood_effect = float(pipeline.predict(neighborhood_frame)[0]) - baseline_price

    monotonic_effect = 0.0
    for feature, dollars_per_unit in MONOTONIC_EFFECTS.items():
        if feature in values:
            change = float(values[feature]) - float(defaults[feature])
            monotonic_effect += change * dollars_per_unit

    model_price = max(0.0, baseline_price + neighborhood_effect + monotonic_effect)

    year_built = int(values.get("YearBuilt", defaults["YearBuilt"]))
    house_age = max(0, CURRENT_YEAR - year_built)
    house_base_value = model_price * HOUSE_SHARE
    house_value = house_base_value * (1 - HOUSE_DEPRECIATION_RATE) ** house_age

    land_model_row = defaults.copy()
    land_model_row.update(values)
    land_model_row["YearBuilt"] = CURRENT_YEAR
    land_model_frame = pd.DataFrame(
        [land_model_row], columns=feature_columns.columns
    )
    land_model_price = max(0.0, float(pipeline.predict(land_model_frame)[0]))

    land_area = max(0.0, float(values.get("LotArea", defaults.get("LotArea", 1))))
    default_land_area = max(1.0, float(defaults.get("LotArea", land_area)))
    land_area_factor = land_area / default_land_area
    land_type = str(values.get("LandType", "Urban / suburban"))
    land_rate = LAND_APPRECIATION_RATES.get(
        land_type, LAND_APPRECIATION_RATES["Urban / suburban"]
    )
    land_years = max(0, CURRENT_YEAR - LAND_REFERENCE_YEAR)
    land_base_value = land_model_price * LAND_SHARE * land_area_factor
    land_value = land_base_value * (1 + land_rate) ** land_years

    return {
        "total_price": max(0.0, house_value + land_value),
        "house_value": max(0.0, house_value),
        "land_value": max(0.0, land_value),
        "house_age": float(house_age),
        "land_rate": land_rate,
    }


def calculate_construction_cost(
    living_area: float,
    room_count: int,
    room_area: float,
    floors: int,
) -> dict[str, float | str]:
    construction_area = max(
        float(living_area), float(room_count) * float(room_area) * int(floors)
    )

    if (
        int(floors) >= 2
        and construction_area >= 2500
        and float(room_area) >= 144
    ):
        rate_type = "maximum"
    elif (
        int(room_count) <= 2
        and construction_area <= 1000
        and float(room_area) <= 120
    ):
        rate_type = "minimum"
    else:
        rate_type = "average"

    rate = CONSTRUCTION_RATES_INR[rate_type]
    return {
        "area": construction_area,
        "rate": rate,
        "rate_type": rate_type,
        "cost_inr": construction_area * rate,
    }