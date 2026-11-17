from datetime import datetime, timedelta
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split


# 생필품 소진일 예측용 GradientBoostingRegressor 학습 및 모델 아티팩트(.pkl) 저장 스크립트
def train_and_save_model(output_path: str = "consumable_predictor.pkl"):
    """합성 소비 데이터를 생성해 회귀 모델을 학습·검증하고 추론용 아티팩트로 저장"""
    # 재현 가능한 난수 생성을 위한 시드 고정
    np.random.seed(42)
    n_samples = 600

    # 학습 대상 생필품 카테고리 (물티슈 매핑용 '일회용 위생용품' 포함)
    categories = ["일회용 위생용품", "욕실용품", "세제", "구강용품"]

    # 카테고리별 기준 용량(base_amount), 기준 단위(base_unit), 기준 소진일(base_days), 샘플링 후보(choices)
    base_spec_map = {
        "일회용 위생용품": {
            "base_amount": 100,
            "base_unit": "매",
            "base_days": 30,
            "choices": [(20, "매"), (60, "매"), (100, "매"), (200, "매"), (5, "팩"), (10, "팩")],
        },
        "욕실용품": {
            "base_amount": 500,
            "base_unit": "ml",
            "base_days": 100,
            "choices": [(300, "ml"), (400, "ml"), (500, "ml"), (600, "ml"), (640, "ml"), (750, "ml"), (1, "L")],
        },
        "세제": {
            "base_amount": 500,
            "base_unit": "ml",
            "base_days": 80,
            "choices": [(500, "ml"), (600, "ml"), (800, "ml"), (840, "ml"), (1, "L"), (1500, "ml"), (2, "L")],
        },
        "구강용품": {
            "base_amount": 500,
            "base_unit": "g",
            "base_days": 60,
            "choices": [(120, "g"), (240, "g"), (360, "g"), (500, "g")],
        },
    }

    def normalize_amount(amount: float, unit: str, target_unit: str) -> float:
        """입력 수량 단위를 카테고리 기준 단위(ml, g 등)로 환산"""
        if unit == target_unit:
            return float(amount)
        if unit == "L" and target_unit == "ml":
            return float(amount) * 1000
        if unit == "ml" and target_unit == "L":
            return float(amount) / 1000
        if unit == "kg" and target_unit == "g":
            return float(amount) * 1000
        if unit == "g" and target_unit == "kg":
            return float(amount) / 1000
        # 팩/롤/개 등 비환산 단위는 그대로 반환
        return float(amount)

    # 카테고리별 기준 대비 용량 비율(amount_ratio)과 개인 소비 성향(user_lifestyle_score) 기반 학습 데이터 생성
    data = []
    for _ in range(n_samples):
        cat = np.random.choice(categories)
        spec = base_spec_map[cat]

        # 카테고리별 후보군에서 (수량, 단위) 무작위 선택 후 기준 용량 대비 비율 산출
        amount, unit = spec["choices"][np.random.choice(len(spec["choices"]))]
        normalized_amount = normalize_amount(amount, unit, spec["base_unit"])
        amount_ratio = normalized_amount / spec["base_amount"]
        scaled_base_days = spec["base_days"] * amount_ratio

        # 사용자 개인 성향 계수(평균 1.0, 표준편차 0.15) 및 관측 노이즈 반영
        user_trait = np.random.normal(1.0, 0.15)
        random_noise = np.random.normal(0, 3)
        actual_days = max(5, int(scaled_base_days * user_trait + random_noise))

        data.append(
            {
                "category": cat,
                "total_amount": amount,
                "unit": unit,
                "amount_ratio": amount_ratio,
                "user_lifestyle_score": round(user_trait, 2),
                "actual_days": actual_days,
            }
        )

    df = pd.DataFrame(data)

    def make_features(input_df: pd.DataFrame) -> pd.DataFrame:
        """원-핫 카테고리 피처, 카테고리별 스케일링 용량 피처, 라이프스타일 점수 피처 생성"""
        dummies = pd.get_dummies(input_df[["category"]], prefix="cat")
        for cat in categories:
            col_name = f"cat_{cat}"
            if col_name not in dummies:
                dummies[col_name] = 0
            # 선택된 카테고리에서만 amount_ratio가 활성화되도록 마스킹
            dummies[f"{cat}_scaled_amount"] = dummies[col_name] * input_df["amount_ratio"]
        dummies["user_lifestyle_score"] = input_df["user_lifestyle_score"]
        return dummies

    # 학습 피처 행렬(X) 및 타깃(y: actual_days) 구성
    X = make_features(df)
    y = df["actual_days"]

    # 학습/검증 데이터 분할 (80:20) 후 검증 성능 출력
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_valid)
    print(f"validation MAE : {mean_absolute_error(y_valid, y_pred):.2f}일")
    print(f"validation RMSE: {np.sqrt(mean_squared_error(y_valid, y_pred)):.2f}일")

    # 전체 데이터(X, y)로 최종 모델 재학습
    model_final = GradientBoostingRegressor(n_estimators=100, random_state=42)
    model_final.fit(X, y)

    # 추론 엔진이 로드할 모델, 컬럼 순서, 기준 스펙, 기본 폴백 스펙 패키징
    artifact = {
        "model": model_final,
        "columns": X.columns.tolist(),
        "base_specs": {
            k: {"base_amount": v["base_amount"], "base_unit": v["base_unit"]}
            for k, v in base_spec_map.items()
        },
        "default_spec": {"base_amount": 1, "base_unit": "개", "base_days": 30},
    }
    joblib.dump(artifact, output_path)
    print(f"저장 완료: {output_path}")


if __name__ == "__main__":
    train_and_save_model()