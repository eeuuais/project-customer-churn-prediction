# H&M 고객 이탈(Churn) 예측 프로젝트

패션 리테일 H&M의 실제 거래 데이터로 **고객 이탈을 예측하는 머신러닝 모델**을 만들고, 피처 엔지니어링과 하이퍼파라미터 튜닝을 단계적으로 고도화하며 **거래 이력 데이터만으로 얻을 수 있는 예측 성능의 한계**를 직접 검증한 프로젝트입니다.

> 상세한 분석 과정과 근거는 별도 상세 보고서에서 다룰 예정입니다. (작성 예정)

---

## 1. 프로젝트 배경

이탈이 이미 일어난 뒤에는 마케팅으로 되돌릴 수 없습니다. 이 프로젝트는 "고객이 실제로 발길을 끊기 전에, 이탈 가능성이 높은 고객을 미리 걸러낼 수 있는가"라는 질문에서 출발했습니다.

- **데이터**: [H&M Personalized Fashion Dataset](https://www.kaggle.com/datasets/sohyunjun0401/h-and-m-personalized-fashion-data) (Kaggle) — 거래 내역 약 2,300만 건, 고객 메타정보, 상품 정보
- **분석 대상**: 가입 후 30일 이상 활동한 고객 약 114만 명
- **타겟 정의**: 자체적으로 cut-off 기준일을 통계적으로 도출하여, 기준일 이후 일정 기간 구매 이력이 없으면 이탈(churn=1)로 라벨링

---

## 2. 분석 프로세스

| 단계 | 노트북 | 내용 |
| :--- | :--- | :--- |
| 1 | [`01_target_definition_v3`](notebooks/01_target_definition_v3.ipynb) | 구매 간격 분포(75분위수=203일)를 근거로 cut-off일 도출 → 이탈 라벨 생성 → 데이터 누수 방지 원칙 하에 RFM·활동성·구매주기·다양성·트렌드·구매 시퀀스 등 파생 피처 설계 |
| 2 | [`02_eda_v3`](notebooks/02_eda_v3.ipynb) | 이탈/유지 고객 간 행동 패턴 차이 분석, 결측치 원인 분해, 다중공선성 및 중복 피처 탐지 |
| 3 | [`03_preprocessing_v3`](notebooks/03_preprocessing_v3.ipynb) | 중복 피처 제거, 결측치 처리 전략 적용, 인코딩, Train/Validation 분할 |
| 4 | [`04_baseline_comparison_v3`](notebooks/04_baseline_comparison_v3.ipynb) | LightGBM · XGBoost · CatBoost · RandomForest 4개 모델 베이스라인 성능 비교 |
| 5 | [`05_tuning_randomsearch_v3`](notebooks/05_tuning_randomsearch_v3.ipynb) | RandomizedSearchCV로 LightGBM·XGBoost 튜닝 (30% 샘플 탐색 → 전체 데이터 재학습) |
| 6 | [`06_tuning_optuna_v3`](notebooks/06_tuning_optuna_v3.ipynb) | Optuna 베이지안 최적화로 동일 모델 재탐색, RandomizedSearchCV 대비 비교 |
| 7 | [`07_final_comparison_v3`](notebooks/07_final_comparison_v3.ipynb) | 피처 버전(v1→v2→v3) 및 튜닝 방법 전체를 종합 비교, 최종 결론 및 비즈니스 활용 방안 도출 |

전체 과정에서 **데이터 누수(data leakage) 방지**를 최우선 원칙으로 두었습니다 — 모든 피처는 cut-off 기준일 이전 데이터만으로 생성했고, 상관관계가 비정상적으로 높은 피처는 별도로 검증했습니다.

---

## 3. 핵심 결과

### 모델 베이스라인 비교 (v3 피처 기준)

| 모델 | Accuracy | AUC-ROC | F1-Score | 학습 시간 |
| :--- | :---: | :---: | :---: | :---: |
| **CatBoost** | 0.7687 | **0.8486** | 0.7728 | 51.8초 |
| XGBoost | 0.7690 | 0.8482 | 0.7728 | 4.6초 |
| **LightGBM** | 0.7692 | 0.8480 | 0.7725 | **3.0초** |
| RandomForest | 0.7602 | 0.8368 | 0.7599 | 39.2초 |

CatBoost가 근소하게 1위지만, LightGBM은 성능 차이가 0.0006에 불과하면서 학습 시간은 17배 빠릅니다.

### 피처를 늘려도, 튜닝을 해도 성능은 정체됐습니다

| 실험 단계 | 시도 내용 | 최고 AUC |
| :--- | :--- | :---: |
| v1 베이스라인 | RFM 기반 16개 피처 | 0.8485 |
| v2 피처 추가 | 구매 트렌드/리듬 등 +6개 | 0.8486 |
| v3 피처 추가 | 마지막 5회 구매 간격 시퀀스 +5개 | 0.8486 |
| RandomizedSearchCV 튜닝 | LightGBM 0.8486 / XGBoost 0.8478 | 0.8486 |
| Optuna 베이지안 최적화 | LightGBM 0.8463 / XGBoost 0.8462 | 0.8463 |

피처를 3단계에 걸쳐 정교화하고, 하이퍼파라미터 탐색 방법을 3가지(Grid → Random → Bayesian)로 바꿔봐도 **AUC 0.848~0.849 구간을 벗어나지 못했습니다.** 특히 정교한 베이지안 최적화가 기본 파라미터보다 오히려 낮은 성능을 보인 것은, 튜닝 방법의 한계가 아니라 **거래 이력 데이터 자체가 담고 있는 정보량의 한계**임을 시사합니다.

---

## 4. 주요 인사이트

- **가장 강력한 이탈 신호는 `recency`(마지막 구매 후 경과일)** 였고, 그 다음은 구매 카테고리 다양성(`unique_product_groups`)이었습니다. EDA에서 발견한 상관관계 순위와 모델의 피처 중요도 순위가 대체로 일치해, 분석 방향의 일관성을 확인했습니다.
- 트렌드/구매 리듬/구매 시퀀스처럼 나중에 추가한 피처들은 이미 RFM 피처가 담고 있던 정보와 상당 부분 겹쳐, 추가적인 성능 개선으로 이어지지 않았습니다.
- **한계의 원인**: 지금 가진 데이터는 "구매 횟수·금액·간격·카테고리"뿐입니다. 이탈은 결과적으로 "마지막 구매 이후 아무 행동도 하지 않았다"는 것인데, 그 행동의 원인(클릭 로그, 세션 체류시간, 장바구니 이탈 등)을 설명할 데이터가 없어 모델이 학습할 수 있는 패턴 자체에 상한선이 생겼습니다.

## 5. 비즈니스 활용 방안

AUC 0.848 수준은 실무에서 세그먼트 기반 마케팅 액션으로 충분히 활용 가능한 수준입니다.

| 이탈 확률 구간 | 세그먼트 | 추천 액션 |
| :---: | :--- | :--- |
| 0.8 이상 | 이탈 임박 | 고가 쿠폰 + 개인화 추천 즉시 발송 |
| 0.6 ~ 0.8 | 이탈 위험 | 신상품 추천 이메일 + 등급 혜택 안내 |
| 0.3 ~ 0.6 | 관심 필요 | 재구매 타이밍에 맞춘 자동 알림 |
| 0.3 미만 | 안전 | 액션 없음 (마케팅 비용 절감) |

## 6. 향후 개선 방향

| 우선순위 | 방법 | 기대 효과 |
| :---: | :--- | :--- |
| 1순위 | 클릭/세션 등 행동 로그 데이터 추가 | AUC 0.90 이상 가능성 |
| 2순위 | LightGBM + XGBoost 앙상블 | AUC 0.001~0.003 소폭 개선 |
| 3순위 | LSTM/Transformer 기반 시퀀스 모델링 | 효과 불확실, 작업량 큼 |

---

## 7. 기술 스택

`Python` `pandas` `NumPy` `scikit-learn` `LightGBM` `XGBoost` `CatBoost` `Optuna` `matplotlib` `seaborn`

## 8. 프로젝트 구조

```
customer_churn/
├── notebooks/                      # 최종 분석 노트북 (v3)
│   ├── 01_target_definition_v3.ipynb
│   ├── 02_eda_v3.ipynb
│   ├── 03_preprocessing_v3.ipynb
│   ├── 04_baseline_comparison_v3.ipynb
│   ├── 05_tuning_randomsearch_v3.ipynb
│   ├── 06_tuning_optuna_v3.ipynb
│   └── 07_final_comparison_v3.ipynb
├── archives/                       # 이전 버전 실험 기록 (v1 RFM 베이스라인, v2 피처 확장)
├── reports/                        # 상세 보고서 (작성 예정)
└── README.md
```

`archives/`에는 이번 v3에 이르기까지 거쳐온 v1(RFM 베이스라인), v2(트렌드 피처 확장) 단계의 실험 노트북이 남아있습니다.

## 9. 상세 보고서

각 단계별 상세한 근거(cut-off 통계 검증, 피처별 설계 의도, 결측치 처리 논리, 모델 선택 기준 등)는 별도 상세 보고서로 정리 예정입니다. *(추후 업데이트)*
