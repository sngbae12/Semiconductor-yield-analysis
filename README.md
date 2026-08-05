# 반도체 수율/불량률 분석 (Semiconductor Yield & Defect Rate Analysis)

UCI SECOM 데이터셋을 이용해 반도체 웨이퍼의 수율/불량률 계산 및 불량 예측 모델 구축 프로젝트

This project analyzes wafer yield and defect rate using the UCI SECOM dataset, and builds a defect prediction model.

---

## 📊 데이터 (Dataset)

- **출처 (Source)**: [UCI Machine Learning Repository - SECOM](https://archive.ics.uci.edu/dataset/179/secom)
- 1567개 웨이퍼 샘플, 590개 센서 측정값, Pass/Fail 라벨 (-1: 정상, 1: 불량)
- 1,567 wafer samples, 590 sensor measurements, Pass/Fail label (-1: normal, 1: defect)

## 📈 주요 결과 (Key Results)

| 항목 (Metric) | 값 (Value) |
|---|---|
| 전체 샘플 수 (Total samples) | 1,567 |
| 불량 수 (Defects) | 104 |
| 불량률 (Defect rate) | 약 6.6% (~6.6%) |
| 수율 (Yield) | 약 93.4% (~93.4%) |

### 모델 성능 (Model Performance, threshold=0.418)

| Class | Precision | Recall | F1-score |
|---|---|---|---|
| 0 (정상/Normal) | 0.95 | 0.93 | 0.94 |
| 1 (불량/Defect) | 0.26 | 0.33 | 0.29 |

threshold를 precision-recall curve의 교차점(0.418)으로 설정해, 정상/불량 판정의 균형점 발견

## 🔍 분석 과정 및 시행착오 (Methodology & Lessons Learned)

### 1. 기초 EDA (Basic EDA)
`Pass/Fail` 컬럼 분포를 확인해 불량률/수율을 계산. 극단적인 클래스 불균형(정상 94% vs 불량 6%)을 확인.


### 2. 전처리 (Preprocessing)
- 결측치 40% 이상인 컬럼 제거, 나머지는 중앙값으로 대체
- Variance Threshold로 분산이 낮은(정보량이 적은) 센서 제거


### 3. 첫 번째 시행착오: 데이터 유출 (First Pitfall: Data Leakage)

처음에는 SMOTE 오버샘플링을 `GridSearchCV` **밖에서** 먼저 적용한 뒤 교차검증을 수행. 그 결과 불량 recall이 1.00(또는 반대로 0.00)이라는 비정상적인 값 생성.

**원인**: SMOTE로 생성된 합성 샘플이 원본 샘플과 함께 서로 다른 CV fold(학습/검증)에 나뉘어 들어가면서, 모델이 사실상 "정답을 미리 본" 상태로 검증되는 데이터 유출 발생.

**해결**: `imblearn.pipeline.Pipeline`으로 SMOTE, 스케일링, 모델 학습을 하나로 묶어, 각 CV fold의 학습 데이터에만 SMOTE가 적용되도록 수정.


### 4. 두 번째 시행착오: 임계값(threshold)의 함정 (Second Pitfall: The Threshold Trap)

파이프라인 수정 후 정직한 결과를 얻었지만, 기본 threshold(0.5) 기준으로는 불량을 잡지 못함(recall=0.00). 이는 모델이 계산한 불량 확률이 대부분 50%를 넘지 못했기 때문.

Precision-recall curve를 그려 두 지표가 만나는 지점(threshold=0.418)을 찾아, 판정 기준을 낮춤으로써 정상/불량 탐지의 균형을 맞춤.

## 🛠 기술 스택 (Tech Stack)

Python, Pandas, Scikit-learn, XGBoost, imbalanced-learn (SMOTE), Matplotlib

## 💡 비즈니스적 해석 (Reflection)

반도체 양산 공정에서는 불량을 정상으로 오판하는 것이 정상을 불량으로 오판하는 것보다 훨씬 큰 비용(고객 클레임, 리콜)을 초래합니다. 따라서 accuracy보다 recall과 precision의 균형을 고려한 threshold 튜닝이 중요하다는 점을 확인했습니다.

