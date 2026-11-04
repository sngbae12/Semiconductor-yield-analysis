# UCI SECOM 반도체 불량 예측 — 모델 비교와 개선 XGBoost

공식 UCI SECOM 데이터로 다섯 모델을 비교한 뒤 **중첩 교차검증 AP가 가장 높은 XGBoost 하나**를 최종 실행 모델로 사용합니다. 독립 평가 데이터는 없습니다. 과거 테스트 314건은 이번 모델 선택·튜닝·평가에서 제외했습니다.

**결과는 부분 개선입니다.** 같은 검증 데이터에서 기존 방식보다 AP·Precision·F1·Accuracy가 높아지고 정상 오탐이 줄었지만, 불량 Recall은 낮아졌습니다. “불량을 더 많이 찾으면서 오탐도 줄였다”는 목표의 동시 달성은 확인하지 못했습니다.

## 프로젝트 수행 요약

> 기존 XGBoost의 낮은 불량 재현율을 개선하고자 학습 방식과 트리 구조가 다른 CatBoost 등 5개 모델을 비교했으며, 중첩 교차검증 AP가 가장 높은 개선 XGBoost를 선정해 동일 검증 데이터에서 F1을 0.2353에서 0.2941로 높였지만, 재현율은 낮아져 불량 탐지와 정상 오탐 사이의 상충관계를 확인했습니다.

## 실행

Python 3.12, Linux x86_64 CPU에서 실제 학습·평가했습니다.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python final_model.py
```

공식 ZIP을 자동 다운로드하며 SHA-256·CRC·스키마를 검증합니다. 네트워크에서 UCI 접근이 차단되면 [공식 ZIP](https://archive.ics.uci.edu/static/public/179/secom.zip)을 다운로드해 **압축을 풀지 않고 `data/secom.zip`**으로 저장하세요. 업로드된 원본 ZIP의 검증 후 실제 실험을 수행했습니다. 데이터 1,567건, 센서 590개, 정상 1,463건, 불량 104건입니다.

명령은 고정된 모델·파라미터·임계값을 사용하며 실험 탐색을 반복하지 않습니다. 학습 939건으로 적합하고 검증 314건의 결과를 출력합니다. `final_results/`에 지표 JSON, 예측 CSV, 혼동행렬 PNG와 재사용 가능한 `model.joblib`이 생성됩니다. 이 지표는 임계값을 정한 검증 세트의 재현 결과이며 새 독립 평가가 아닙니다. 모델 파일은 다시 생성할 수 있어 Git에 포함하지 않습니다.

## 실제 비교 결과

| 모델 | CV AP 평균 ± SD | 검증 AP | Accuracy | Precision | Recall | F1 | 정상 오탐 |
|---|---:|---:|---:|---:|---:|---:|---:|
| XGBoost | 0.1997 ± 0.0709 | 0.1643 | 92.36% | 38.46% | 23.81% | 0.2941 | 8 |
| CatBoost | 0.1794 ± 0.0221 | 0.1396 | 90.45% | 26.32% | 23.81% | 0.2500 | 14 |
| LightGBM | 0.1780 ± 0.0684 | 0.1436 | 85.35% | 16.22% | 28.57% | 0.2069 | 31 |
| Random Forest | 0.1767 ± 0.0567 | 0.1367 | 67.20% | 12.04% | 61.90% | 0.2016 | 95 |
| Logistic Regression | 0.1404 ± 0.0460 | 0.1329 | 91.72% | 27.27% | 14.29% | 0.1875 | 8 |

CV AP는 학습 데이터 내부 외부 3-fold의 AP 평균 ± 표준편차입니다. 나머지 지표는 **동일한 검증 314건에서 모델별 F1 최대 임계값**을 적용한 값입니다. 이 표의 검증 성능으로 모델 계열을 다시 선택하지 않았습니다. 임계값 조정 후 검증 성능은 낙관적이며 독립 성능 추정치가 아닙니다.

## 최종 모델과 기존 방식의 같은 조건 비교

| 지표 | 기존 XGBoost + SMOTE | 최종 XGBoost |
|---|---:|---:|
| 중첩 CV AP 평균 | 0.1486 | 0.1997 |
| 검증 AP | 0.1349 | 0.1643 |
| 검증 Accuracy | 83.44% | 92.36% |
| 검증 불량 Precision | 17.02% | 38.46% |
| 검증 불량 Recall | 38.10% | 23.81% |
| 검증 불량 F1 | 0.2353 | 0.2941 |
| 검증 정상 오탐 / 정상 수 | 39 / 293 | 8 / 293 |
| 검증 오탐률 | 13.31% | 2.73% |
| 검증 탐지한 불량 / 불량 수 | 8 / 21 | 5 / 21 |

기존 방식도 동일한 외부 분할에서 원래 27개 조합을 내부 CV로 다시 탐색했습니다. 기존의 최고 탐색 CV AP 0.2446은 중첩 CV가 아니므로 위의 외부 CV 평균과 직접 비교하지 않습니다. 검증의 기존 모델은 원래 선택된 파라미터를 재학습해 재현했습니다.

최종 임계값은 **0.2143448293209076**, 판정은 `불량 확률 >= 임계값`입니다. 검증 혼동행렬은 행=실제, 열=예측, 순서=정상·불량입니다.

```text
[[285,  8],
 [ 16,  5]]
```

검증 확률의 모든 고유 임계값을 검사해도 정상 오탐을 기존 39건 이하로 유지하면서 기존 불량 8건 이상을 탐지하는 점은 없었습니다. 따라서 임계값만 바꾸어 두 목표를 모두 달성했다고 주장하지 않습니다.

과거 테스트 결과(기본 임계값 Accuracy 89.17%, Recall·Precision·F1 0.1905, AP 0.1270 / 조정 F1 0.1695)는 [기존 결과](results/metrics.json)에 보존했습니다. 이는 **참고 기록**이며, 이번 검증 결과와 서로 다른 데이터에서 계산된 개선률을 제시하지 않습니다.

## 최종 학습 구성

- 학습 fold 안에서 결측률 20% 초과 센서 제거 → 평균 대체 → 상수 제거 → 표준화. ANOVA 선택은 100% 유지입니다.
- 최종 모델에서는 SMOTE와 불량 가중치를 사용하지 않습니다. 이 선택도 학습 CV에서 이루어졌습니다.
- XGBoost: 깊이 5, 트리 300개, 학습률 0.08, `min_child_weight=5`, `reg_lambda=10`, `colsample_bytree=0.6`, seed=42, CPU `hist`.
- 검증 F1이 최대인 임계값을 사용하며 동률이면 높은 임계값을 선택해 오탐을 줄입니다. 임계값 결정 후 재학습하지 않습니다.
- 재현 가능한 전체 설정은 [final_config.json](final_config.json)에 고정했습니다.

## 실험 재현과 근거 파일

최종 실행에 CatBoost·LightGBM이나 탐색 코드는 필요하지 않습니다. 모든 모델 비교를 재실행할 때만 추가 의존성을 설치합니다.

```bash
python -m pip install -r requirements-experiments.txt
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m experiments.compare_models --jobs 2 --output-dir experiments/run-002
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m experiments.baseline_cv --output-dir experiments/run-002
python -m unittest discover -s tests -v
```

다섯 모델 × 24개 설정을 비교했으며, 각 모델에서 무보정·balanced 가중치·SMOTE를 각각 8개 설정씩 탐색했습니다. 결측률·대체 방법·분산 필터·변수 선택과 모델 하이퍼파라미터를 함께 비교했습니다. 동일한 학습 데이터와 CV 분할, 동일한 탐색 예산을 사용했고 모든 전처리·변수 선택·재표본화는 fold 안에서 학습했습니다. 중첩 CV 및 전체 학습 재탐색에 약 1,505회 fit, 별도로 기존 방식 비교를 실행했습니다. 계산량이 큰 설정이 있어 전체 실험은 수십 분 이상 걸릴 수 있습니다.

- [실험 방법](experiments/README.md)
- [학습 중첩 CV 비교](experiments/run-001/training_comparison.csv)
- [검증 성능 비교](experiments/run-001/validation_comparison.csv)
- [기존 방식 재평가](experiments/run-001/baseline_comparison.json)
- [최종 실험 보고서](experiments/run-001/report.json)
- `experiments/run-001/*_search.csv`: 모든 탐색 결과
- `experiments/run-001/*_nested.json`: 모델별 외부 fold 지표
- `experiments/run-001/*_thresholds.csv`: 검증 임계값별 Recall·Precision·F1·오탐률·오탐 수

테스트 20개는 전처리의 학습 범위, 분할, SMOTE·가중치의 분리, 다섯 모델의 학습 연결, 임계값·확률 저장, 최종 모델 저장·복원 등을 검증합니다. 합성 데이터 테스트를 실제 성능의 근거로 사용하지 않습니다.

## 해석상 한계와 원본 복구

불량 학습 표본은 62건, 검증 불량은 21건뿐입니다. 3-fold AP 변동이 크고, 여러 모델 중 최고를 고르는 선택 편향도 남습니다. 내부 탐색 후 OOF 확률로 임계값을 정하므로 내부 OOF 자체는 독립 추정치가 아닙니다. 과거 검증·테스트를 이미 열람한 이력과 무작위 분할 때문에 향후 생산 시점·배치의 일반화나 사업상 비용 개선을 보장하지 않습니다. 추가 외부 데이터 평가가 필요합니다.

기존 코드·노트북·결과는 보존했습니다. `secom_analysis.py`, `secom_yield_analysis.ipynb`, `results/`는 과거 실행 흐름이며, 기존 의존성은 `requirements-legacy.txt`입니다. 원본 전체는 기준 커밋 `6d41e85b7095c9779010b389c0cde25f5b7238a0`에서 복구할 수 있습니다. 새 작업은 별도 브랜치 `improve/secom-model-comparison`에서 진행하며 기존 main을 변경하지 않습니다.
