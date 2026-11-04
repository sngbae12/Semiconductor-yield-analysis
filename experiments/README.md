# 다섯 모델 실제 데이터 비교

공식 UCI ZIP과 동일한 SHA-256을 가진 업로드 데이터를 사용해 실제 학습을 완료했습니다. 데이터·라벨을 변경하지 않았습니다. 데이터 ZIP과 학습된 바이너리 모델은 Git에 포함하지 않습니다.

## 결과

중첩 CV AP 기준 XGBoost(0.1997 ± 0.0709)를 선택했습니다. 검증 결과 역시 F1이 가장 높았지만 이 결과를 이용해 모델 계열을 다시 선택하지 않았습니다. [모델별 비교표](run-001/comparison.md), [전체 보고서](run-001/report.json), [기존 방식 재평가](run-001/baseline_comparison.json)를 확인할 수 있습니다.

같은 검증 데이터의 기존 방식 대비 AP·F1·Precision·Accuracy·정상 오탐은 개선됐으나 Recall은 38.1%에서 23.8%로 낮아졌습니다. 불량을 더 찾으면서 오탐도 줄이는 목표를 동시에 달성하지 못했습니다. 검증은 임계값 조정 데이터이며 독립 평가 데이터가 아닙니다.

## 사전에 고정한 비교 절차

1. 기존 seed=42 학습 939건만 탐색에 사용합니다. 검증 314건은 임계값 선택, 기존 테스트 314건은 이번 탐색·평가에서 모두 제외합니다. 분할 행 번호는 `split_rows.json`에 기록했습니다.
2. XGBoost·CatBoost·LightGBM·Random Forest·Logistic Regression 각각 24개 설정을 동일한 3-fold AP로 탐색합니다. 무보정·학습 fold 기반 balanced 가중치·SMOTE를 각각 8개씩 비교하고 SMOTE와 가중치를 중복 적용하지 않습니다.
3. 결측률 20/40%, 평균/중앙값 대체, 분산 0/0.05, ANOVA 25/50/100% 변수 선택과 모델 하이퍼파라미터를 탐색합니다. 모든 변환·변수 선택·재표본화·가중치는 학습 fold 안에서 학습합니다. 모델별 후보 목록은 `candidates()`로 재현할 수 있습니다. 유한한 탐색 예산이며 전역 최적을 보장하지 않습니다.
4. 외부 3-fold(seed=2026)에서 내부 3-fold(seed=42) 탐색을 반복합니다. 외부 fold의 라벨은 내부 모델·임계값 결정에 사용하지 않습니다. 외부 평가 임계값은 내부 OOF 확률의 F1으로 결정합니다. 내부 OOF는 하이퍼파라미터 선택 후 생성되어 자체 점수는 독립 추정치가 아닙니다.
5. **외부 평균 AP 최대인 모델 계열을 검증 확인 전에 고정합니다.** 이번 실행에서는 모든 모델의 외부 fold 평가가 끝난 뒤, Logistic Regression의 마지막 전체 학습 탐색을 기다리는 동안 CV만으로 XGBoost를 먼저 고정하고 재현 실행했습니다. `selection_before_validation.json`에 해당 시점의 전체 CV AP를 기록했습니다. 이후 전체 실행이 완료된 `selected_family.json`의 선택과 일치합니다.
6. 각 계열은 학습 전체 내부 CV로 다시 튜닝합니다. 모델별 검증 지표는 표에 기록하되 계열을 다시 선택하지 않습니다. 선택 모델의 검증 F1 최대 임계값을 사용하고 정확히 동률이면 높은 임계값으로 오탐을 줄입니다. 모든 고유 임계값의 Precision·Recall·F1·오탐 수·오탐률을 저장해 tradeoff를 공개합니다.
7. 5개 계열 중 최고를 고르는 선택 편향은 중첩 CV 보고에도 남습니다. 검증 데이터도 과거 열람 이력이 있으므로 어느 지표도 새 외부 데이터 성능으로 부르지 않습니다.

모델당 중첩 CV 216회 + 전체 학습 탐색 72회 + 내부 OOF 9회 + refit 4회, 다섯 모델 합계 약 1,505회 fit입니다. 기존 방식은 같은 외부 분할에서 원래 27개 조합을 재탐색해 별도 비교했습니다. 작업자 2개, 모델 내부 스레드 1개를 사용했습니다.

## 실행

저장소 루트에서 실행합니다. 전체 실험은 일부 설정이 오래 걸려 수십 분 이상 소요될 수 있습니다. 출력 폴더는 기존 폴더를 덮어쓰지 않도록 새 경로를 사용하세요.

```bash
source .venv/bin/activate
python -m pip install -r requirements-experiments.txt
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m experiments.compare_models --jobs 2 --output-dir experiments/run-002
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m experiments.baseline_cv --output-dir experiments/run-002
python -m experiments.export_final --run-dir experiments/run-002 --destination new_config.json
python -m unittest discover -s tests -v
```

`export_final`은 완료된 결과에서 선택된 설정을 그대로 내보내며 모델을 재선택하지 않습니다. 프로젝트의 `final_config.json`은 이번 실행의 고정 결과입니다. 새 실험 결과를 최종 프로젝트에 반영할 경우 설정·최종 모델 구현·문서를 함께 검토해야 합니다. 기본 최종 실행은 `python final_model.py`이며 탐색 코드를 호출하지 않습니다.

## 수렴과 재현 한계

Logistic Regression의 전체 학습 탐색 중 `Liblinear failed to converge` 경고가 발생했습니다. 해당 설정의 점수를 조용히 삭제하거나 성공으로 위장하지 않았으며 정확한 후보/fold는 로그에서 식별되지 않습니다. 선택된 Logistic Regression 설정은 별도 재학습해 수렴 여부를 확인했습니다. [수렴 기록](run-001/runtime_notes.json)과 [실행 로그](run-001/execution.log)를 함께 보세요. 이 경고로 Logistic Regression의 탐색 공간 전체가 충분히 최적화됐다고 주장할 수 없습니다. 최종 XGBoost 재현과 모델 저장·복원 검증은 통과했습니다.

최종 실행 재현 지표는 `run-001/final_reproduction.json`에 있습니다. 기존 원본은 커밋 `6d41e85`와 로컬 `archive/secom-before-model-comparison` 브랜치에서 복구할 수 있습니다. 기존 소스·테스트·노트북·results는 수정하지 않았습니다. 합성 데이터 테스트는 코드 동작만 검증하며 성능의 근거로 쓰지 않습니다.
