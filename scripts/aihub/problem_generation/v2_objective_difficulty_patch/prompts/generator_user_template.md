아래 정보를 바탕으로 `4지 단일정답 객관식` 문제를 생성하라.

## 메타
- 문서유형: {doc_type_name}
- source subset: {source_subset}
- 생성 모드: {problem_generation_mode}
- 문서유형별 near-miss 지침: {doc_type_nearmiss_hint}

## source 문제
{transformed_problem}

## 기준 정답
{short_answer}

## 기준 해설
{generated_explanation}

## rule basis
{rule_basis}

## fact basis
{fact_basis}

## 참고용 기존 `v2` 객관식 문제
{problem_v2_generated_stem}

## 참고용 기존 `v2` 선택지
- A. {problem_v2_choice_a}
- B. {problem_v2_choice_b}
- C. {problem_v2_choice_c}
- D. {problem_v2_choice_d}
- 정답: {problem_v2_correct_choice}

## 생성 지시
- 문제 본문은 `1∼2문장`으로 쓴다.
- 반드시 하나의 쟁점만 묻는다.
- `4지`를 모두 작성한다.
- 정답은 정확히 하나만 되게 만든다.
- 정답은 기준 정답과 의미상 일치해야 하지만, 기준 정답 문장을 그대로 베끼지 않는다.
- 오답은 아래 범주를 사용하되, 기존 `v2`보다 더 가까운 near-miss로 만든다.
  - `요건 1개 누락`
  - `주체/기간/효과 1개 치환`
  - `전제조건 또는 예외 삭제`
  - `판단기준/사안적용 혼동`
  - `적용 범위 1개 과대/과소화`
- 오답 3개 중 최소 2개는 정답과 같은 legal anchor를 공유해야 한다.
- 오답은 전부 그럴듯해야 하지만, source와 기준 해설을 보면 명확히 틀려야 한다.
- 문제 본문과 선택지 어디에도 기준 정답을 거의 그대로 노출하지 않는다.
- 출력은 JSON 객체만 준다.
