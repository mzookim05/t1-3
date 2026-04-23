아래 정보를 바탕으로 `4지 단일정답 객관식` 문제를 생성하라.

## 메타
- 문서유형: {doc_type_name}
- source subset: {source_subset}
- 생성 모드: {problem_generation_mode}
- 문서유형 힌트: {doc_type_prompt_hint}

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

## 참고용 `v1` 서술형 문제
{problem_v1_generated_problem}

## 생성 지시
- 문제 본문은 `1∼2문장`으로 쓴다.
- `4지`를 모두 작성한다.
- 정답은 정확히 하나만 되게 만든다.
- distractor 유형은 아래 범주 안에서만 고른다.
  - `요건 일부 누락`
  - `효과/절차 혼동`
  - `범위 과대/과소`
- 문제 본문과 선택지 어디에도 기준 정답을 거의 그대로 노출하지 않는다.
- 하나의 쟁점만 묻게 만든다.
- 출력은 JSON 객체만 준다.
