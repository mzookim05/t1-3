다음 객관식 문제가 source와 evidence에 grounded 되어 있는지 평가하라.

## 메타
- 문서유형: {doc_type_name}
- 생성 모드: {problem_generation_mode}

## source 문제
{source_problem}

## gold short answer
{gold_short_answer}

## gold explanation
{gold_reference_explanation}

## rule basis
{rule_basis}

## fact basis
{fact_basis}

## generated problem
- stem: {generated_stem}
- A: {choice_a}
- B: {choice_b}
- C: {choice_c}
- D: {choice_d}
- correct_choice: {correct_choice}

판정 기준:
- 원문에 없는 새 사실, 조문 번호, 기관명, 날짜, 조건이 들어가면 감점
- source 문제와 해설에서 벗어난 선택지가 많으면 감점

JSON으로만 답하라.
필드:
- score: 1~5 정수
- pass_or_fail: `pass` 또는 `fail`
- error_tags: 허용 태그만 사용
- one_sentence_reason

허용 태그:
- 원문 외 사실 추가
- 복수 쟁점 혼합
