다음 객관식 문제의 오답 선택지 품질과 문제 형식 적합성을 평가하라.

## 문서유형
{doc_type_name}

## 생성 모드
{problem_generation_mode}

## source 문제
{source_problem}

## gold short answer
{gold_short_answer}

## generated problem
- stem: {generated_stem}
- A: {choice_a}
- B: {choice_b}
- C: {choice_c}
- D: {choice_d}
- correct_choice: {correct_choice}

판정 기준:
- 오답은 그럴듯하지만 최종적으로는 명확히 틀려야 한다
- 문제 본문이 정답을 과하게 노출하면 감점
- 하나의 stem에 둘 이상의 질문이 섞이면 감점
- 선택지 길이와 형식이 너무 불균형하면 감점

JSON으로만 답하라.
필드:
- score: 1~5 정수
- pass_or_fail: `pass` 또는 `fail`
- error_tags: 허용 태그만 사용
- one_sentence_reason

허용 태그:
- 정답 누설
- 복수 쟁점 혼합
- 형식 부적합
- 오답이 정답 가능
