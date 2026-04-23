다음 객관식 문제가 gold short answer 기준으로 정확히 `1개`의 정답만 갖는지 평가하라.

## 문서유형
{doc_type_name}

## gold short answer
{gold_short_answer}

## gold explanation
{gold_reference_explanation}

## generated problem
- stem: {generated_stem}
- A: {choice_a}
- B: {choice_b}
- C: {choice_c}
- D: {choice_d}
- correct_choice: {correct_choice}

판정 기준:
- 정답 선택지가 gold short answer와 직접 대응해야 한다
- 다른 선택지가 gold short answer와 사실상 같으면 hard fail 수준이다
- 선택지 중복, 정답 비유일, 오답이 정답 가능 상태를 강하게 본다

JSON으로만 답하라.
필드:
- score: 1~5 정수
- pass_or_fail: `pass` 또는 `fail`
- error_tags: 허용 태그만 사용
- one_sentence_reason

허용 태그:
- 정답 비유일
- 오답이 정답 가능
- 선택지 중복
- 정답 누설
