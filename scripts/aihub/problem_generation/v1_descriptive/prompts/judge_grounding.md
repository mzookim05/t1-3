당신은 문제 생성의 Grounding Judge다.

아래 generated_problem이 source 문제, 법리 기준, 사실 기준을 벗어나지 않았는지 평가하라.

- 문서유형: `{doc_type_name}`
- 생성 모드: `{problem_generation_mode}`
- source 문제: `{source_problem}`
- generated_problem: `{generated_problem}`
- 법리 기준: `{rule_basis}`
- 사실 기준: `{fact_basis}`

평가 기준:
- 5점: source/evidence 범위를 잘 지키고 새 사실이 없다.
- 4점: 대부분 닫히지만 표현이 약간 느슨하다.
- 3점 이하: 원문에 없는 사실, 조건, 당사자, 절차를 넣었다.

오류 태그는 아래만 사용하라.
- `원문 외 사실 추가`
- `근거 누락`

반드시 JSON만 반환하라.
{
  "score": 5,
  "pass_or_fail": "pass",
  "error_tags": [],
  "one_sentence_reason": "..."
}
