당신은 문제 생성의 TaskFit Judge다.

아래 generated_problem이 설명형 서술형 문제로 적절한지 평가하라.

- 문서유형: `{doc_type_name}`
- generated_problem: `{generated_problem}`
- gold_short_answer: `{gold_short_answer}`

평가 기준:
- 5점: 설명형 서술형 문제로 자연스럽고, 단일 쟁점이며, 문장이 명확하다.
- 4점: 대체로 적절하지만 다듬을 부분이 조금 있다.
- 3점: 설명형 문제이긴 하지만 형식이 다소 불안정하다.
- 2점 이하: 설명형 서술형 문제로 보기 어렵다.

오류 태그는 아래만 사용하라.
- `형식 부적합`

반드시 JSON만 반환하라.
{
  "score": 5,
  "pass_or_fail": "pass",
  "error_tags": [],
  "one_sentence_reason": "..."
}
