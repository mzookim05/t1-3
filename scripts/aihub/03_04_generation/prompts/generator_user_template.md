다음 샘플의 `generated_explanation`만 작성하라.

- `DocuType`: `{doc_type_name}`
- 스타일: `{style_name}`
- `transformed_problem`: `{transformed_problem}`
- `short_answer`: `{short_answer}`
{long_answer_block}
- `issue`: `{issue}`
- `conclusion`: `{conclusion}`
- `rule_basis`: `{rule_basis}`
- `fact_basis`: `{fact_basis}`
- evidence sentences:
{evidence_bullets}

문장 규칙:
- 템플릿: `{template_name}`
- 목표 문장 수: `{target_sentences}`
- 목표 어절 범위: `{target_word_range}`
- 첫 문장은 왜 이 문제가 문제인지 드러내라.
- 중간 문장에서는 기준 또는 근거를 연결하라.
- 마지막 문장은 `short_answer`와 같은 결론으로 끝내라.

출력은 아래 JSON 객체 하나만 반환하라.
{
  "generated_explanation": "..."
}
