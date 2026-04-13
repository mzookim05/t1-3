다음 샘플의 `generated_explanation`만 작성하라.

- `DocuType`: `{doc_type_name}`
- 스타일: `{style_name}`
- `transformed_problem`: `{transformed_problem}`
- `short_answer`: `{short_answer}`
{long_answer_block}
- `law_question_mode`: `{law_question_mode}`
- `law_generation_hint`: `{law_generation_hint}`
- `law_closing_target`: `{law_closing_target}`
- `law_required_terms`: `{law_required_terms}`
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
- `law_generation_hint`가 비어 있지 않으면 그 지시를 우선 따르라.
- `law_question_mode`가 `definition`이면 마지막 문장에서 `law_closing_target`를 축약하지 말고, `law_required_terms`가 `없음`이 아니면 해당 표현을 모두 포함하라.
- `law_question_mode`가 `scope` 또는 `requirement`이면 마지막 문장은 `law_closing_target`와 같은 범위·요건 구성을 유지하라.

출력은 아래 JSON 객체 하나만 반환하라.
{
  "generated_explanation": "..."
}
