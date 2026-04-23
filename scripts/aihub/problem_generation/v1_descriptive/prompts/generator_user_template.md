다음 seed를 바탕으로 설명형 서술형 문제를 1개 생성하라.

- 문서유형: `{doc_type_name}`
- source subset: `{source_subset}`
- problem_generation_mode: `{problem_generation_mode}`
- 문서유형 힌트: `{doc_type_prompt_hint}`

현재 잠긴 source 문제:
{transformed_problem}

현재 잠긴 정답:
{short_answer}

현재 잠긴 teacher-side 해설:
{generated_explanation}

핵심 법리 또는 조문 기준:
{rule_basis}

핵심 사실 또는 적용 기준:
{fact_basis}

반드시 아래 원칙을 지켜라.

1. 같은 결론 방향으로 풀 수 있어야 한다.
2. 정답을 문제 안에 그대로 노출하지 않는다.
3. 원문에 없는 새 사실이나 새 조건을 넣지 않는다.
4. 설명형 서술형 문제 1개만 만든다.
5. 정답은 현재 잠긴 short answer로 채점 가능해야 한다.

반드시 JSON만 반환하라.
