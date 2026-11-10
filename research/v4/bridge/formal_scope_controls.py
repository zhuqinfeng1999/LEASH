"""User-selected planner/method pairs; historical evidence is never deleted."""

MODELS = ('gpt-6-astra', 'gpt-6.1-sol', 'gpt-6-luna')
METHODS = ('B0', 'Ours')


def method_of(row):
    return 'Ours' if row.get('method') == 'Ours-conf' else row.get('method')


def selected_pairs(plan):
    authorization = plan.get('authorization', {})
    methods = set(authorization.get('enabled_methods', METHODS))
    if not methods.issubset(METHODS):
        raise ValueError('Unknown enabled formal method')
    explicit = authorization.get('enabled_model_methods')
    if explicit is None:
        return [(model, method) for model in MODELS for method in METHODS if method in methods]
    from research.v4.bridge.formal_planner_extensions import extension_model_ids
    allowed_models = MODELS + extension_model_ids(plan)
    pairs = []
    for row in explicit:
        pair = (row['planner_model'], row['method'])
        if pair[0] not in allowed_models or pair[1] not in METHODS or pair in pairs:
            raise ValueError('Invalid or duplicate planner/method scope')
        if pair[1] in methods:
            pairs.append(pair)
    return pairs


def in_scope(plan, row):
    model = row.get('planner_model', row.get('model', row.get('model_id')))
    # Legacy tests/entries without a model use the original method filter.
    if model is None and plan.get('authorization', {}).get('enabled_model_methods') is None:
        return method_of(row) in plan.get('authorization', {}).get('enabled_methods', METHODS)
    if (model, method_of(row)) not in selected_pairs(plan):return False
    backends=plan.get('authorization',{}).get('enabled_reviewer_backends')
    return backends is None or row.get('reviewer_backend','luna') in backends


def method_name(plan, row):
    backend=row.get('reviewer_backend')
    if backend in ('openai_decisions','jev'):
        return {'openai_decisions':'Ours（gpt-decision）','jev':'Ours（jev）'}[backend]
    if method_of(row) == 'Ours' and row.get('planner_model', row.get('model')) == 'gpt-6-astra':
        return plan.get('authorization', {}).get('ours_display_name', 'Ours')
    return method_of(row)
