from pathlib import Path
import argparse
import pandas as pd
import statsmodels.formula.api as smf

HERE = Path(__file__).resolve().parent
DEFAULT_DATA = HERE / 'datasets' / 'rawData.csv'
OUTCOMES = ('cn', 'en', 'se', 'grat_all')


def prepare_task_data(raw: pd.DataFrame) -> pd.DataFrame:
    """Turn participant-level responses into one row per completed task.

    Source item columns end in algorithm number (e.g. eval_1_3).
    Task and sequence position are derived from the recorded algorithm order.
    """
    if not raw['id'].is_unique:
        raise ValueError('Each participant ID must occur exactly once in rawData.csv.')

    records = []
    for _, participant in raw.iterrows():
        for position, ordinal in enumerate(('first', 'second', 'third'), start=1):
            algorithm = int(participant[f'{ordinal}_algorithm'])
            task = int(participant[f'{ordinal}_task'])
            if algorithm not in (1, 2, 3) or task not in (1, 2, 3):
                raise ValueError('Expected algorithms and task IDs 1, 2, and 3.')

            values = {
                item: float(participant[f'{item}_{algorithm}'])
                for item in ('cn_1', 'cn_2', 'en_1', 'se_1',
                             'eval_1', 'eval_2', 'eval_3')
            }
            records.append({
                'id': int(participant['id']),
                'algorithm': algorithm,
                'task': task,
                'position': position,
                'cn': (values['cn_1'] + values['cn_2']) / 2,
                'en': values['en_1'],
                'se': values['se_1'],
                'eval': (values['eval_1'] + values['eval_2'] + values['eval_3']) / 3,
                'eval_no_effort': (values['eval_1'] + values['eval_3']) / 2,
            })

    data = pd.DataFrame(records)
    data['grat_all'] = data[['cn', 'en', 'se']].mean(axis=1)
    if data.isna().any().any():
        raise ValueError('Unexpected missing values in analysis variables.')
    if not (data.groupby('id').size() == 3).all():
        raise ValueError('Expected exactly three observations per participant.')
    for participant_id, part in data.groupby('id'):
        if set(part['algorithm']) != {1, 2, 3} or set(part['task']) != {1, 2, 3}:
            raise ValueError(f'Incomplete algorithm/task rotation for participant {participant_id}.')
    return data


def estimate(data: pd.DataFrame, outcome: str, predictor: str) -> dict:
    """Within-between OLS with participant-clustered SEs and t reference."""
    frame = data.copy()
    frame['between'] = frame.groupby('id')[predictor].transform('mean')
    frame['within'] = frame[predictor] - frame['between']
    formula = f'{outcome} ~ within + between + C(algorithm) + C(task) + C(position)'
    fitted = smf.ols(formula, data=frame).fit(
        cov_type='cluster', cov_kwds={'groups': frame['id']}, use_t=True
    )
    lower, upper = fitted.conf_int().loc['within']
    return {
        'analysis': 'primary' if predictor == 'eval' else 'sensitivity',
        'outcome': outcome,
        'coefficient': fitted.params['within'],
        'ci_lower': lower,
        'ci_upper': upper,
        'p_value': fitted.pvalues['within'],
        'n_participants': frame['id'].nunique(),
        'n_observations': len(frame),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=DEFAULT_DATA,
                        help='Path to rawData.csv')
    parser.add_argument('--output', type=Path, default=None,
                        help='Optional CSV output path (not written by default)')
    args = parser.parse_args()

    data = prepare_task_data(pd.read_csv(args.data))
    results = pd.DataFrame([
        estimate(data, outcome, predictor)
        for predictor, outcomes in (
            ('eval', OUTCOMES),
            ('eval_no_effort', ('cn', 'en', 'se')),
        )
        for outcome in outcomes
    ])
    print(f'Participants: {data.id.nunique()}; task-level observations: {len(data)}')
    print('Model: outcome ~ within + between + C(algorithm) + C(task) + C(position)')
    print('Inference: participant-clustered covariance; t-reference (use_t=True)')
    for row in results.itertuples(index=False):
        p = '< .001' if row.p_value < 0.001 else f'{row.p_value:.3f}'
        print(f'{row.analysis:11s} {row.outcome:8s} '
              f'b={row.coefficient:.3f}  95% CI '
              f'[{row.ci_lower:.3f}, {row.ci_upper:.3f}]  p {p}')
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(args.output, index=False)
        print(f'Saved detailed results to {args.output}')


if __name__ == '__main__':
    main()
