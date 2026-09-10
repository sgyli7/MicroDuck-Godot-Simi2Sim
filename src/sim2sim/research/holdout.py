"""One final, fixed-seed evaluation after the candidate bundle is frozen."""
import argparse, json, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from .tasks import TASKS, BASELINE
from .evaluate import run_suite
from .roller_evaluate import run_suite as roller_suite
from .play_sequence import run as play_sequence
from .keyboard_walk import run as keyboard_walk


def run(bundle, workers=6, seeds=range(1000, 1030)):
    bundle = Path(bundle)
    frozen = json.loads((bundle / 'bundle.json').read_text())
    if not frozen.get('verification_completed_unix'): raise ValueError('Bundle is not verified')
    out = bundle / 'holdout'; out.mkdir(exist_ok=False)
    bank = json.loads((bundle / 'bank.json').read_text())
    plan = dict(seeds=list(seeds), started_unix=time.time(), bundle_frozen_unix=frozen['frozen_unix'],
                selection_changes_allowed=False, normal_scene='microduck_ball_stand_fix',
                labels=['candidate', 'source_mujoco', 'previous_godot'],
                source_roller_profile='source_play', godot_roller_profile='xml')
    (out / 'plan.json').write_text(json.dumps(plan, indent=2))
    summaries = []
    for skill, task in TASKS.items():
        cases = [('candidate', bank[skill], 'godot'), ('source_mujoco', task.source, 'mujoco'),
                 ('previous_godot', BASELINE / task.previous, 'godot')]
        if skill in ('roller', 'roller_crouch'): cases.append(('source_mujoco_xml', task.source, 'mujoco'))
        for label, source, backend in cases:
            dest = out / skill / label
            if skill == 'roller':
                result = roller_suite(skill, source, backend, seeds, workers, dest,
                                      reference_profile='source_play' if label == 'source_mujoco' else 'xml')
            else:
                scene = 'microduck_ball_stand_fix' if task.robot != 'microduck_roller' else None
                result = run_suite(skill, source, backend, seeds, workers, dest,
                                   entry='both', scene_robot=scene,
                                   reference_profile='source_play' if skill == 'roller_crouch' and label == 'source_mujoco' else 'xml')
            short = {k: v for k, v in result.items() if k != 'episodes'}
            short.update(label=label, episodes=len(result['episodes']))
            summaries.append(short)
            (out / 'summary.json').write_text(json.dumps(summaries, indent=2))
            print(json.dumps(short), flush=True)
    sequences = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(play_sequence, bank, out / 'continuous' / f'{seed}.json',
                               seed, 'microduck_ball_stand_fix'): seed for seed in seeds}
        for future in as_completed(futures):
            result = future.result()
            sequences.append(dict(seed=futures[future],
                skills={x['label']: x['metrics'] for x in result['segments']
                        if x['label'] in ('roulade', 'ground_pick', 'kick_left', 'kick_right')},
                ordinary_falls=[x['label'] for x in result['segments'] if x['label'] != 'roulade'
                    and x['metrics'].get('fell', x['metrics'].get('unintended_fall', False))]))
    (out / 'continuous_summary.json').write_text(json.dumps(sequences, indent=2))
    keyboard = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(keyboard_walk, source, out / 'keyboard' / label / f'{condition}_{seed}.json',
                               seed, condition, backend, 'microduck_ball_stand_fix'): label
                   for label, source, backend in [('candidate', bank['walking'], 'godot'),
                       ('source_mujoco', TASKS['walking'].source, 'mujoco')]
                   for condition in ('forward', 'turn', 'mixed') for seed in seeds}
        for future in as_completed(futures):
            row = future.result(); row['label'] = futures[future]; keyboard.append(row)
    (out / 'keyboard_summary.json').write_text(json.dumps(keyboard, indent=2))
    (out / 'completed.json').write_text(json.dumps(dict(completed_unix=time.time(),
        elapsed=time.time()-plan['started_unix'], scenarios=len(sequences), keyboard=len(keyboard)), indent=2))


def main():
    p = argparse.ArgumentParser(); p.add_argument('bundle', type=Path)
    p.add_argument('--workers', type=int, default=6)
    a = p.parse_args(); run(a.bundle, a.workers)


if __name__ == '__main__': main()
