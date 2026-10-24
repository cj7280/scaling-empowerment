"""Render the first world of a saved metric rollout, without another simulation."""
from contextlib import nullcontext
from pathlib import Path
import hashlib
import json

import numpy as np


def render_evaluation(source, output):
    from empowerment_warp._vendor.plotting.plot_style import configure_rendering_environment
    configure_rendering_environment()
    from empowerment_warp.media.locomotion import (
        _IncumbentAntRenderEnv, _native_locomotion_model, _load_incumbent_locomotion_renderer)
    from PIL import Image
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with np.load(source) as trace:
        metadata = json.loads(str(trace['metadata']))
        # Include reset and the terminal action, then stop. Always world zero.
        steps = int(trace['alive'][:100, 0].sum())
        states = {name: np.concatenate([trace['initial_' + name][0:1], trace[name][:steps]])
                  for name in ('qpos', 'qvel', 'mocap_pos', 'mocap_quat')}
    qpos, qvel = states['qpos'], states['qvel']
    module = _load_incumbent_locomotion_renderer()
    builder = metadata['environment'] == 'builderbench_corner3'
    if builder:
        from empowerment_warp.envs.builderbench.config import BuilderBenchConfig
        from empowerment_warp.envs.builderbench.model import build_model
        from empowerment_warp.media.builderbench import _IncumbentBuilderBenchRenderEnv
        model = build_model(BuilderBenchConfig(num_envs=1))
        ids = model.mocap_ids
        env = _IncumbentBuilderBenchRenderEnv(model, dict(qpos_seq=qpos, qvel_seq=qvel,
            mocap_pos_seq=states['mocap_pos'][:, ids], mocap_quat_seq=states['mocap_quat'][:, ids]))
        renderer = module._BuilderBenchGhostRenderer(env, trail_steps=72, trail_stride=3)
        arm = np.array([*module.GRAY_RGB, module.BUILDERBENCH_GRAY_ARM_ALPHA])
        blocks = np.array([*module.BUILDERBENCH_GRAY_BLOCK_RGB, module.BUILDERBENCH_BLOCK_ALPHA])
        def scene(scene, index):
            kwargs = dict(qvel=qvel[index], mocap_pos=states['mocap_pos'][index, ids],
                          mocap_quat=states['mocap_quat'][index, ids])
            renderer.add_arm(scene, qpos[index], arm, **kwargs)
            renderer.add_blocks(scene, qpos[index], blocks, **kwargs)
        stride, fps = 3, 24
    else:
        env = _IncumbentAntRenderEnv(_native_locomotion_model('ant_soccer'), qpos[0], qvel[0])
        renderer = module._AntGhostRenderer(env, trail_steps=72, trail_stride=3)
        body = np.array([*module.GRAY_RGB, module.ANT_BODY_ALPHA_GRAY])
        ball = np.array([*module.BALL_SOLID_RGB, module.BALL_SOLID_ALPHA])
        def scene(scene, index):
            renderer.add_ant(scene, qpos[index], body)
            renderer.add_ball(scene, qpos[index], ball)
        stride, fps = 1, 38
    background = nullcontext() if builder else renderer.empowerment_background()
    with background, renderer.hidden_base_geoms():
        frames = module._render_video_frames(env, renderer, len(qpos), scene, stride,
                                             min_render_size=640, render_workers=1)
    speed = fps * stride * metadata['control_dt']
    label = f"Potential policy | {metadata['checkpoint_step']/1e6:.1f}M | World 0 | {speed:g}x"
    frames = module.add_video_label_band(frames, lambda index: label)
    metadata.update(source_trace=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        frame_stride=stride, fps=fps, playback_speed=speed, rendered_actions=steps,
        raw_frame_indices=module._rendered_frame_indices(len(qpos), stride),
        new_simulation=False, video=str(output / 'potential.mp4'))
    module._save_env_overlay_bundle(bundle={'saved_paths': {}}, output_dir=str(output),
        stem='potential', saved_key='video', frames=frames, fps=fps,
        artifact_role='matched_potential_evaluation', metadata=metadata, require_motion=False)
    for name, index in [('start', 0), ('middle', len(frames)//2), ('end', len(frames)-1)]:
        Image.fromarray(frames[index]).save(output / f'{name}.png')
    path = output / 'provenance.json'
    path.write_text(json.dumps(metadata, indent=2) + '\n')
    if not Path(metadata['video']).is_file():
        raise FileNotFoundError(metadata['video'])
    return dict(video=metadata['video'], fps=fps, metadata=str(path))
