"""auto_qc 的单元测试。"""

import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import h5py
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import auto_qc as qc
qc.check_deps()


class QualityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def episode(self, **data):
        p = self.root / 'sample.h5'
        with h5py.File(p, 'w') as f:
            for name, value in data.items():
                f[name] = value
        return p

    def verdict(self, **data):
        return qc.check_episode(self.episode(**data), {})[0]

    def test_float_and_column_timestamps(self):
        for ts in [np.arange(20) / 30, (np.arange(20) / 30).reshape(-1, 1)]:
            self.assertTrue(self.verdict(timestamp=ts))

    def test_timestamp_gap_and_reverse(self):
        self.assertFalse(self.verdict(time_s=np.r_[np.arange(19) / 30, 5]))
        self.assertFalse(self.verdict(timestamp=np.arange(20)[::-1]))

    def test_high_rate_200hz_imu_ns_jitter_and_no_duplicate_reasons(self):
        # 复现 200Hz IMU 纳秒时间戳（正常中位间隔 ~5.04ms = 5,044,500ns，偶发抖动 28.07ms = 28,073,500ns < 1000ms）
        diffs_ns = np.full(400, 5_044_500, dtype=np.int64)
        diffs_ns[120] = 28_073_500  # 28.07ms 正常 USB/ROS 缓冲抖动，不应误报为 28073.5ms 丢帧
        imu_ts = np.r_[1_790_000_000_000_000_000, 1_790_000_000_000_000_000 + np.cumsum(diffs_ns)]

        p = self.root / 'imu_200hz.h5'
        with h5py.File(p, 'w') as f:
            f['observations/imu/angular_velocity'] = np.ones((401, 3), dtype=np.float32)
            f['observations/imu/timestamp_ns'] = imu_ts
            f['observations/imu/timestamp_ns_standardized'] = imu_ts
            f['observations/imu/arrival_timestamp_ns'] = imu_ts
            f['observations/imu/arrival_timestamp_ns_standardized'] = imu_ts

        is_pass, findings = qc.check_episode(str(p), {})
        self.assertTrue(is_pass, msg=f'False positive on 200Hz IMU jitter: {findings}')

        # 当真正发生 >1000ms (1.5s = 1,500,000,000ns) 严重丢帧时，原因只应输出 1 次而非重复 4 遍
        diffs_bad = diffs_ns.copy()
        diffs_bad[120] = 1_500_000_000
        imu_ts_bad = np.r_[1_790_000_000_000_000_000, 1_790_000_000_000_000_000 + np.cumsum(diffs_bad)]
        with h5py.File(p, 'w') as f:
            f['observations/imu/angular_velocity'] = np.ones((401, 3), dtype=np.float32)
            f['observations/imu/timestamp_ns'] = imu_ts_bad
            f['observations/imu/timestamp_ns_standardized'] = imu_ts_bad
            f['observations/imu/arrival_timestamp_ns'] = imu_ts_bad
            f['observations/imu/arrival_timestamp_ns_standardized'] = imu_ts_bad

        is_pass_bad, findings_bad = qc.check_episode(str(p), {})
        self.assertFalse(is_pass_bad)
        gap_msgs = [msg for _, msg in findings_bad if '严重丢帧' in msg]
        self.assertEqual(len(gap_msgs), 1)
        self.assertIn('1500.0ms', gap_msgs[0])
        self.assertIn('5.0ms', gap_msgs[0])

    def test_nan_and_zero_actions(self):
        self.assertFalse(self.verdict(state=np.ones((20, 6)), value=np.nan))
        self.assertFalse(self.verdict(action=np.zeros(20)))

    def test_vla_zero_commanded_action_not_bypassed_by_direction_or_executed(self):
        # 即使 action/normal_direction 恒为单位向量、action/executed 含非零初始关节角，
        # 只要核心控制指令 action/action 与 action/commanded 全为 0，仍应判定为无有效操作信号
        p = self.root / 'zero_cmd.h5'
        with h5py.File(p, 'w') as f:
            f['action/action'] = np.zeros((20, 32), dtype=np.float32)
            f['action/commanded'] = np.zeros((20, 32), dtype=np.float32)
            f['action/normal_direction'] = np.tile(np.array([0.0, 0.0, 1.0], dtype=np.float32), (20, 1))
            f['action/tangent_direction'] = np.tile(np.array([1.0, 0.0, 0.0], dtype=np.float32), (20, 1))
            f['action/executed'] = np.ones((20, 32), dtype=np.float32)
        is_pass, findings = qc.check_episode(str(p), {})
        self.assertFalse(is_pass)
        self.assertTrue(any('动作数据全为0' in msg for _, msg in findings), findings)

    def test_all_zero_valid_mask_with_all_nan_aligned_data_fails(self):
        p = self.root / 'all_invalid_mask.h5'
        with h5py.File(p, 'w') as f:
            f['state'] = np.ones((20, 6), dtype=np.float32)
            f['aligned/imu/angular_velocity'] = np.full((20, 3), np.nan, dtype=np.float32)
            f['aligned/imu/angular_velocity_valid'] = np.zeros(20, dtype=np.uint8)
        is_pass, findings = qc.check_episode(str(p), {})
        self.assertFalse(is_pass)
        self.assertTrue(any('aligned/imu/angular_velocity' in msg for _, msg in findings), findings)

    def test_action_dimension_mismatch_in_folder_is_flagged(self):
        rec = self.root / 'record_data_20261015_000010'
        rec.mkdir()
        u = np.linspace(0.0, 1.0, 40)
        base = np.stack([np.sin(np.pi * u), np.cos(np.pi * u), u, 1.0 - u], axis=1).astype(np.float32)
        for i in range(4):
            with h5py.File(rec / f'ep_{i}.h5', 'w') as f:
                f['action/commanded'] = base
        # 第 5 个文件第 0 列含 NaN 导致有效列维度变为 3 维（与基准 4 维不一致）
        broken = base.copy()
        broken[0, 0] = np.nan
        with h5py.File(rec / 'ep_dim_mismatch.h5', 'w') as f:
            f['action/commanded'] = broken

        anomalies = qc.check_folder_action_consistency(
            str(rec), [f'ep_{i}.h5' for i in range(4)] + ['ep_dim_mismatch.h5']
        )
        self.assertIn('ep_dim_mismatch.h5', anomalies)

    def test_empty_and_mismatched_stream(self):
        self.assertFalse(self.verdict(state=np.ones((20, 6)), image=np.empty((0, 16, 16, 3), dtype=np.uint8)))
        self.assertFalse(self.verdict(action=np.ones((20, 6)), timestamp=np.arange(19)))

    def test_late_black_and_freeze_boundary(self):
        frames = np.random.default_rng(1).integers(40, 200, (80, 16, 16, 3), dtype=np.uint8)
        frames[60:] = 0
        self.assertFalse(self.verdict(image=frames))
        self.assertFalse(self.verdict(image=np.full((15, 16, 16, 3), 100, dtype=np.uint8)))

    def test_long_sequence_chunked_black_and_freeze(self):
        rng = np.random.default_rng(4)
        frames_black = rng.integers(40, 200, (300, 16, 16, 3), dtype=np.uint8)
        frames_black[180:195] = 0
        self.assertFalse(self.verdict(image=frames_black))

        frames_freeze = rng.integers(40, 200, (300, 16, 16, 3), dtype=np.uint8)
        frames_freeze[250:265] = frames_freeze[250]
        self.assertFalse(self.verdict(image=frames_freeze))

    def test_separated_dark_frames_and_normalized_pixels(self):
        frames = np.random.default_rng(2).random((40, 16, 16, 3)).astype('float32')
        frames[::2] = 0
        self.assertTrue(self.verdict(image=frames))

    def test_real_armdog_vla_schema_and_mask_validation(self):
        p = self.root / 'episode_1_vla_valid.hdf5'
        t_main, t_imu, t_joints = 30, 300, 100
        with h5py.File(p, 'w') as f:
            f.attrs['episode_len'] = t_main
            f.attrs['imu_sample_count'] = t_imu
            f.attrs['raw_joints_sample_count'] = t_joints

            f['metadata/action_space'] = np.array(b'hybrid_arm_joint_position')
            f['metadata/episode_id'] = np.array(b'ep_1')
            f['metadata/done'] = np.zeros(t_main, dtype=np.uint8)
            f['metadata/reset_flag'] = np.zeros(t_main, dtype=np.uint8)
            sync_diag = np.ones((t_main, 28), dtype=np.float32)
            sync_diag[:, 2] = np.nan
            f['metadata/sync_diag'] = sync_diag
            f['metadata/timestamps'] = np.arange(t_main, dtype=np.float64) * 66.7 + 1.79e12
            f['metadata/timestamps_ns'] = np.arange(t_main, dtype=np.int64) * 66_700_000 + 1_790_000_000_000_000

            for cam in ('d435i', 'thirdview', 'wrist'):
                f[f'calibration/cameras/{cam}/K'] = np.full((3, 3), np.nan, dtype=np.float32)
                f[f'calibration/cameras/{cam}/T_camera_base'] = np.full((4, 4), np.nan, dtype=np.float32)
                f[f'calibration/cameras/{cam}/distortion'] = np.full((8,), np.nan, dtype=np.float32)

            f['vla/reward'] = np.full((t_main,), np.nan, dtype=np.float32)
            f['vla/success'] = np.zeros((1,), dtype=np.uint8)
            f['vla/frame_timestamp_ns'] = np.arange(t_main, dtype=np.int64) * 66_700_000 + 1_790_000_000_000_000

            imu_ts = np.arange(t_imu, dtype=np.int64) * 5_000_000 + 1_790_000_000_000_000_000
            for k in ('angular_velocity', 'linear_acceleration', 'orientation'):
                f[f'observations/imu/{k}'] = np.ones((t_imu, 3), dtype=np.float32)
            for k in ('timestamp_ns', 'timestamp_ns_standardized', 'arrival_timestamp_ns', 'source_timestamp_ns'):
                f[f'observations/imu/{k}'] = imu_ts

            joint_ts = np.arange(t_joints, dtype=np.int64) * 20_000_000 + 1_790_000_000_000_000_000
            for k in ('position', 'velocity', 'torque'):
                f[f'observations/joints/{k}'] = np.ones((t_joints, 16), dtype=np.float32)
            for k in ('timestamp_ns', 'timestamp_ns_standardized', 'arrival_timestamp_ns', 'source_timestamp_ns'):
                f[f'observations/joints/{k}'] = joint_ts

            f['observations/qpos'] = np.ones((t_main, 17), dtype=np.float32)
            f['observations/depth'] = np.ones((t_main, 1024), dtype=np.uint16) * 1000
            f['observations/foot_contact'] = np.zeros((t_main, 4), dtype=np.uint8)
            f['observations/foot_contact_valid'] = np.zeros((t_main,), dtype=np.uint8)

            f['aligned/frame_timestamp_ns'] = np.arange(t_main, dtype=np.int64) * 66_700_000 + 1_790_000_000_000_000_000
            for grp, keys in (('imu', ('angular_velocity', 'linear_acceleration', 'orientation')),
                              ('joints', ('position', 'velocity', 'torque'))):
                for k in keys:
                    arr = np.ones((t_main, 4), dtype=np.float32)
                    arr[0] = np.nan
                    valid = np.ones((t_main,), dtype=np.uint8)
                    valid[0] = 0
                    f[f'aligned/{grp}/{k}'] = arr
                    f[f'aligned/{grp}/{k}_valid'] = valid
                    f[f'aligned/{grp}/{k}_dt_ns'] = np.linspace(-5000, 5000, t_main).astype(np.int64)

                    win = np.ones((t_main, 20, 4), dtype=np.float32)
                    win[:, :3] = np.nan
                    wmask = np.ones((t_main, 20), dtype=np.uint8)
                    wmask[:, :3] = 0
                    wts = np.tile(np.arange(20, dtype=np.int64), (t_main, 1))
                    wts[:, :3] = -1
                    f[f'aligned/{grp}/{k}_window'] = win
                    f[f'aligned/{grp}/{k}_window_mask'] = wmask
                    f[f'aligned/{grp}/{k}_window_timestamp_ns'] = wts

            f['action/action'] = np.ones((t_main, 32), dtype=np.float32)
            f['action/commanded'] = np.ones((t_main, 32), dtype=np.float32)
            exec_act = np.full((t_main, 32), np.nan, dtype=np.float32)
            exec_act[:, :7] = 1.0
            exec_act[:, 14:17] = 0.5
            f['action/executed'] = exec_act
            for k in ('ee_force_cmd', 'force_direction', 'force_intensity', 'force_mode',
                      'normal_force_intent', 'normal_force_level', 'tangent_force_intent', 'tangent_force_level'):
                f[f'action/{k}'] = np.zeros((t_main, 3), dtype=np.float32)

        is_pass, findings = qc.check_episode(str(p), {})
        self.assertTrue(is_pass, msg=f'Unexpected findings on valid VLA episode: {findings}')

        with h5py.File(p, 'a') as f:
            bad_pos = np.ones((t_main, 4), dtype=np.float32)
            bad_pos[5] = np.nan
            f['aligned/joints/position'][:] = bad_pos
        is_pass_bad, findings_bad = qc.check_episode(str(p), {})
        self.assertFalse(is_pass_bad)
        self.assertTrue(any('aligned/joints/position' in msg for _, msg in findings_bad))

    def test_folder_action_consistency_detects_missing_actions_only(self):
        rec = self.root / 'record_data_20261014_000001'
        rec.mkdir()
        rng = np.random.default_rng(7)

        def make_base_traj(n_frames=200):
            u = np.linspace(0.0, 1.0, n_frames)
            traj = np.zeros((n_frames, 32), dtype=np.float32)
            traj[:, 0] = np.sin(np.pi * u) * 0.8
            traj[:, 1] = -0.8 - 1.5 * u
            traj[:, 2] = 0.5 + 1.0 * np.sin(0.5 * np.pi * u)
            traj[:, 3] = -0.7 + 0.9 * (u ** 1.2)
            traj[:, 4] = 1.5 * u
            traj[:, 5] = -0.8 + 0.6 * u
            traj[:, 6] = np.where(u < 0.4, 1.0, 0.0)
            return traj

        base = make_base_traj(200)

        for idx, length in enumerate([200, 165, 235, 180, 215, 190]):
            t_old = np.linspace(0.0, 1.0, 200)
            t_new = np.linspace(0.0, 1.0, length)
            warped = np.zeros((length, 32), dtype=np.float32)
            for c in range(7):
                warped[:, c] = np.interp(t_new, t_old, base[:, c]) + rng.normal(0, 0.01, size=length)
            with h5py.File(rec / f'ep_normal_{idx}.h5', 'w') as f:
                f['action/commanded'] = warped

        with h5py.File(rec / 'ep_early_stop.h5', 'w') as f:
            f['action/commanded'] = base[:110].copy()

        missing_step = base.copy()
        missing_step[:, 6] = 1.0
        with h5py.File(rec / 'ep_missing_step.h5', 'w') as f:
            f['action/commanded'] = missing_step

        wrong_order = np.vstack([base[100:], base[:100]])
        with h5py.File(rec / 'ep_wrong_order.h5', 'w') as f:
            f['action/commanded'] = wrong_order

        qual = self.root / 'passed'
        unqual = self.root / 'fail'
        json_dir = self.root / 'json'
        with contextlib.redirect_stdout(io.StringIO()):
            counts = qc.run_qc(str(self.root), str(qual), str(unqual), dry_run=False, rules={}, json_dir=str(json_dir))

        self.assertEqual(counts, (7, 2))
        report = json.loads((json_dir / f'{rec.name}.json').read_text(encoding='utf-8'))
        self.assertEqual(report['pass_count'], 7)
        self.assertEqual(report['pending_count'], 2)
        self.assertIn('动作缺失', report['pending_files']['ep_early_stop.h5'][0])
        self.assertIn('动作缺失', report['pending_files']['ep_missing_step.h5'][0])
        self.assertIn('ep_wrong_order.h5', report['passed_files'])

    def test_missing_waypoint_with_shorter_path_is_still_detected(self):
        # 核心行程没有缺失，但跳过中间途经点且相位对齐后持续偏离。
        base = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]], dtype=np.float32)
        skipped = np.array([[0, 0], [1, 1], [0, 0]], dtype=np.float32)
        rng = np.random.default_rng(4)
        names = []
        for i in range(7):
            points, length = (base, 160) if i < 6 else (skipped, 120)
            action = np.stack([
                np.interp(np.linspace(0, 1, length), np.linspace(0, 1, len(points)), points[:, c])
                for c in range(2)
            ], axis=1)
            if i < 6:
                action += rng.normal(0, 0.003, action.shape)
            name = f'ep_{i}.h5'
            names.append(name)
            with h5py.File(self.root / name, 'w') as f:
                f['action/commanded'] = action
        anomalies = qc.check_folder_action_consistency(str(self.root), names)
        self.assertIn('动作缺失', anomalies['ep_6.h5'][0])
        for name in names[:-1]:
            self.assertNotIn(name, anomalies)

    def test_truncated_hdf5_moves_from_record_to_pending_and_does_not_pollute_failures(self):
        source = self.root / 'record' / 'record_data_20261014_000002'
        source.mkdir(parents=True)
        with h5py.File(source / 'ep_good.h5', 'w') as f:
            f['state'] = np.ones((20, 6))
        with h5py.File(source / 'episode_140_vla_valid.hdf5', 'w') as f:
            f['state'] = np.ones((20_000, 6))
        bad = source / 'episode_140_vla_valid.hdf5'
        with open(bad, 'r+b') as stream:
            stream.truncate(1024)
        status, findings = qc.check_episode(str(bad), {})
        self.assertIsNone(status)
        self.assertTrue(any(level == 'PENDING' for level, _ in findings))
        self.assertTrue(any('当前大小 1024 字节' in msg and 'HDF5 声明末尾' in msg
                            for _, msg in findings), findings)

        good_dir, pending_dir, json_dir = (self.root / name for name in ('valid', 'pending', 'json'))
        with contextlib.redirect_stdout(io.StringIO()):
            counts = qc.run_qc(str(source.parent), str(good_dir), str(pending_dir),
                               rules={}, json_dir=str(json_dir))
        self.assertEqual(counts, (1, 1))
        self.assertFalse(bad.exists())
        self.assertFalse(source.exists())
        self.assertTrue((pending_dir / source.name / bad.name).is_file())
        report = json.loads((json_dir / (source.name + '.json')).read_text(encoding='utf-8'))
        self.assertEqual(report['total_files'], 2)
        self.assertEqual(report['pending_count'], 1)
        self.assertIn(bad.name, report['pending_files'])
        self.assertIn('ep_good.h5', report['passed_files'])

    def test_pending_copy_and_dry_run_preserve_record(self):
        for mode in ('copy', 'dry'):
            with self.subTest(mode=mode):
                source = self.root / mode / 'record' / 'record_data_20261014_000003'
                source.mkdir(parents=True)
                bad = source / 'episode_140_vla_valid.hdf5'
                with h5py.File(bad, 'w') as f:
                    f['state'] = np.ones((20_000, 6))
                with open(bad, 'r+b') as stream:
                    stream.truncate(1024)
                root = source.parent.parent
                with contextlib.redirect_stdout(io.StringIO()):
                    counts = qc.run_qc(str(source.parent), str(root / 'passed'), str(root / 'pending'),
                                       copy_mode=mode == 'copy', dry_run=mode == 'dry',
                                       json_dir=str(root / 'json'))
                self.assertEqual(counts, (0, 1))
                self.assertTrue(bad.is_file())
                self.assertEqual((root / 'pending' / source.name / bad.name).is_file(), mode == 'copy')

    def test_pending_folder_collision_does_not_overwrite_existing_review(self):
        source = self.root / 'record' / 'record_data_20261014_000004'
        source.mkdir(parents=True)
        bad = source / 'episode_140_vla_valid.hdf5'
        with h5py.File(bad, 'w') as f:
            f['state'] = np.ones((20_000, 6))
        with open(bad, 'r+b') as stream:
            stream.truncate(1024)
        prior = self.root / 'pending' / source.name
        prior.mkdir(parents=True)
        (prior / bad.name).write_bytes(b'keep prior review')
        with contextlib.redirect_stdout(io.StringIO()):
            qc.run_qc(str(source.parent), str(self.root / 'passed'), str(self.root / 'pending'),
                      json_dir=str(self.root / 'json'))
        self.assertEqual((prior / bad.name).read_bytes(), b'keep prior review')
        self.assertTrue((self.root / 'pending' / (source.name + '_1') / bad.name).is_file())

    def test_input_directory_equal_to_review_directory_is_rejected(self):
        with self.assertRaises(ValueError):
            qc.run_qc(str(self.root / 'record'), str(self.root / 'passed'), str(self.root / 'record'))

    def test_transient_hdf5_open_error_retries_before_classification(self):
        p = self.episode(state=np.ones((20, 6)))
        real_open = h5py.File
        attempts = []

        def open_after_transient_error(*args, **kwargs):
            attempts.append(args[0])
            if len(attempts) == 1:
                raise OSError('truncated file: eof = 1024, sblock->base_addr = 0, stored_eof = 2048')
            return real_open(*args, **kwargs)

        with patch.object(qc.h5py, 'File', side_effect=open_after_transient_error):
            result, findings = qc.check_episode(str(p), {})
        self.assertTrue(result, findings)
        self.assertEqual(len(attempts), 2)

    def test_truncated_single_file_report_is_pending(self):
        p = self.root / 'episode_140_vla_valid.hdf5'
        with h5py.File(p, 'w') as f:
            f['state'] = np.ones((20_000, 6))
        with open(p, 'r+b') as stream:
            stream.truncate(1024)
        result = subprocess.run(
            [sys.executable, str(Path(qc.__file__)), '--file', str(p), '--json-dir', str(self.root / 'json')],
            cwd=self.root, capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 1)
        report = json.loads((self.root / 'json' / (p.stem + '.json')).read_text(encoding='utf-8'))
        self.assertIn(p.name, report['pending_files'])

    def test_other_unreadable_hdf5_still_fails(self):
        p = self.root / 'garbage.h5'
        p.write_bytes(b'not an HDF5 file')
        result, findings = qc.check_episode(str(p), {})
        self.assertFalse(result)
        self.assertTrue(any('文件损坏' in msg for _, msg in findings))

    def test_jpeg_decode_and_corruption(self):
        rng = np.random.default_rng(3)
        payloads = []
        for _ in range(20):
            buf = io.BytesIO()
            Image.fromarray(rng.integers(30, 220, (32, 32, 3), dtype=np.uint8)).save(buf, format='JPEG')
            payloads.append(buf.getvalue())
        padded = np.zeros((20, max(map(len, payloads))), dtype=np.uint8)
        for i, payload in enumerate(payloads):
            padded[i, :len(payload)] = np.frombuffer(payload, dtype=np.uint8)
        self.assertTrue(self.verdict(image=padded, image_len=np.array(list(map(len, payloads)), dtype=np.float64)))
        padded[0] = 1
        self.assertFalse(self.verdict(image=padded, image_len=np.array(list(map(len, payloads)), dtype=np.float64)))

    def test_single_empty_jpeg_frame_with_and_without_lengths(self):
        rng = np.random.default_rng(23)
        payloads = []
        for _ in range(20):
            buf = io.BytesIO()
            Image.fromarray(rng.integers(30, 220, (32, 32, 3), dtype=np.uint8)).save(buf, format='JPEG')
            payloads.append(buf.getvalue())
        lengths = np.array([len(data) for data in payloads], dtype=np.float64)
        padded = np.zeros((20, max(lengths.astype(int))), dtype=np.uint8)
        for i, data in enumerate(payloads):
            padded[i, :len(data)] = np.frombuffer(data, dtype=np.uint8)
        padded[7] = 0
        lengths[7] = 0

        for kwargs in ({'rgb_len': lengths}, {}):
            with self.subTest(has_lengths=bool(kwargs)):
                p = self.episode(rgb=padded, **kwargs)
                passed, findings = qc.check_episode(str(p), {})
                self.assertFalse(passed)
                self.assertTrue(any('空帧' in msg and '7' in msg for _, msg in findings), findings)

    def test_short_frame_in_normal_sized_jpeg_stream(self):
        rng = np.random.default_rng(27)
        payloads = []
        for _ in range(20):
            buf = io.BytesIO()
            Image.fromarray(rng.integers(30, 220, (128, 128, 3), dtype=np.uint8)).save(buf, format='JPEG')
            payloads.append(buf.getvalue())
        lengths = np.array([len(payload) for payload in payloads], dtype=np.float64)
        padded = np.zeros((20, max(lengths.astype(int))), dtype=np.uint8)
        for i, payload in enumerate(payloads):
            padded[i, :len(payload)] = np.frombuffer(payload, dtype=np.uint8)
        lengths[12] = 300
        passed, findings = qc.check_episode(str(self.episode(rgb=padded, rgb_len=lengths)), {})
        self.assertFalse(passed)
        self.assertTrue(any('空帧' in msg and '12' in msg for _, msg in findings), findings)

    def test_invalid_jpeg_lengths_are_reported(self):
        rgb = np.full((20, 20), 1, dtype=np.uint8)
        lengths = np.full(20, 20, dtype=np.float64)
        lengths[3] = np.nan
        lengths[5] = 21
        passed, findings = qc.check_episode(str(self.episode(rgb=rgb, rgb_len=lengths)), {})
        self.assertFalse(passed)
        self.assertTrue(any('空帧' in msg and '3' in msg for _, msg in findings), findings)
        self.assertTrue(any('长度异常' in msg and '5' in msg for _, msg in findings), findings)

    def test_real_episode_minor_action_variation_is_not_missing_step(self):
        # 真实任务样本：episode_5 的小幅摆动维度行程只有 8%；
        # episode_7 的最大途经点误差 0.25、路径行程 83%，但时序轨迹一致。
        repo = Path(__file__).resolve().parents[1]
        paths = sorted((repo / 'passed/record_data_20260924_143938').glob('*.hdf5'))
        paths += sorted((repo / 'pending/record_data_20260924_143938').glob('*.hdf5'))
        if len(paths) != 11:
            self.skipTest('示例 episode 未提供')
        folder = str(self.root)
        for p in paths:
            with h5py.File(p) as src, h5py.File(self.root / p.name, 'w') as dst:
                dst['action/commanded'] = src['action/commanded'][:]
        anomalies = qc.check_folder_action_consistency(folder, [p.name for p in paths])
        self.assertNotIn('episode_5_vla_valid.hdf5', anomalies, anomalies)
        self.assertNotIn('episode_7_vla_valid.hdf5', anomalies, anomalies)

    def test_default_cli_reads_record_not_pending(self):
        source = self.root / 'record' / 'record_data_20261014_000005'
        source.mkdir(parents=True)
        with h5py.File(source / 'ep_good.h5', 'w') as f:
            f['state'] = np.ones((20, 6))
        prior = self.root / 'pending' / 'record_data_20261014_000006'
        prior.mkdir(parents=True)
        (prior / 'untouched.h5').write_bytes(b'pending review')
        result = subprocess.run([sys.executable, str(Path(qc.__file__))], cwd=self.root,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((self.root / 'passed' / source.name / 'ep_good.h5').is_file())
        self.assertEqual((prior / 'untouched.h5').read_bytes(), b'pending review')

    def test_single_file_exit(self):
        p = self.episode(state=np.ones((2, 6)))
        result = subprocess.run(
            [sys.executable, str(Path(qc.__file__)), '-f', str(p)],
            cwd=self.root,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 1)
        self.assertFalse((self.root / 'pending').exists())
        self.assertTrue((self.root / 'json' / 'sample.json').is_file())

    def test_modes_and_collision(self):
        for mode in ('dry', 'copy', 'move'):
            source = self.root / mode
            source.mkdir()
            target, rejected = self.root / (mode + '_ok'), self.root / (mode + '_bad')
            rec = source / 'record_data_20261013_000123'
            rec.mkdir()
            with h5py.File(rec / 'sample_valid.h5', 'w') as f:
                f['state'] = np.ones((20, 6))
            target.mkdir()
            (target / rec.name).mkdir()
            with contextlib.redirect_stdout(io.StringIO()):
                counts = qc.run_qc(str(source), str(target), str(rejected), copy_mode=mode == 'copy', dry_run=mode == 'dry')
            self.assertEqual(counts, (1, 0))
            self.assertTrue((target / rec.name).is_dir())
            self.assertEqual((target / (rec.name + '_1')).is_dir(), mode != 'dry')
            self.assertEqual(source.exists() and (source / rec.name).exists(), mode != 'move')

    def test_folder_fail_suffix_writes_reason(self):
        rec = self.root / 'record_data_20261013_000123'
        rec.mkdir()
        with h5py.File(rec / 'cam_fail.h5', 'w') as f:
            f['x'] = np.ones((20, 6))
        unqual = self.root / 'pending'
        with contextlib.redirect_stdout(io.StringIO()):
            qc.run_qc(str(self.root), str(self.root / 'qual'), str(unqual), dry_run=False, rules={})
        out = unqual / rec.name
        self.assertTrue(out.is_dir())
        with h5py.File(out / 'cam_fail.h5', 'r') as f:
            self.assertEqual(f.attrs.get('qc_result'), 'FAIL')
            self.assertIn('失败标记', f.attrs.get('qc_reason', ''))

    def test_folder_pass_stays_clean(self):
        rec = self.root / 'record_data_20261013_000124'
        rec.mkdir()
        with h5py.File(rec / 'cam_valid.h5', 'w') as f:
            f['state'] = np.ones((20, 6))
        qual = self.root / 'qual'
        with contextlib.redirect_stdout(io.StringIO()):
            qc.run_qc(str(self.root), str(qual), str(self.root / 'unqual'), dry_run=False, rules={})
        out = qual / rec.name
        self.assertTrue(out.is_dir())
        with h5py.File(out / 'cam_valid.h5', 'r') as f:
            self.assertNotIn('qc_result', f.attrs)

    def test_mixed_folder_splits_files_and_writes_json(self):
        rec = self.root / 'record_data_20261013_000125'
        rec.mkdir()
        with h5py.File(rec / 'ep_0_valid.h5', 'w') as f:
            f['state'] = np.ones((20, 6))
        with h5py.File(rec / 'ep_1_fail.h5', 'w') as f:
            f['state'] = np.ones((20, 6))

        qual = self.root / 'passed'
        unqual = self.root / 'pending'
        json_dir = self.root / 'json'
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            counts = qc.run_qc(str(self.root), str(qual), str(unqual), dry_run=False, rules={}, json_dir=str(json_dir))

        self.assertEqual(counts, (1, 1))
        self.assertTrue((qual / rec.name / 'ep_0_valid.h5').is_file())
        self.assertFalse((qual / rec.name / 'ep_1_fail.h5').exists())
        self.assertTrue((unqual / rec.name / 'ep_1_fail.h5').is_file())
        self.assertFalse((unqual / rec.name / 'ep_0_valid.h5').exists())

        self.assertNotIn('失败标记', buf.getvalue())
        report_file = json_dir / f'{rec.name}.json'
        self.assertTrue(report_file.is_file())
        report = json.loads(report_file.read_text(encoding='utf-8'))
        self.assertEqual(report['folder'], rec.name)
        self.assertEqual(report['pass_count'], 1)
        self.assertEqual(report['pending_count'], 1)
        self.assertEqual(report['passed_files'], ['ep_0_valid.h5'])
        self.assertIn('ep_1_fail.h5', report['pending_files'])

    def test_empty_shell_folder_is_deleted_and_not_rescanned(self):
        record_dir = self.root / 'record'
        passed_dir = self.root / 'passed'
        pending_dir = self.root / 'pending'
        json_dir = self.root / 'json'

        # 第一个文件夹含 HDF5 以及非 HDF5 附属文件，流转后整个空壳文件夹应被彻底删除
        rec1 = record_dir / 'record_data_20261016_000001'
        rec1.mkdir(parents=True)
        with h5py.File(rec1 / 'ep_0_valid.h5', 'w') as f:
            f['state'] = np.ones((20, 6))
        (rec1 / 'notes.txt').write_text('sidecar', encoding='utf-8')

        with contextlib.redirect_stdout(io.StringIO()):
            counts1 = qc.run_qc(str(record_dir), str(passed_dir), str(pending_dir), json_dir=str(json_dir))
        self.assertEqual(counts1, (1, 0))
        self.assertFalse(rec1.exists())

        # 若 record 中残留了空壳文件夹，下一个新文件夹进入时应自动清理旧空壳且不重复识别或覆盖旧 JSON
        rec1.mkdir(parents=True)
        rec2 = record_dir / 'record_data_20261016_000002'
        rec2.mkdir(parents=True)
        with h5py.File(rec2 / 'ep_0_valid.h5', 'w') as f:
            f['state'] = np.ones((20, 6))

        with contextlib.redirect_stdout(io.StringIO()):
            counts2 = qc.run_qc(str(record_dir), str(passed_dir), str(pending_dir), json_dir=str(json_dir))
        self.assertEqual(counts2, (1, 0))
        self.assertFalse(rec1.exists())
        self.assertFalse(rec2.exists())
        report1 = json.loads((json_dir / 'record_data_20261016_000001.json').read_text(encoding='utf-8'))
        self.assertEqual(report1['pass_count'], 1)

    def test_parallel_workers_matches_serial_results(self):
        record_dir = self.root / 'record'
        rec = record_dir / 'record_data_20261016_000099'
        rec.mkdir(parents=True)
        for i in range(4):
            with h5py.File(rec / f'ep_{i}_valid.h5', 'w') as f:
                f['state'] = np.ones((20, 6))
        with h5py.File(rec / 'ep_bad_fail.h5', 'w') as f:
            f['state'] = np.ones((20, 6))

        with contextlib.redirect_stdout(io.StringIO()):
            counts = qc.run_qc(
                str(record_dir),
                str(self.root / 'passed'),
                str(self.root / 'pending'),
                json_dir=str(self.root / 'json'),
                workers=2,
            )
        self.assertEqual(counts, (4, 1))
        report = json.loads((self.root / 'json' / f'{rec.name}.json').read_text(encoding='utf-8'))
        self.assertEqual(report['pass_count'], 4)
        self.assertEqual(report['pending_count'], 1)

    def test_folder_scene_redundancy_flags_over_limit_cluster_and_writes_json(self):
        record_dir = self.root / 'record'
        rec = record_dir / 'record_data_20261023_000001'
        rec.mkdir(parents=True)
        rng = np.random.default_rng(101)

        # 构造静态机位 A 的背景 (64x64)，前 4 个文件机位不动、仅局部小区域(目标物)移动
        bg_a = rng.integers(50, 180, (64, 64, 3), dtype=np.uint8)
        # 构造移动机器人后的新机位 B 的背景
        bg_b = rng.integers(50, 180, (64, 64, 3), dtype=np.uint8)

        for i in range(5):
            base_bg = bg_a.copy() if i < 4 else bg_b.copy()
            frames = []
            for t in range(15):
                frame = base_bg.copy()
                # 模拟桌面局部小物体在不同位置移动 + 帧间动态变化（避免冻结）
                r0 = (i * 8 + t) % 48
                c0 = (i * 10 + t * 2) % 48
                frame[r0:r0 + 8, c0:c0 + 8] = (220 - t * 5)
                frames.append(frame)
            with h5py.File(rec / f'ep_{i}.h5', 'w') as f:
                f['action/commanded'] = np.linspace(0.0, 1.0, 15 * 6, dtype=np.float32).reshape(15, 6)
                f['observations/third_rgb'] = np.stack(frames, axis=0)

        passed_dir = self.root / 'passed'
        pending_dir = self.root / 'pending'
        json_dir = self.root / 'json'
        with contextlib.redirect_stdout(io.StringIO()):
            counts = qc.run_qc(
                str(record_dir),
                str(passed_dir),
                str(pending_dir),
                json_dir=str(json_dir),
                rules={'max_same_scene_ratio': 0.50},
            )

        # 机位 A 占 4/5 = 80% > 50%，该机位簇下 4 个文件全部进入 pending，机位 B 的 ep_4.h5 进入 passed
        self.assertEqual(counts, (1, 4))
        self.assertTrue((passed_dir / rec.name / 'ep_4.h5').is_file())
        for i in range(4):
            self.assertTrue((pending_dir / rec.name / f'ep_{i}.h5').is_file())

        report = json.loads((json_dir / f'{rec.name}.json').read_text(encoding='utf-8'))
        self.assertEqual(report['pass_count'], 1)
        self.assertEqual(report['pending_count'], 4)
        self.assertIn('scene_diversity', report)
        self.assertEqual(report['scene_diversity']['evaluated_files'], 5)
        self.assertEqual(report['scene_diversity']['scene_cluster_count'], 2)
        self.assertEqual(report['scene_diversity']['max_scene_cluster_size'], 4)
        self.assertAlmostEqual(report['scene_diversity']['max_scene_ratio'], 0.8)
        for i in range(4):
            self.assertIn('同一场景/机位采集占比过高', report['pending_files'][f'ep_{i}.h5'][0])

    def test_folder_scene_diversity_passes_when_robot_moves(self):
        record_dir = self.root / 'record'
        rec = record_dir / 'record_data_20261023_000002'
        rec.mkdir(parents=True)
        rng = np.random.default_rng(202)

        # 6 个文件分布在 3 个不同机位（每个机位 2 个文件，占比 33.3% <= 50%）
        backgrounds = [rng.integers(40, 200, (64, 64, 3), dtype=np.uint8) for _ in range(3)]
        for i in range(6):
            base_bg = backgrounds[i // 2]
            frames = []
            for t in range(15):
                frame = base_bg.copy()
                frame[t:t + 8, t:t + 8] = 240
                frames.append(frame)
            with h5py.File(rec / f'ep_{i}.h5', 'w') as f:
                f['action/commanded'] = np.linspace(0.0, 1.0, 15 * 6, dtype=np.float32).reshape(15, 6)
                f['observations/third_rgb'] = np.stack(frames, axis=0)

        passed_dir = self.root / 'passed'
        pending_dir = self.root / 'pending'
        json_dir = self.root / 'json'
        with contextlib.redirect_stdout(io.StringIO()):
            counts = qc.run_qc(
                str(record_dir),
                str(passed_dir),
                str(pending_dir),
                json_dir=str(json_dir),
                rules={'max_same_scene_ratio': 0.50},
            )

        self.assertEqual(counts, (6, 0))
        report = json.loads((json_dir / f'{rec.name}.json').read_text(encoding='utf-8'))
        self.assertEqual(report['scene_diversity']['scene_cluster_count'], 3)
        self.assertAlmostEqual(report['scene_diversity']['max_scene_ratio'], 2 / 6, places=3)

    def test_depth_occlusion_detects_close_cable_and_blind_zone_but_ignores_brief_grasp(self):
        # 1. 持续贴脸数据线/异物 (0 < depth < 120mm 占 12.5% >= 8%，持续全部 20 帧) -> FAIL
        depth_cable = np.full((20, 32, 32), 800, dtype=np.uint16)
        depth_cable[:, :4, :] = 60  # 4/32 = 12.5% 区域在 60mm 处贴脸遮挡
        p_cable = self.episode(action=np.linspace(0, 1, 20), depth=depth_cable)
        passed, findings = qc.check_episode(str(p_cable), {})
        self.assertFalse(passed)
        self.assertTrue(any("贴脸像素持续占比" in msg for _, msg in findings), findings)

        # 2. 持续大面积红外盲区黑洞 (depth == 0 占 90% >= 85%) -> FAIL
        depth_blind = np.zeros((20, 32, 32), dtype=np.uint16)
        depth_blind[:, :3, :] = 800
        p_blind = self.episode(action=np.linspace(0, 1, 20), depth=depth_blind)
        passed_b, findings_b = qc.check_episode(str(p_blind), {})
        self.assertFalse(passed_b)
        self.assertTrue(any("深度盲区/阻挡像素持续占比" in msg for _, msg in findings_b), findings_b)

        # 3. 机械臂正常下探抓取（仅中间 3/20 = 15% 的帧靠近桌面 <120mm）-> PASS 不误报
        depth_grasp = np.full((20, 32, 32), 800, dtype=np.uint16)
        depth_grasp[9:12, :8, :] = 80
        p_grasp = self.episode(action=np.linspace(0, 1, 20), depth=depth_grasp)
        passed_g, findings_g = qc.check_episode(str(p_grasp), {})
        self.assertTrue(passed_g, msg=f"False positive on brief grasp depth: {findings_g}")

    def test_rgb_static_cable_occlusion_on_moving_camera_is_detected(self):
        # 模拟移动相机：背景随时间平移变化 (std >> 5.0, 中位亮度 ~120)，但镜头顶部 3/16 (18.75%) 被黑色数据线焊死遮挡
        yy, xx = np.meshgrid(np.arange(64), np.arange(64), indexing="ij")
        frames = np.stack([
            np.clip(120 + 50 * np.sin((xx + t * 5) / 6.0) + 30 * np.cos((yy + t * 3) / 8.0), 40, 220).astype(np.uint8)
            for t in range(20)
        ], axis=0)
        frames = np.repeat(frames[:, :, :, None], 3, axis=3)
        frames[:, :12, :, :] = 5  # 顶部 12 行 (18.75%) 恒定极暗死区
        p = self.episode(action=np.linspace(0, 1, 20), rgb=frames)
        passed, findings = qc.check_episode(str(p), {})
        self.assertFalse(passed)
        self.assertTrue(any("常驻暗区" in msg for _, msg in findings), findings)

    def test_folder_camera_occlusion_detects_mid_batch_cable_droop(self):
        record_dir = self.root / "record"
        rec = record_dir / "record_data_20261023_000003"
        rec.mkdir(parents=True)
        rng = np.random.default_rng(404)

        # 5 个不同机位背景（避免触发同场景占比），其中 ep_4 的第三视角镜头中途垂下一根黑色数据线且深度有效率骤降
        for i in range(5):
            bg = rng.integers(90, 160, (64, 64, 3), dtype=np.uint8)
            frames = []
            for t in range(15):
                fr = bg.copy()
                fr[(t * 2) % 48:(t * 2) % 48 + 10, (i * 8 + t) % 48:(i * 8 + t) % 48 + 10] = 220
                if i == 4:
                    # 在 ep_4 中模拟数据线垂下遮挡左上方 25% 视野
                    fr[:32, :32, :] = 6
                frames.append(fr)

            depth = np.full((15, 32, 32), 850, dtype=np.uint16)
            if i == 4:
                # 数据线导致 ep_4 深度有效率从 100% 跌至 50% (低于 85% 单文件上限，但显著低于同批 100% 基准)
                depth[:, :16, :] = 0

            with h5py.File(rec / f"ep_{i}.h5", "w") as f:
                f["action/commanded"] = np.linspace(0.0, 1.0, 15 * 6, dtype=np.float32).reshape(15, 6)
                f["observations/third_rgb"] = np.stack(frames, axis=0)
                f["observations/third_depth"] = depth

        passed_dir = self.root / "passed"
        pending_dir = self.root / "pending"
        json_dir = self.root / "json"
        with contextlib.redirect_stdout(io.StringIO()):
            counts = qc.run_qc(str(record_dir), str(passed_dir), str(pending_dir), json_dir=str(json_dir))

        self.assertEqual(counts, (4, 1))
        report = json.loads((json_dir / f"{rec.name}.json").read_text(encoding="utf-8"))
        self.assertIn("ep_4.h5", report["pending_files"])
        self.assertTrue(any("遮挡" in r for r in report["pending_files"]["ep_4.h5"]), report["pending_files"]["ep_4.h5"])

    def test_custom_input_and_output_folder_creates_passed_pending_and_json_in_pending(self):
        in_folder = self.root / "my_custom_task"
        out_folder = self.root / "custom_output"
        in_folder.mkdir(parents=True)
        with h5py.File(in_folder / "ep_0_valid.h5", "w") as f:
            f["state"] = np.ones((20, 6))
        with h5py.File(in_folder / "ep_1_fail.h5", "w") as f:
            f["state"] = np.ones((20, 6))

        result = subprocess.run(
            [sys.executable, str(Path(qc.__file__)), "-i", str(in_folder), "-o", str(out_folder)],
            cwd=self.root,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

        passed_sub = out_folder / "my_custom_task_passed"
        pending_sub = out_folder / "my_custom_task_pending"
        self.assertTrue(passed_sub.is_dir())
        self.assertTrue(pending_sub.is_dir())
        self.assertTrue((passed_sub / "ep_0_valid.h5").is_file())
        self.assertTrue((pending_sub / "ep_1_fail.h5").is_file())

        # JSON 报告应位于 <文件夹名称>_pending 文件夹内部
        json_file = pending_sub / "my_custom_task.json"
        self.assertTrue(json_file.is_file())
        report = json.loads(json_file.read_text(encoding="utf-8"))
        self.assertEqual(report["folder"], "my_custom_task")
        self.assertEqual(report["pass_count"], 1)
        self.assertEqual(report["pending_count"], 1)

    def test_custom_output_creates_both_passed_and_pending_even_when_all_pass(self):
        in_folder = self.root / "batch_all_ok"
        out_folder = self.root / "out_all_ok"
        in_folder.mkdir(parents=True)
        with h5py.File(in_folder / "ep_0.h5", "w") as f:
            f["state"] = np.ones((20, 6))

        with contextlib.redirect_stdout(io.StringIO()):
            counts = qc.run_qc(str(in_folder), output_dir=str(out_folder))
        self.assertEqual(counts, (1, 0))
        self.assertTrue((out_folder / "batch_all_ok_passed").is_dir())
        self.assertTrue((out_folder / "batch_all_ok_pending").is_dir())
        self.assertTrue((out_folder / "batch_all_ok_passed" / "ep_0.h5").is_file())
        self.assertTrue((out_folder / "batch_all_ok_pending" / "batch_all_ok.json").is_file())

    def test_progress_bar_renders_and_can_be_disabled(self):
        in_folder = self.root / "prog_task"
        out_folder = self.root / "prog_out"
        in_folder.mkdir(parents=True)
        for i in range(3):
            with h5py.File(in_folder / f"ep_{i}.h5", "w") as f:
                f["state"] = np.ones((20, 6))

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            qc.run_qc(str(in_folder), copy_mode=True, output_dir=str(out_folder))
        out_text = buf.getvalue()
        self.assertIn("3/3 (100%)", out_text)
        self.assertIn("文件/s", out_text)

        buf_no_prog = io.StringIO()
        with contextlib.redirect_stdout(buf_no_prog):
            qc.run_qc(
                str(in_folder),
                copy_mode=True,
                rules={"show_progress": False},
                output_dir=str(self.root / "prog_out_2"),
            )
        self.assertNotIn("文件/s", buf_no_prog.getvalue())

    def test_rgbd_thin_cable_joint_occlusion_is_detected(self):
        # 模拟仅占画面 6.25% (2/32 行，低于单流 8%~10% 阈值) 的细数据线：
        # 在 RGB 中形成常驻暗带，在同位置 Depth 中形成 RealSense 散斑盲区 (=0)
        rng = np.random.default_rng(505)
        rgb = rng.integers(80, 160, (15, 64, 64, 3), dtype=np.uint8)
        rgb[:, 10:14, :, :] = 10  # 4/64 = 6.25% 细暗线
        depth = np.full((15, 64, 64), 800, dtype=np.uint16)
        depth[:, 10:14, :] = 0    # 同位置双目红外盲区 (=0)

        p = self.episode(**{"observations/third_rgb": rgb, "observations/third_depth": depth, "action/commanded": np.linspace(0, 1, 15)})
        passed, findings = qc.check_episode(str(p), {})
        self.assertFalse(passed)
        self.assertTrue(any("RGB-D" in msg and "遮挡" in msg for _, msg in findings), findings)

    def test_multi_position_single_joint_range_variation_is_not_flagged(self):
        # 复现 record_data_20260101_123456 中多机位采集场景：
        # 单个连续关节因机位角度变化行程降至 12%~30%，但整体 6-DOF 路径占比 >= 90% 且夹爪正常开合，不应误报动作缺失
        folder = self.root / "multi_pos_batch"
        folder.mkdir(parents=True)
        u = np.linspace(0.0, 1.0, 160)

        def make_traj(j0_scale=1.0, offset=0.0):
            tr = np.zeros((160, 7), dtype=np.float32)
            tr[:, 0] = offset + j0_scale * 0.8 * np.sin(np.pi * u)
            tr[:, 1] = offset - 0.8 - 1.5 * u
            tr[:, 2] = 0.5 + 1.1 * np.sin(0.5 * np.pi * u)
            tr[:, 3] = -0.7 + 1.0 * (u ** 1.1)
            tr[:, 4] = 1.4 * u
            tr[:, 5] = -0.8 + 0.9 * u
            tr[:, 6] = np.where(u < 0.45, 1.0, 0.0)
            return tr

        names = []
        for i in range(6):
            name = f"episode_{i}_vla_valid.hdf5"
            names.append(name)
            # episode_5 模拟正对目标机位：腰部关节 0 行程仅为基准的 15%，且整体存在机位偏置，但总路径与夹爪完整
            tr = make_traj(j0_scale=0.15, offset=0.35) if i == 5 else make_traj(j0_scale=1.0, offset=0.02 * i)
            with h5py.File(folder / name, "w") as f:
                f["action/commanded"] = tr

        anomalies = qc.check_folder_action_consistency(str(folder), names)
        self.assertNotIn("episode_5_vla_valid.hdf5", anomalies, anomalies)

    def test_adaptive_rgbd_cable_with_realistic_luminance_and_motion(self):
        # 复现真实 RealSense 线缆遮挡：线缆受环境反光灰度均值 ~34 (高于旧阈值 25)，
        # 机械臂运动下背景标准差 ~28、线缆区标准差 ~5.0 (高于旧阈值 3.0)，且深度图对应位置为盲区 (=0)
        rng = np.random.default_rng(707)
        t_steps = np.linspace(0, 2 * np.pi, 14)[:, None, None, None]
        y_grid = np.linspace(0, np.pi, 64)[None, :, None, None]
        bg = 115.0 + 40.0 * np.sin(t_steps + y_grid) + rng.normal(0, 3.0, (14, 64, 64, 3))
        rgb = np.clip(bg, 45, 220).astype(np.uint8)
        # 顶部 4 行 (6.25% 面积) 存在灰度 ~34、带轻微反光波动 (std~5.0) 的数据线
        cable_wave = (34.0 + 5.0 * np.sin(np.linspace(0, 3.0, 14))[:, None, None, None])
        rgb[:, 6:10, :, :] = np.clip(cable_wave, 26, 42).astype(np.uint8)
        depth = np.full((14, 64, 64), 750, dtype=np.uint16)
        depth[:, 6:10, :] = 0

        p = self.episode(**{
            "observations/d435i_rgb": rgb,
            "observations/d435i_depth": depth,
            "action/commanded": np.linspace(0, 1, 14),
        })
        passed, findings = qc.check_episode(str(p), {})
        self.assertFalse(passed)
        self.assertTrue(any("RGB-D" in msg and "遮挡" in msg for _, msg in findings), findings)

    def test_explicit_directory(self):
        target = str(self.root / 'new_output')
        self.assertEqual(qc.resolve_dir(target, ['passed']), target)


if __name__ == '__main__':
    unittest.main()
