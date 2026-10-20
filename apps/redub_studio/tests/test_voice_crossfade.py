"""Synthetic audio timeline tests; no model, GPU, network or SoundFile required."""
import unittest

try:
    import numpy as np
except ImportError:
    np = None

from redub_studio.workers.voice_convert import (
    CONTEXT_SAMPLES, CROSSFADE_HALF_SAMPLES, SAMPLE_RATE, Window,
    align_prediction, blend_window, plan_crossfades, plan_windows, retain_core,
)


def prediction_for(window, samples):
    return np.pad(samples[window.read_start:window.read_end],
                  (window.pad_left, window.pad_right))


def stitch(windows, predictions):
    widths = plan_crossfades(windows)
    pending = None
    blocks, records = [], []
    for index, (window, prediction) in enumerate(zip(windows, predictions)):
        block, pending, record = blend_window(
            prediction, window, widths[index-1] if index else 0,
            widths[index] if index < len(widths) else 0, pending)
        blocks.append(block)
        if record is not None:
            records.append(record)
    assert pending is None
    return np.concatenate(blocks), records


def tiny_windows(lengths):
    total, start, windows = sum(lengths), 0, []
    for index, length in enumerate(lengths):
        end = start+length
        left, right = start-CONTEXT_SAMPLES, end+CONTEXT_SAMPLES
        windows.append(Window(index, start, end, max(0,left), min(total,right),
                              max(0,-left), max(0,right-total)))
        start = end
    return windows


@unittest.skipIf(np is None, "NumPy is required in the voice environment")
class VoiceCrossfadeTests(unittest.TestCase):
    def test_single_window_is_identical_to_previous_core_crop(self):
        for length in (1, 7, 239, 479, 480, SAMPLE_RATE//2, 3*SAMPLE_RATE+17):
            with self.subTest(length=length):
                samples = np.random.default_rng(length).normal(size=length).astype(np.float32)
                windows = plan_windows(length)
                self.assertEqual(len(windows), 1)
                prediction = prediction_for(windows[0], samples)
                previous, _ = retain_core(prediction, len(prediction), length, 4800)
                actual, records = stitch(windows, [prediction])
                np.testing.assert_array_equal(actual, previous)
                self.assertEqual(records, [])

    def test_matching_predictions_preserve_all_global_samples_exactly(self):
        for length in (30*SAMPLE_RATE+17, 60*SAMPLE_RATE+1):
            with self.subTest(length=length):
                samples = np.random.default_rng(length).normal(size=length).astype(np.float32)
                windows = plan_windows(length)
                actual, records = stitch(windows, [prediction_for(w,samples) for w in windows])
                self.assertEqual(len(actual), length)
                np.testing.assert_array_equal(actual, samples)
                self.assertEqual(len(records), len(windows)-1)
                for window, record in zip(windows[1:], records):
                    self.assertEqual(record['boundary_sample'], window.start)
                    self.assertEqual(record['start_sample'], window.start-CROSSFADE_HALF_SAMPLES)
                    self.assertEqual(record['end_sample'], window.start+CROSSFADE_HALF_SAMPLES)
                    self.assertEqual(record['samples'], 480)

    def test_opposite_predictions_replace_click_with_bounded_smooth_transition(self):
        windows = tiny_windows([6000, 6000])
        predictions = [np.full(w.input_samples, value, dtype=np.float32)
                       for w,value in zip(windows,(1.0,-1.0))]
        actual, records = stitch(windows, predictions)
        self.assertEqual(len(actual), 12000)
        self.assertEqual(len(records), 1)
        np.testing.assert_array_equal(actual[:5760], np.ones(5760,dtype=np.float32))
        np.testing.assert_array_equal(actual[6240:], -np.ones(5760,dtype=np.float32))
        self.assertEqual(float(actual[5760]), 1.0)
        self.assertEqual(float(actual[6239]), -1.0)
        self.assertLess(float(np.max(np.abs(np.diff(actual)))), 0.01)
        self.assertGreaterEqual(float(actual.min()), -1.0)
        self.assertLessEqual(float(actual.max()), 1.0)

    def test_different_predictions_mix_same_timestamps_without_shifting_events(self):
        windows = tiny_windows([3000,3000])
        samples = np.arange(6000,dtype=np.float32)/6000
        first = prediction_for(windows[0],samples)
        second = prediction_for(windows[1],samples)+0.25
        actual, records = stitch(windows,[first,second])
        self.assertEqual(len(actual), len(samples))
        start,end = records[0]['start_sample'],records[0]['end_sample']
        np.testing.assert_array_equal(actual[:start],samples[:start])
        np.testing.assert_array_equal(actual[end:],samples[end:]+0.25)
        self.assertEqual(float(actual[start]), float(samples[start]))
        self.assertEqual(float(actual[end-1]),float(samples[end-1]+0.25))
        self.assertTrue(np.all(actual[start:end]>=samples[start:end]))
        self.assertTrue(np.all(actual[start:end]<=samples[start:end]+0.25))

    def test_tiny_adjacent_cores_clamp_fades_without_losing_samples(self):
        windows = tiny_windows([1,2,3,10,17,200,1])
        samples = np.arange(windows[-1].end,dtype=np.float32)
        widths = plan_crossfades(windows)
        self.assertEqual(widths[0],0)
        actual,records = stitch(windows,[prediction_for(w,samples) for w in windows])
        np.testing.assert_array_equal(actual,samples)
        self.assertEqual(len(records),sum(h>0 for h in widths))
        self.assertTrue(all(record['samples']<=20 for record in records))

    def test_wrong_absolute_overlap_is_rejected(self):
        windows = tiny_windows([1000,1000])
        first=np.zeros(windows[0].input_samples,dtype=np.float32)
        second=np.zeros(windows[1].input_samples,dtype=np.float32)
        _,pending,_ = blend_window(first,windows[0],0,240)
        with self.assertRaisesRegex(ValueError,'时间位置'):
            blend_window(second,windows[1],240,0,(pending[0]+1,pending[1]))
        with self.assertRaisesRegex(ValueError,'时间位置'):
            blend_window(second,windows[1],240,0,None)

    def test_only_small_trailing_token_error_is_corrected_before_blending(self):
        windows=plan_windows(30*SAMPLE_RATE+17)
        samples=np.random.default_rng(10).normal(size=windows[-1].end).astype(np.float32)
        predictions=[]
        for w,delta in zip(windows,(960,-960)):
            prediction=prediction_for(w,samples)
            changed=(np.pad(prediction,(0,delta)) if delta>0 else prediction[:delta])
            aligned,actual_delta=align_prediction(changed,w.input_samples,4800)
            self.assertEqual(actual_delta,delta)
            predictions.append(aligned)
        actual,_=stitch(windows,predictions)
        np.testing.assert_array_equal(actual,samples)
        with self.assertRaisesRegex(ValueError,'超过容差'):
            align_prediction(np.zeros(6000,dtype=np.float32),1000,4800)


if __name__ == '__main__':
    unittest.main()
