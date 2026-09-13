/**
 * deinterleaveToOutput (#4965)
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 *
 * A pure function (#4301) — no AudioContext needed. Existing coverage is
 * entirely indirect via BufferScheduler.test.ts, which only reaches the
 * mono and full-supply-stereo branches. This covers the two branches that
 * were previously untested: the stereo silence-padding tail (the guard
 * against the #4331 buffer-underrun regression) and the 3+-channel branch.
 */

import { describe, it, expect } from 'vitest';
import { deinterleaveToOutput, type ChannelWritableOutput } from '../deinterleaveToOutput';

/** A fake Web Audio output: one Float32Array per channel, zero-initialized —
 * matches how a real AudioBuffer's channel data starts (silence). */
function fakeOutput(numberOfChannels: number, frames: number): ChannelWritableOutput {
  const channels = Array.from({ length: numberOfChannels }, () => new Float32Array(frames));
  return {
    numberOfChannels,
    getChannelData: (channel: number) => channels[channel],
  };
}

describe('deinterleaveToOutput', () => {
  describe('mono source', () => {
    it('copies the same mono data into every output channel', () => {
      const output = fakeOutput(2, 4);
      // Exactly float32-representable values (0.5 is a power of two) so the
      // Float32Array round-trip can't introduce a rounding mismatch.
      const samples = new Float32Array([0.5, -0.5, 0.25, -0.25]);

      deinterleaveToOutput(output, samples, 4, 1);

      expect(Array.from(output.getChannelData(0))).toEqual([0.5, -0.5, 0.25, -0.25]);
      expect(Array.from(output.getChannelData(1))).toEqual([0.5, -0.5, 0.25, -0.25]);
    });
  });

  describe('stereo source', () => {
    it('de-interleaves left/right correctly on a full supply', () => {
      const output = fakeOutput(2, 3);
      // L0 R0 L1 R1 L2 R2
      const samples = new Float32Array([1, -1, 2, -2, 3, -3]);

      deinterleaveToOutput(output, samples, 3, 2);

      expect(Array.from(output.getChannelData(0))).toEqual([1, 2, 3]);
      expect(Array.from(output.getChannelData(1))).toEqual([-1, -2, -3]);
    });

    it('zero-fills the tail exactly when fewer frames are supplied than needed (#4331)', () => {
      const output = fakeOutput(2, 5);
      // Only 2 full stereo frames supplied (4 samples), 5 requested.
      const samples = new Float32Array([1, -1, 2, -2]);
      // Pre-seed the output with non-zero "garbage" so a correct zero-fill
      // is actually exercised, not just coincidentally already zero.
      output.getChannelData(0).fill(99);
      output.getChannelData(1).fill(99);

      deinterleaveToOutput(output, samples, 5, 2);

      const left = output.getChannelData(0);
      const right = output.getChannelData(1);
      expect(Array.from(left)).toEqual([1, 2, 0, 0, 0]);
      expect(Array.from(right)).toEqual([-1, -2, 0, 0, 0]);
    });

    it('zero-fills entirely when zero frames are supplied', () => {
      const output = fakeOutput(2, 3);
      output.getChannelData(0).fill(99);
      output.getChannelData(1).fill(99);

      deinterleaveToOutput(output, new Float32Array(0), 3, 2);

      expect(Array.from(output.getChannelData(0))).toEqual([0, 0, 0]);
      expect(Array.from(output.getChannelData(1))).toEqual([0, 0, 0]);
    });

    it('treats a trailing incomplete frame as absent rather than reading past it', () => {
      // 2 full frames + one dangling left-only sample (5 raw values) --
      // framesToProcess must floor to 2, not round up and read undefined.
      const output = fakeOutput(2, 3);
      const samples = new Float32Array([1, -1, 2, -2, 3]);

      deinterleaveToOutput(output, samples, 3, 2);

      expect(Array.from(output.getChannelData(0))).toEqual([1, 2, 0]);
      expect(Array.from(output.getChannelData(1))).toEqual([-1, -2, 0]);
    });
  });

  describe('multichannel (3+) source', () => {
    it('deinterleaves a 3-channel source correctly', () => {
      const output = fakeOutput(3, 2);
      // Frame0: ch0=1 ch1=2 ch2=3 ; Frame1: ch0=4 ch1=5 ch2=6
      const samples = new Float32Array([1, 2, 3, 4, 5, 6]);

      deinterleaveToOutput(output, samples, 2, 3);

      expect(Array.from(output.getChannelData(0))).toEqual([1, 4]);
      expect(Array.from(output.getChannelData(1))).toEqual([2, 5]);
      expect(Array.from(output.getChannelData(2))).toEqual([3, 6]);
    });

    it('deinterleaves a 4-channel source and pads a short final frame with silence', () => {
      const output = fakeOutput(4, 2);
      // Frame0 complete (4 samples); frame1 only supplies 2 of 4 channels.
      const samples = new Float32Array([1, 2, 3, 4, 5, 6]);
      output.getChannelData(2).fill(99);
      output.getChannelData(3).fill(99);

      deinterleaveToOutput(output, samples, 2, 4);

      expect(Array.from(output.getChannelData(0))).toEqual([1, 5]);
      expect(Array.from(output.getChannelData(1))).toEqual([2, 6]);
      expect(Array.from(output.getChannelData(2))).toEqual([3, 0]);
      expect(Array.from(output.getChannelData(3))).toEqual([4, 0]);
    });

    it('handles an output with fewer channels than the source provides', () => {
      // e.g. a 5.1 source down to a stereo-shaped output buffer -- the
      // multichannel branch only ever writes output.numberOfChannels worth.
      const output = fakeOutput(2, 2);
      const samples = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);

      deinterleaveToOutput(output, samples, 2, 6);

      expect(Array.from(output.getChannelData(0))).toEqual([1, 7]);
      expect(Array.from(output.getChannelData(1))).toEqual([2, 8]);
    });
  });
});
