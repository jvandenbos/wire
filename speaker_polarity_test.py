#!/usr/bin/env python3
"""
Speaker Polarity Test
Tests if left and right speakers have matching polarity by playing test tones
and analyzing the phase relationship of the recorded signals.
"""

import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from scipy import signal

# Audio settings
SAMPLE_RATE = 44100
TEST_FREQUENCIES = [80, 120, 160, 200, 250]  # Hz - multiple low frequencies
DURATION = 0.4   # seconds per tone burst
PAUSE = 0.25     # seconds between tests
AMPLITUDE = 0.7

def generate_tone_burst(frequency, duration, sample_rate, amplitude=0.7):
    """Generate a sine wave tone burst with smooth envelope."""
    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
    # Apply fade in/out to avoid clicks
    envelope = np.ones_like(t)
    fade_samples = int(0.01 * sample_rate)  # 10ms fade
    envelope[:fade_samples] = np.linspace(0, 1, fade_samples)
    envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)
    return amplitude * envelope * np.sin(2 * np.pi * frequency * t)

def play_and_record(stereo_audio, sample_rate):
    """Play stereo audio and record from microphone simultaneously."""
    recording = sd.playrec(
        stereo_audio,
        samplerate=sample_rate,
        channels=1,  # mono recording
        dtype=np.float32
    )
    sd.wait()
    return recording.flatten()

def extract_tone_segment(recording, sample_rate, start_time, duration):
    """Extract a segment of the recording."""
    start_sample = int(start_time * sample_rate)
    end_sample = int((start_time + duration) * sample_rate)
    return recording[start_sample:end_sample]

def find_best_alignment(signal1, signal2, max_shift_samples=500):
    """Find the best alignment between two signals and return correlation and shift."""
    cross_corr = signal.correlate(signal1, signal2, mode='full')
    center = len(cross_corr) // 2

    # Look for peak within allowed shift range
    search_start = center - max_shift_samples
    search_end = center + max_shift_samples
    search_range = cross_corr[search_start:search_end]

    peak_idx = np.argmax(np.abs(search_range))
    peak_value = search_range[peak_idx]
    shift = peak_idx - max_shift_samples

    # Normalize
    norm = np.sqrt(np.sum(signal1**2) * np.sum(signal2**2))
    if norm > 0:
        return peak_value / norm, shift
    return 0, 0

def test_single_frequency(frequency):
    """Test polarity at a single frequency. Returns results dict."""
    tone = generate_tone_burst(frequency, DURATION, SAMPLE_RATE, AMPLITUDE)
    silence = np.zeros(int(PAUSE * SAMPLE_RATE), dtype=np.float32)

    # Create stereo signals: Left channel only, then Right channel only
    left_tone = np.column_stack([tone, np.zeros_like(tone)])
    right_tone = np.column_stack([np.zeros_like(tone), tone])

    # Combine into one sequence: silence, left, silence, right, silence
    full_sequence = np.vstack([
        np.column_stack([silence, silence]),
        left_tone,
        np.column_stack([silence, silence]),
        right_tone,
        np.column_stack([silence, silence])
    ]).astype(np.float32)

    # Play and record
    recording = play_and_record(full_sequence, SAMPLE_RATE)

    # Calculate timing for extraction
    left_start = PAUSE
    right_start = PAUSE + DURATION + PAUSE

    # Extract recorded segments for each speaker
    left_recording = extract_tone_segment(recording, SAMPLE_RATE, left_start, DURATION)
    right_recording = extract_tone_segment(recording, SAMPLE_RATE, right_start, DURATION)

    # Apply bandpass filter to isolate test tone
    # Use second-order sections (sos) for numerical stability at low frequencies
    nyquist = SAMPLE_RATE / 2
    low = max((frequency - 30) / nyquist, 0.002)
    high = min((frequency + 30) / nyquist, 0.999)
    sos = signal.butter(4, [low, high], btype='band', output='sos')

    left_filtered = signal.sosfiltfilt(sos, left_recording)
    right_filtered = signal.sosfiltfilt(sos, right_recording)

    # Calculate phase correlation
    correlation, shift = find_best_alignment(left_filtered, right_filtered)

    return {
        'frequency': frequency,
        'left_raw': left_recording,
        'right_raw': right_recording,
        'left_filtered': left_filtered,
        'right_filtered': right_filtered,
        'left_level': np.max(np.abs(left_filtered)),
        'right_level': np.max(np.abs(right_filtered)),
        'correlation': correlation,
        'shift_samples': shift,
        'shift_ms': (shift / SAMPLE_RATE) * 1000
    }

def plot_waveforms(results_list):
    """Create waveform visualization for all tested frequencies."""
    n_freq = len(results_list)

    fig, axes = plt.subplots(n_freq, 2, figsize=(14, 3 * n_freq))
    fig.suptitle('Speaker Polarity Test - Waveform Analysis', fontsize=14, fontweight='bold')

    if n_freq == 1:
        axes = axes.reshape(1, -1)

    for i, result in enumerate(results_list):
        freq = result['frequency']
        corr = result['correlation']

        # Time axis (show middle portion for clarity)
        n_samples = len(result['left_filtered'])
        start = n_samples // 4
        end = 3 * n_samples // 4
        t = np.arange(end - start) / SAMPLE_RATE * 1000  # ms

        left_sig = result['left_filtered'][start:end]
        right_sig = result['right_filtered'][start:end]

        # Normalize for display
        max_val = max(np.max(np.abs(left_sig)), np.max(np.abs(right_sig)), 0.0001)
        left_norm = left_sig / max_val
        right_norm = right_sig / max_val

        # Left plot: overlay both channels
        ax1 = axes[i, 0]
        ax1.plot(t, left_norm, 'b-', label='Left Speaker', linewidth=1.5, alpha=0.8)
        ax1.plot(t, right_norm, 'r-', label='Right Speaker', linewidth=1.5, alpha=0.8)
        ax1.set_ylabel('Amplitude')
        ax1.set_title(f'{freq} Hz - Overlay (correlation: {corr:+.3f})')
        ax1.legend(loc='upper right', fontsize=8)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(-1.3, 1.3)

        # Right plot: difference/sum analysis
        ax2 = axes[i, 1]

        # Align signals for comparison
        if result['shift_samples'] > 0:
            left_aligned = left_norm[result['shift_samples']:]
            right_aligned = right_norm[:len(left_aligned)]
        elif result['shift_samples'] < 0:
            right_aligned = right_norm[-result['shift_samples']:]
            left_aligned = left_norm[:len(right_aligned)]
        else:
            left_aligned = left_norm
            right_aligned = right_norm

        min_len = min(len(left_aligned), len(right_aligned))
        left_aligned = left_aligned[:min_len]
        right_aligned = right_aligned[:min_len]
        t_aligned = np.arange(min_len) / SAMPLE_RATE * 1000

        sum_sig = left_aligned + right_aligned
        diff_sig = left_aligned - right_aligned

        ax2.plot(t_aligned, sum_sig, 'g-', label='L + R (sum)', linewidth=1.5, alpha=0.8)
        ax2.plot(t_aligned, diff_sig, 'm-', label='L - R (diff)', linewidth=1.5, alpha=0.8)
        ax2.set_ylabel('Amplitude')
        ax2.set_title(f'{freq} Hz - Sum/Difference Analysis')
        ax2.legend(loc='upper right', fontsize=8)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(-2.5, 2.5)

        # Add phase indicator
        if corr > 0.3:
            phase_text = "IN PHASE ✓"
            color = 'green'
        elif corr < -0.3:
            phase_text = "OUT OF PHASE ✗"
            color = 'red'
        else:
            phase_text = "UNCLEAR"
            color = 'orange'

        ax2.text(0.98, 0.95, phase_text, transform=ax2.transAxes,
                fontsize=10, fontweight='bold', color=color,
                ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    axes[-1, 0].set_xlabel('Time (ms)')
    axes[-1, 1].set_xlabel('Time (ms)')

    plt.tight_layout()
    plt.savefig('polarity_test_waveforms.png', dpi=150, bbox_inches='tight')
    print("\nWaveform plot saved to: polarity_test_waveforms.png")
    plt.show()

def run_polarity_test():
    """Run the speaker polarity test at multiple frequencies."""
    print("=" * 60)
    print("SPEAKER POLARITY TEST - MULTI-FREQUENCY")
    print("=" * 60)
    print()
    print("Testing frequencies:", TEST_FREQUENCIES, "Hz")
    print()
    print("For best results:")
    print("  - Position yourself equidistant from both speakers")
    print("  - Reduce background noise")
    print("  - Set volume to a moderate level")
    print()

    input("Press Enter to begin the test...")
    print()

    results = []

    for i, freq in enumerate(TEST_FREQUENCIES):
        print(f"Testing {freq} Hz... ({i+1}/{len(TEST_FREQUENCIES)})")
        result = test_single_frequency(freq)
        results.append(result)
        print(f"  L:{result['left_level']:.4f}  R:{result['right_level']:.4f}  "
              f"Correlation: {result['correlation']:+.3f}")

    print()
    print("-" * 60)
    print("RESULTS SUMMARY")
    print("-" * 60)
    print()
    print(f"{'Freq (Hz)':<12} {'L Level':<10} {'R Level':<10} {'Correlation':<14} {'Status'}")
    print("-" * 60)

    correlations = []
    for r in results:
        if r['left_level'] > 0.001 and r['right_level'] > 0.001:
            correlations.append(r['correlation'])

        if r['correlation'] > 0.3:
            status = "IN PHASE ✓"
        elif r['correlation'] < -0.3:
            status = "OUT OF PHASE ✗"
        else:
            status = "UNCLEAR"

        print(f"{r['frequency']:<12} {r['left_level']:<10.4f} {r['right_level']:<10.4f} "
              f"{r['correlation']:+.3f}        {status}")

    print()
    print("=" * 60)

    if correlations:
        avg_corr = np.mean(correlations)
        print(f"AVERAGE CORRELATION: {avg_corr:+.3f}")
        print()

        if avg_corr > 0.3:
            print("OVERALL: PASS - Speakers are IN PHASE (correct polarity)")
        elif avg_corr < -0.3:
            print("OVERALL: FAIL - Speakers are OUT OF PHASE (reversed polarity)")
            print()
            print("One speaker likely has reversed polarity.")
            print("Check the speaker wire connections (+/- terminals).")
        else:
            print("OVERALL: INCONCLUSIVE")
            print("Try adjusting position or reducing noise.")

    print("=" * 60)

    # Generate waveform plots
    print("\nGenerating waveform visualization...")
    plot_waveforms(results)

def check_audio_devices():
    """List available audio devices."""
    print("Available audio devices:")
    print(sd.query_devices())
    print()
    print(f"Default input device:  {sd.query_devices(kind='input')['name']}")
    print(f"Default output device: {sd.query_devices(kind='output')['name']}")
    print()

if __name__ == "__main__":
    try:
        check_audio_devices()
        run_polarity_test()
    except KeyboardInterrupt:
        print("\nTest cancelled.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        print()
        print("Make sure you have the required packages installed:")
        print("  pip install numpy scipy sounddevice matplotlib")
