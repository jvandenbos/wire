# Wire - Speaker Polarity Test

A Python tool that tests if your left and right speakers have correct polarity (phase alignment) using your computer's microphone — similar to the auto-calibration feature found in AV receivers.

## What It Does

Plays test tones through each speaker individually, records them via microphone, and analyzes the phase relationship. If one speaker has reversed polarity (swapped +/- wires), the tool will detect it.

- Tests multiple frequencies (80, 120, 160, 200, 250 Hz) for reliability
- Generates waveform visualizations showing phase alignment
- Works with any stereo speaker setup (bookshelf speakers, studio monitors, AV systems)

## Installation

```bash
# Clone the repo
git clone https://github.com/jvandenbos/wire.git
cd wire

# Install dependencies
pip install numpy scipy sounddevice matplotlib
```

### macOS Note
You may need to grant Terminal/your IDE microphone access in System Preferences → Privacy & Security → Microphone.

## Usage

```bash
python speaker_polarity_test.py
```

Position yourself (and your Mac's microphone) roughly equidistant from both speakers, then follow the prompts.

## Interpreting Results

| Correlation | Meaning |
|-------------|---------|
| +0.7 to +1.0 | **IN PHASE** — correct polarity |
| -0.7 to -1.0 | **OUT OF PHASE** — one speaker has reversed polarity |
| -0.3 to +0.3 | **INCONCLUSIVE** — try reducing noise or repositioning |

If you get consistent negative correlation across frequencies, check your speaker wire connections — one speaker likely has the +/- terminals swapped.

## Example Output

```
Freq (Hz)    L Level    R Level    Correlation    Status
------------------------------------------------------------
80           0.1394     0.0814     +0.988        IN PHASE ✓
120          0.3052     0.1664     +0.996        IN PHASE ✓
160          0.3715     0.2093     +0.995        IN PHASE ✓
200          0.2418     0.1227     +0.994        IN PHASE ✓
250          0.2395     0.1155     +0.991        IN PHASE ✓

AVERAGE CORRELATION: +0.993
OVERALL: PASS - Speakers are IN PHASE (correct polarity)
```

The tool also generates a `polarity_test_waveforms.png` visualization showing the recorded waveforms and sum/difference analysis.

## Keywords

speaker polarity test, phase test, speaker phase checker, audio polarity, speaker wiring test, left right speaker test, speaker diagnostic, home theater calibration, stereo phase alignment
