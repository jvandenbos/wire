# Wire - Speaker Polarity Test Tool

## Project Overview
Python tool that tests speaker polarity/phase alignment by playing test tones through L/R speakers individually, recording via microphone, and analyzing phase correlation. Similar to AV receiver auto-calibration features.

## Tech Stack
- Python 3
- numpy - signal generation and array operations
- scipy.signal - bandpass filtering (SOS for numerical stability), cross-correlation
- sounddevice - audio playback and recording (playrec for simultaneous I/O)
- matplotlib - waveform visualization

## Architecture
1. **Tone generation**: Sine wave bursts with fade envelope to prevent clicks
2. **Playback sequence**: silence → left tone → silence → right tone → silence
3. **Recording**: Mono mic capture during playback
4. **Filtering**: Bandpass filter (SOS) isolates test frequency, critical for low frequencies (<100Hz)
5. **Analysis**: Cross-correlation finds best alignment, returns phase correlation (-1 to +1)
6. **Visualization**: Overlay plots + sum/difference analysis

## Key Technical Notes
- Use `signal.butter(..., output='sos')` + `sosfiltfilt()` for low-frequency stability (not b,a coefficients)
- Test frequencies: 80-250 Hz range works well for polarity detection
- Correlation > +0.3 = in phase, < -0.3 = out of phase
- Subwoofers can cause phase anomalies around crossover frequency

## Files
- `speaker_polarity_test.py` - main script, run directly
- `polarity_test_waveforms.png` - generated output (gitignored)
