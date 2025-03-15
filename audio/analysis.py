import numpy as np 
import librosa
from scipy.signal import find_peaks
import logging 

logger = logging.getLogger(__name__)

def analyze_audio_patterns(audio_buffer, sample_rate):
    """
    Analyzing the audio for patterns that indicate concentration
    Returns a score between 0-1 (normalized)
    """
    logger.debug(f"Analyzing audio patterns from buffer of length {len(audio_buffer)}")
    
    if isinstance(audio_buffer, bytes):
        samples = np.frombuffer(audio_buffer, dtype=np.int16)
    else:
        samples = audio_buffer
        
    # Normalizing
    samples = samples.astype(np.float32) / 32768.0
    
    # Audio Features - spectral centroid
    spec_score = 0.5
    if len(samples) > 512:
        try:
            spec_cent = librosa.feature.spectral_centroid(y=samples, sr=sample_rate)[0]
            spec_cent_norm = np.mean(spec_cent) / 4000  # Normalizing
            spec_score = 1 - min(1.0, spec_cent_norm)
        except Exception as e:
            logger.warning(f"Error calculating spectral centroid: {e}")
    
    # Rhythm regularity through autocorrelation
    rhythm_score = 0.5
    if len(samples) > 1024:
        try:
            corr = np.correlate(samples, samples, mode='full')
            corr = corr[len(corr)//2:]
            
            peaks, _ = find_peaks(corr, height=0.1*np.max(corr))
            
            if len(peaks) > 2:
                peak_intervals = np.diff(peaks)
                rhythm_regularity = 1 - np.std(peak_intervals) / np.mean(peak_intervals)
                rhythm_score = max(0, min(1, rhythm_regularity))
        except Exception as e:
            logger.warning(f"Error calculating rhythm regularity: {e}")
    
    # Sound consistency
    consistency_score = 0.5
    try:
        amplitude_envelope = np.abs(samples)
        amp_std = np.std(amplitude_envelope)
        consistency_score = 1 - min(1.0, amp_std * 10)
    except Exception as e:
        logger.warning(f"Error calculating sound consistency: {e}")
    
    # Combining scores
    final_score = 0.3 * spec_score + 0.3 * rhythm_score + 0.4 * consistency_score
    logger.debug(f"Audio analysis scores: spec={spec_score:.2f}, rhythm={rhythm_score:.2f}, consistency={consistency_score:.2f}, final={final_score:.2f}")
    
    return final_score
    