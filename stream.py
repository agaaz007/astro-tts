import boto3
import json
import sounddevice as sd
import threading
import time
import numpy as np
import sys
import collections
import sounddevice as sd
print("Default I/O devices:", sd.default.device)
print(sd.query_devices()) 

SAMPLE_RATE = 44100
sd.default.samplerate = SAMPLE_RATE
sd.default.channels = 1

# Prime playback once ~0.3 s (≈ 13 k samples) have been buffered instead of waiting
# a full second. This lets audio begin much sooner while still avoiding underruns.
MIN_BUFFER_SECONDS = 0.7
# BLOCKSIZE          = 1024        # exactly what ALSA/CoreAudio will request
MIN_BUFFER_SIZE = int(SAMPLE_RATE * MIN_BUFFER_SECONDS)


class StreamingAudioPlayer:
    """
    A high-performance, thread-safe audio player using a deque of audio chunks.
    This avoids costly memory re-allocations and copying, preventing audio jitter.
    """
    def __init__(self, sample_rate=SAMPLE_RATE, channels=1):
        self.sample_rate = sample_rate
        self.channels = channels
        self._buffer = collections.deque()
        self._buffer_lock = threading.Lock()
        self._total_samples_in_buffer = 0
        self._local_chunk = np.array([], dtype=np.float32)
        self.download_complete = threading.Event()
        self.playback_started = threading.Event()
        self.stream = None

    def add_audio_chunk(self, audio_data):
        """Adds a numpy audio array to the deque in a thread-safe way."""
        with self._buffer_lock:
            self._buffer.append(audio_data)
            self._total_samples_in_buffer += len(audio_data)

    def _audio_callback(self, outdata, frames, time, status):
        """Pulls audio from the deque."""
        if status:
            print(f"Playback status warning: {status}", file=sys.stderr)

        # Wait until a small safety buffer is available before starting playback.
        if not self.playback_started.is_set():
            with self._buffer_lock:
                buffered = self._total_samples_in_buffer
            if buffered < MIN_BUFFER_SIZE:
                outdata.fill(0)
                return
            else:
                print(f"Buffer primed with {buffered} samples. Starting playback...")
                self.playback_started.set()

        frames_filled = 0
        while frames_filled < frames:
            if len(self._local_chunk) > 0:
                needed = frames - frames_filled
                to_copy = min(needed, len(self._local_chunk))
                outdata[frames_filled : frames_filled + to_copy] = self._local_chunk[:to_copy].reshape(-1, 1)
                self._local_chunk = self._local_chunk[to_copy:]
                frames_filled += to_copy
            else:
                try:
                    with self._buffer_lock:
                        self._local_chunk = self._buffer.popleft()
                        self._total_samples_in_buffer -= len(self._local_chunk)
                except IndexError:
                    if self.download_complete.is_set():
                        outdata[frames_filled:].fill(0)
                        raise sd.CallbackStop
                    else:
                        outdata[frames_filled:].fill(0)
                        return
        if frames_filled < frames:
            outdata[frames_filled:].fill(0)

    def start_playback(self):
        """Starts the sounddevice output stream."""
        self.download_complete.clear()
        self.playback_started.clear()
        print(f"Audio player started. Pre-buffering {MIN_BUFFER_SECONDS}s of audio...")
        self.stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype='float32',
            # blocksize=BLOCKSIZE,         # 🆕
            latency='low',               # 🆕 keep Round-Trip under ~50 ms
            callback=self._audio_callback
        )
        self.stream.start()

    # --- NEW: Simple method to signal download is done ---
    def signal_download_complete(self):
        """Tells the player that no more chunks will be added."""
        self.download_complete.set()
        print("📡 Download stream finished.")

    # --- MODIFIED: This method now just waits for the stream to finish ---
    def wait_for_completion(self):
        """Waits for the audio buffer to be fully played, then cleans up."""
        if not self.download_complete.is_set():
            print("Warning: wait_for_completion() called before download was signaled as complete.", file=sys.stderr)

        print("Waiting for audio buffer to clear...")
        if self.stream:
            while self.stream.active:
                time.sleep(0.1)
            self.stream.stop(ignore_errors=True)
            self.stream.close(ignore_errors=True)
            print("🔊 Audio player stopped.")

# (SageMakerStreamer class is unchanged)
class SageMakerStreamer:
    def __init__(self, endpoint_name, region='ap-south-1'):
        self.endpoint_name = endpoint_name
        self.client = boto3.client('sagemaker-runtime', region_name=region)

    def invoke_streaming(self, prompt, description):
        payload = {"prompt": prompt, "description": description}
        print(f"Sending to endpoint: {self.endpoint_name}")
        response = self.client.invoke_endpoint_with_response_stream(
            EndpointName=self.endpoint_name,
            ContentType='application/json',
            Body=json.dumps(payload),
            Accept='audio/pcm'
        )
        event_stream = response['Body']
        for event in event_stream:
            chunk = event.get('PayloadPart', {}).get('Bytes')
            if chunk:
                yield chunk

    def process_audio_chunk(self, audio_bytes):
        if not audio_bytes:
            return None
        audio_arr = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        return audio_arr

# --- MODIFIED: The download thread now has a simpler responsibility ---
def download_and_process(streamer, player, prompt, desc):
    """Downloads audio and adds it to the player's buffer."""
    start_time = time.time()
    try:
        audio_stream = streamer.invoke_streaming(prompt, desc)
        print("Audio stream opened. Receiving data...")
        first_chunk_received = False
        
        for chunk_bytes in audio_stream:
            if not first_chunk_received:
                print(f"⏱️ Time to first chunk: {time.time() - start_time:.2f}s")
                first_chunk_received = True

            audio_data = streamer.process_audio_chunk(chunk_bytes)
            
            if audio_data is not None and audio_data.size > 0:
                player.add_audio_chunk(audio_data)

    except Exception as e:
        print(f"❌ Download error: {e}", file=sys.stderr)
    finally:
        # The download thread's ONLY job at the end is to signal it's done.
        player.signal_download_complete()

# --- MODIFIED: The main thread now orchestrates the final shutdown ---
def main():
    ENDPOINT_NAME = "frijun"
    streamer = SageMakerStreamer(ENDPOINT_NAME)
    player = StreamingAudioPlayer()

    try:
        prompt = input("Prompt: ").strip() or "This is a test of a real-time, low-latency, text-to-speech system."
        desc = input("Description: ").strip() or "A female is the speaker with a clear voice"
        print("-" * 20)

        player.start_playback()

        download_thread = threading.Thread(
            target=download_and_process,
            args=(streamer, player, prompt, desc)
        )
        download_thread.start()
        print("🚀 Download thread started. Audio should begin shortly...")

        # Wait for the download thread to finish its job.
        download_thread.join()
        
        # NOW, after the download is fully complete, tell the player to wait
        # for its buffer to drain and then stop.
        player.wait_for_completion()
        
        print("✅ All done.")

    except KeyboardInterrupt:
        print("\nInterrupted by user. Stopping.")
        if player.stream and player.stream.active:
             player.stream.stop(ignore_errors=True)
             player.stream.close(ignore_errors=True)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# Add this after setting sd.default.device to test audio output
def test_audio():
    print("Testing audio output...")
    # Generate a simple sine wave
    duration = 1  # seconds
    frequency = 440  # Hz
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)
    test_tone = 0.5 * np.sin(2 * np.pi * frequency * t)
    
    try:
        sd.play(test_tone, SAMPLE_RATE)
        sd.wait()
        print("Audio test complete - Did you hear a beep?")
    except Exception as e:
        print(f"Audio test failed: {e}")

# Add this test call before your main() function
if __name__ == "__main__":
    test_audio()
    main()