import boto3
import json
import sounddevice as sd
import threading
import time
import numpy as np
import sys
import collections # Import the collections library for deque

SAMPLE_RATE = 44100
sd.default.samplerate = SAMPLE_RATE
sd.default.channels = 1

# --- Configuration for pre-buffering ---
MIN_BUFFER_SECONDS = 0.5 # A slightly larger buffer can help with network jitter
MIN_BUFFER_SIZE = int(SAMPLE_RATE * MIN_BUFFER_SECONDS)


class StreamingAudioPlayer:
    """
    A high-performance, thread-safe audio player using a deque of audio chunks.
    This avoids costly memory re-allocations and copying, preventing audio jitter.
    """
    def __init__(self, sample_rate=SAMPLE_RATE, channels=1):
        self.sample_rate = sample_rate
        self.channels = channels
        # --- NEW: Use a deque for efficient appends and pops ---
        self._buffer = collections.deque()
        self._buffer_lock = threading.Lock()
        self._total_samples_in_buffer = 0  # Keep track of total samples
        self._local_chunk = np.array([], dtype=np.float32) # For leftovers
        self.download_complete = threading.Event()
        self.playback_started = threading.Event()
        self.stream = None

    def add_audio_chunk(self, audio_data):
        """Adds a numpy audio array to the deque in a thread-safe way."""
        with self._buffer_lock:
            self._buffer.append(audio_data)
            self._total_samples_in_buffer += len(audio_data)

    def _audio_callback(self, outdata, frames, time, status):
        """
        Pulls audio from the deque. This is more complex than the simple
        array slicing but is far more performant.
        """
        if status:
            print(f"Playback status warning: {status}", file=sys.stderr)

        # Pre-buffering check
        if not self.playback_started.is_set():
            if self._total_samples_in_buffer < MIN_BUFFER_SIZE:
                outdata.fill(0)
                return
            else:
                print(f"Buffer primed with {self._total_samples_in_buffer} samples. Starting playback...")
                self.playback_started.set()

        frames_filled = 0
        while frames_filled < frames:
            # Do we have any leftover audio from the previous callback?
            if len(self._local_chunk) > 0:
                needed = frames - frames_filled
                to_copy = min(needed, len(self._local_chunk))
                outdata[frames_filled : frames_filled + to_copy] = self._local_chunk[:to_copy].reshape(-1, 1)
                self._local_chunk = self._local_chunk[to_copy:]
                frames_filled += to_copy
            else:
                # No leftovers, get a new chunk from the main buffer
                try:
                    with self._buffer_lock:
                        self._local_chunk = self._buffer.popleft()
                        self._total_samples_in_buffer -= len(self._local_chunk)
                except IndexError:
                    # Buffer is empty
                    if self.download_complete.is_set():
                        # Fill the rest with silence and stop
                        outdata[frames_filled:].fill(0)
                        raise sd.CallbackStop
                    else:
                        # Waiting for more data, play silence for now
                        outdata[frames_filled:].fill(0)
                        return # Exit the callback for now

        # This should not be reached if logic is correct, but as a safeguard
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
            callback=self._audio_callback
        )
        self.stream.start()

    def stop_playback(self):
        """Waits for the stream to finish and then cleans up."""
        print("Waiting for audio buffer to clear...")
        self.download_complete.set()
        if self.stream:
            # Wait for the stream to become inactive
            while self.stream.active:
                time.sleep(0.1)
            self.stream.stop(ignore_errors=True)
            self.stream.close(ignore_errors=True)
            print("🔊 Audio player stopped.")

# (The SageMakerStreamer class and main functions remain the same)
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


def download_and_process(streamer, player, prompt, desc):
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
        print("📡 Download stream finished.")
        player.stop_playback()

def main():
    ENDPOINT_NAME = "frijun"
    streamer = SageMakerStreamer(ENDPOINT_NAME)
    player = StreamingAudioPlayer()

    try:
        prompt = input("Prompt: ").strip() or "This is a test of a real-time, low-latency, text-to-speech system."
        desc = input("Description: ").strip() or "A female speaker with a clear voice"
        print("-" * 20)

        player.start_playback()

        download_thread = threading.Thread(
            target=download_and_process,
            args=(streamer, player, prompt, desc)
        )
        download_thread.start()
        print("🚀 Download thread started. Audio should begin shortly...")

        download_thread.join()
        
        print("✅ All done.")

    except KeyboardInterrupt:
        print("\nInterrupted by user. Stopping.")
        if player.stream and player.stream.active:
             player.stream.stop(ignore_errors=True)
             player.stream.close(ignore_errors=True)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()