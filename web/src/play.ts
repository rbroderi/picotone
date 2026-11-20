import type { DecodedAudio } from './decoder.js';

/** Options of {@link play}. */
export interface PlayOptions {
  /** Where to connect the source (default `context.destination`). */
  readonly destination?: AudioNode;
  /** Context time to start at (default now). */
  readonly when?: number;
  /**
   * Loop the whole buffer (default: `audio.loop`, true for files made with
   * `encodec-rs encode --loop`, whose end runs into their start without a click).
   */
  readonly loop?: boolean;
}

/** Plays decoded audio through an `AudioBufferSourceNode` and returns it (call `stop()` to end a loop). */
export function play(context: BaseAudioContext, audio: DecodedAudio, options: PlayOptions = {}): AudioBufferSourceNode {
  const buffer = context.createBuffer(audio.channels.length, Math.max(1, audio.length), audio.sampleRate);
  audio.channels.forEach((ch, c) => buffer.copyToChannel(new Float32Array(ch), c));
  const source = context.createBufferSource();
  source.buffer = buffer;
  // No loopStart/loopEnd: the whole buffer is the loop.
  source.loop = options.loop ?? audio.loop;
  source.connect(options.destination ?? context.destination);
  source.start(options.when ?? context.currentTime);
  return source;
}
