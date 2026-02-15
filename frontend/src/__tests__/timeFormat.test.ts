import { describe, expect, it } from 'vitest';

import type { ChatSource } from '../api';
import { formatSourceTimecode } from '../timeFormat';

const baseSource = (overrides: Partial<ChatSource>): ChatSource => ({
  source_kind: 'utterance',
  citation_id: null,
  utterance_id: '',
  youtube_video_id: '',
  youtube_url: '',
  seconds_since_start: 0,
  timestamp_str: '',
  speaker_id: '',
  speaker_name: '',
  speaker_title: null,
  text: '',
  video_title: null,
  video_date: null,
  bill_id: null,
  bill_number: null,
  bill_title: null,
  excerpt: null,
  source_url: null,
  chunk_index: null,
  page_number: null,
  matched_terms: null,
  ...overrides,
});

describe('formatSourceTimecode', () => {
  it('formats using seconds_since_start when available', () => {
    const source = baseSource({ seconds_since_start: 4000, timestamp_str: '46:17:400' });

    expect(formatSourceTimecode(source)).toBe('01:06:40');
  });

  it('falls back to youtube_url t param when seconds are missing', () => {
    const source = baseSource({
      seconds_since_start: 0,
      timestamp_str: '49:9:00',
      youtube_url: 'https://www.youtube.com/watch?v=abc123&t=2948',
    });

    expect(formatSourceTimecode(source)).toBe('00:49:08');
  });

  it('parses timestamp_str when seconds_since_start is missing', () => {
    const source = baseSource({ seconds_since_start: 0, timestamp_str: '1:06:48' });

    expect(formatSourceTimecode(source)).toBe('01:06:48');
  });

  it('returns null for invalid timestamps with missing seconds', () => {
    const source = baseSource({ seconds_since_start: 0, timestamp_str: '46:17:400' });

    expect(formatSourceTimecode(source)).toBe('00:46:17');
  });

  it('treats large leading values as minutes when parsing three-part timestamps', () => {
    const source = baseSource({ seconds_since_start: 0, timestamp_str: '49:08:00' });

    expect(formatSourceTimecode(source)).toBe('00:49:08');
  });

  it('treats long leading values as minutes even without millis', () => {
    const source = baseSource({ seconds_since_start: 0, timestamp_str: '46:23:40' });

    expect(formatSourceTimecode(source)).toBe('00:46:23');
  });
});
