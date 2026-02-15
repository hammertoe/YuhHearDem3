import { describe, expect, it } from 'vitest';

import type { ChatSource } from '../api';
import { groupSourcesByDocument } from '../sourceGrouping';

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

describe('groupSourcesByDocument', () => {
  it('groups bill excerpts and video sources separately', () => {
    const sources: ChatSource[] = [
      baseSource({
        source_kind: 'bill_excerpt',
        bill_id: 'B1',
        bill_title: 'Public Service (Appointments) Bill, 2025',
        source_url: 'https://example.com/bill.pdf#page=5',
        page_number: 5,
      }),
      baseSource({
        source_kind: 'bill_excerpt',
        bill_id: 'B1',
        bill_title: 'Public Service (Appointments) Bill, 2025',
        source_url: 'https://example.com/bill.pdf#page=7',
        page_number: 7,
      }),
      baseSource({
        source_kind: 'utterance',
        youtube_video_id: 'vid-123',
        youtube_url: 'https://youtube.com/watch?v=vid-123&t=20',
        video_title: 'The Honourable The Senate - Part 1',
        speaker_name: 'Shantal Munro-Knight',
      }),
    ];

    const groups = groupSourcesByDocument(sources);

    expect(groups).toHaveLength(2);
  });

  it('sorts utterance sources by timestamp within a group', () => {
    const sources: ChatSource[] = [
      baseSource({
        source_kind: 'utterance',
        youtube_video_id: 'vid-123',
        youtube_url: 'https://youtube.com/watch?v=vid-123&t=120',
        seconds_since_start: 120,
        speaker_name: 'Speaker A',
      }),
      baseSource({
        source_kind: 'utterance',
        youtube_video_id: 'vid-123',
        youtube_url: 'https://youtube.com/watch?v=vid-123&t=10',
        seconds_since_start: 10,
        speaker_name: 'Speaker B',
      }),
    ];

    const groups = groupSourcesByDocument(sources);

    expect(groups[0]?.items.map((item) => item.source.seconds_since_start)).toEqual([10, 120]);
    expect(groups[0]?.items.map((item) => item.index)).toEqual([1, 2]);
  });

  it('sorts bill excerpts by page number within a group', () => {
    const sources: ChatSource[] = [
      baseSource({
        source_kind: 'bill_excerpt',
        bill_id: 'B1',
        bill_title: 'Public Service (Appointments) Bill, 2025',
        source_url: 'https://example.com/bill.pdf#page=12',
        page_number: 12,
      }),
      baseSource({
        source_kind: 'bill_excerpt',
        bill_id: 'B1',
        bill_title: 'Public Service (Appointments) Bill, 2025',
        source_url: 'https://example.com/bill.pdf#page=3',
        page_number: 3,
      }),
    ];

    const groups = groupSourcesByDocument(sources);

    expect(groups[0]?.items.map((item) => item.source.page_number)).toEqual([3, 12]);
    expect(groups[0]?.items.map((item) => item.index)).toEqual([1, 2]);
  });
});
