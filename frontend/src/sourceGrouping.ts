import type { ChatSource } from './api';

export type SourceGroupItem = {
  source: ChatSource;
  index: number;
  source_key: string;
};

export type SourceGroup = {
  key: string;
  kind: 'utterance' | 'bill_excerpt';
  title: string;
  subtitle: string | null;
  items: SourceGroupItem[];
};

function normalizeKeyPart(value: string | null | undefined): string {
  return String(value || '').trim();
}

function buildSourceKey(source: ChatSource): string {
  if (source.source_kind === 'bill_excerpt') {
    return (
      normalizeKeyPart(source.citation_id) ||
      `bill:${normalizeKeyPart(source.bill_id)}:${Number(source.chunk_index || 0)}`
    );
  }
  return normalizeKeyPart(source.utterance_id);
}

function groupKeyForSource(source: ChatSource): {
  key: string;
  title: string;
  subtitle: string | null;
  kind: 'utterance' | 'bill_excerpt';
} {
  if (source.source_kind === 'bill_excerpt') {
    const billId = normalizeKeyPart(source.bill_id);
    const fallback = normalizeKeyPart(source.source_url) || normalizeKeyPart(source.bill_title);
    const title = normalizeKeyPart(source.bill_title) || normalizeKeyPart(source.bill_number) || 'Bill PDF';
    const subtitle =
      normalizeKeyPart(source.bill_number) && title !== normalizeKeyPart(source.bill_number)
        ? normalizeKeyPart(source.bill_number)
        : null;
    return {
      key: `bill:${billId || fallback || title}`,
      title,
      subtitle,
      kind: 'bill_excerpt',
    };
  }

  const videoId = normalizeKeyPart(source.youtube_video_id);
  const fallback = normalizeKeyPart(source.youtube_url) || normalizeKeyPart(source.video_title);
  const title = normalizeKeyPart(source.video_title) || 'Parliamentary Debate';
  const subtitle = normalizeKeyPart(source.video_date) || null;
  return {
    key: `video:${videoId || fallback || title}`,
    title,
    subtitle,
    kind: 'utterance',
  };
}

export function groupSourcesByDocument(sources: ChatSource[]): SourceGroup[] {
  const groups: SourceGroup[] = [];
  const indexByKey = new Map<string, number>();

  sources.forEach((source) => {
    const groupKey = groupKeyForSource(source);
    const sourceKey = buildSourceKey(source);
    let groupIndex = indexByKey.get(groupKey.key);

    if (groupIndex === undefined) {
      groupIndex = groups.length;
      indexByKey.set(groupKey.key, groupIndex);
      groups.push({
        key: groupKey.key,
        kind: groupKey.kind,
        title: groupKey.title,
        subtitle: groupKey.subtitle,
        items: [],
      });
    }

    groups[groupIndex].items.push({
      source,
      index: 0,
      source_key: sourceKey,
    });
  });

  groups.forEach((group) => {
    if (group.kind === 'bill_excerpt') {
      group.items.sort((a, b) => {
        const aPage = Number(a.source.page_number || 0);
        const bPage = Number(b.source.page_number || 0);
        if (aPage !== bPage) return aPage - bPage;
        const aChunk = Number(a.source.chunk_index || 0);
        const bChunk = Number(b.source.chunk_index || 0);
        if (aChunk !== bChunk) return aChunk - bChunk;
        return a.source_key.localeCompare(b.source_key);
      });
      return;
    }

    group.items.sort((a, b) => {
      const aSeconds = Number(a.source.seconds_since_start || 0);
      const bSeconds = Number(b.source.seconds_since_start || 0);
      if (aSeconds !== bSeconds) return aSeconds - bSeconds;
      return a.source_key.localeCompare(b.source_key);
    });
  });

  let globalIndex = 1;
  groups.forEach((group) => {
    group.items.forEach((item) => {
      item.index = globalIndex;
      globalIndex += 1;
    });
  });

  return groups;
}
