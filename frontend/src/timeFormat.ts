import type { ChatSource } from './api';

function formatSeconds(totalSeconds: number): string {
  const safe = Math.max(0, Math.floor(totalSeconds));
  const hours = Math.floor(safe / 3600);
  const minutes = Math.floor((safe % 3600) / 60);
  const seconds = safe % 60;
  const pad = (value: number) => String(value).padStart(2, '0');

  return `${pad(hours)}:${pad(minutes)}:${pad(seconds)}`;
}

function parseTimestampToSeconds(raw: string): number | null {
  const value = String(raw || '').trim();
  if (!value) return null;

  const sanitized = value.replace(/[,]/g, '.');
  const parts = sanitized.split(':').map((part) => part.trim());
  if (parts.length < 2 || parts.length > 3) return null;

  const last = parts[parts.length - 1] || '';
  const [secPart] = last.split('.');
  if (!secPart || !/^[0-9]+$/.test(secPart)) return null;

  const secondsRaw = Number(secPart);
  if (!Number.isFinite(secondsRaw) || secondsRaw < 0) return null;

  const minutesPart = parts[parts.length - 2] || '';
  if (!/^[0-9]+$/.test(minutesPart)) return null;
  const minutesRaw = Number(minutesPart);
  if (!Number.isFinite(minutesRaw) || minutesRaw < 0) return null;

  if (parts.length === 3) {
    const hoursPart = parts[0] || '';
    if (!/^[0-9]+$/.test(hoursPart)) return null;
    const hoursRaw = Number(hoursPart);
    if (!Number.isFinite(hoursRaw) || hoursRaw < 0) return null;

    const treatAsMinutes = hoursRaw >= 24;
    const hasMillis = secPart.length > 2 || secondsRaw > 59;
    if (treatAsMinutes || hasMillis) {
      const minutes = hoursRaw;
      const seconds = minutesRaw;
      if (seconds > 59) return null;
      return minutes * 60 + seconds;
    }

    if (minutesRaw > 59 || secondsRaw > 59) return null;
    return hoursRaw * 3600 + minutesRaw * 60 + secondsRaw;
  }

  if (secondsRaw > 59) return null;
  return minutesRaw * 60 + secondsRaw;
}

function parseYoutubeUrlSeconds(raw: string): number | null {
  const value = String(raw || '').trim();
  if (!value) return null;

  let url: URL | null = null;
  try {
    url = new URL(value);
  } catch {
    return null;
  }

  const params = url.searchParams;
  const rawT = params.get('t') || params.get('start') || '';
  if (!rawT) return null;

  if (/^\d+$/.test(rawT)) {
    const seconds = Number(rawT);
    return Number.isFinite(seconds) && seconds >= 0 ? seconds : null;
  }

  const match = rawT.match(/(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?/i);
  if (!match) return null;
  const hours = match[1] ? Number(match[1]) : 0;
  const minutes = match[2] ? Number(match[2]) : 0;
  const seconds = match[3] ? Number(match[3]) : 0;
  if ([hours, minutes, seconds].some((v) => Number.isNaN(v) || v < 0)) return null;
  return hours * 3600 + minutes * 60 + seconds;
}

export function formatSourceTimecode(source: ChatSource): string | null {
  const seconds = Number(source.seconds_since_start);
  if (Number.isFinite(seconds) && seconds > 0) {
    return formatSeconds(seconds);
  }

  const urlSeconds = parseYoutubeUrlSeconds(source.youtube_url || '');
  if (urlSeconds !== null) {
    return formatSeconds(urlSeconds);
  }

  const parsed = parseTimestampToSeconds(source.timestamp_str || '');
  if (parsed !== null) {
    return formatSeconds(parsed);
  }

  if (Number.isFinite(seconds) && seconds === 0) {
    return formatSeconds(0);
  }

  return null;
}
