# Date Normalization Implementation Summary

## What Was Implemented

Added automatic normalization of relative date entities in the knowledge graph extraction script (`build_knowledge_graph.py`).

## Key Features

### 1. Relative Date Detection
Identifies relative date expressions containing keywords:
- today, yesterday, tomorrow
- last, next, ago, recent
- this, current

### 2. Anchor-Based Resolution
Uses video's `upload_date` from `video_metadata` as anchor point for resolving relative dates.

### 3. Smart Date Parsing
- Primary: Uses `dateparser` library for flexible date parsing
- Secondary: Custom handling for "last [Month]" patterns (e.g., "last December" → "2025-12")
- Fallback: Skips failed resolutions silently

### 4. Date Validation
Validates that resolved dates are within reasonable range (±100 years from current date).

### 5. Format-Based Output
- Year-month precision: "2025-12" (when day is not significant)
- Full date: "2025-12-01" (when specific day is meaningful)

## Examples

| Original Text | Resolved Date |
|---------------|---------------|
| today | 2026-01-06 |
| last year | 2025 |
| last December | 2025-12 |
| next week | 2026-01-13 (approximately) |

## Technical Changes

### New Dependencies
```bash
pip install dateparser
```

### Updated Data Structures
**Entity dataclass** now includes:
```python
resolved_date: Optional[str] = None  # Normalized absolute date
is_relative: bool = False           # Whether normalization was attempted
```

### New Methods in KnowledgeGraphBuilder

1. **`_is_relative_date(text: str) -> bool`**
   - Checks if text appears to be a relative date
   - Skips already-formatted dates (YYYY-MM-DD, YYYY)

2. **`_format_resolved_date(date_obj: datetime) -> str`**
   - Formats resolved date based on precision
   - Returns "YYYY-MM" or "YYYY-MM-DD" as appropriate

3. **`_is_reasonable_date(year: int) -> bool`**
   - Validates that date year is reasonable
   - Prevents absurd dates (e.g., 1800, 3000)

4. **`_resolve_last_month_pattern(text: str, anchor_date: datetime) -> Optional[datetime]`**
   - Manually resolves "last [Month]" patterns
   - Handles cases dateparser misses

5. **`normalize_relative_dates(anchor_date: datetime)`**
   - Main method for normalizing all DATE entities
   - Combines dateparser + custom resolution
   - Validates resolved dates

### Modified Methods

**`build_knowledge_graph()`**:
- Extracts anchor date from `video_metadata.upload_date`
- **Fails hard** if no valid anchor date found (as requested)
- Calls `normalize_relative_dates()` after entity extraction

**`_create_knowledge_graph()`**:
- Includes `resolved_date` and `is_relative` fields for DATE entities in output

**`export_networkx_html()`**:
- Shows resolved date in node tooltip when available

## Error Handling

- **No anchor date**: Script fails with clear error message (not fallback to `datetime.now()`)
- **Failed resolutions**: Skipped silently (as requested)
- **Invalid dates**: Rejected, not included in output

## Testing

Tested with various date expressions:
```python
"today" → "2026-01-06"  ✓
"last year" → "2025"  ✓
"last December" → "2025-12"  ✓
"last year December" → "2025-12"  ✓
"next week" → "2026-01-13"  ✓
"2025-12-01" → unchanged  ✓
"the 90s" → unchanged  ✓ (edge case, left as-is)
```

## Backward Compatibility

- **Existing graphs**: No changes required
- **New graphs**: Include `resolved_date` and `is_relative` fields for DATE entities
- **Optional fields**: JSON consumers can handle missing fields gracefully

## Requirements Updated

**README.md** updated to include:
- Date normalization feature in features list
- Dedicated "Date Normalization" section with examples
- `dateparser` in requirements
- Example JSON output showing normalized DATE entity

## Edge Cases Handled

1. **"this day" / "this new year"**: Not resolved (dateparser returns None)
2. **"the 90s"**: Left as textual label (not a date)
3. **"Easter Monday"**: Left as textual label (too ambiguous)
4. **"ages past" / "years"**: Not recognized as dates (temporal phrases)
5. **Already formatted dates**: Skipped (e.g., "2025-12-01")

## Future Enhancements (Not Implemented)

- LLM fallback for complex expressions
- Month-year precision (currently only year-month or full date)
- Time resolution (e.g., "3pm", "in 2 hours") - left as-is per requirements
- Multiple interpretation tracking for ambiguous dates
