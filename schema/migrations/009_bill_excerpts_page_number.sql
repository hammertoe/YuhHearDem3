-- Add page-level references for bill excerpt citations.

ALTER TABLE bill_excerpts
ADD COLUMN IF NOT EXISTS page_number INTEGER;
