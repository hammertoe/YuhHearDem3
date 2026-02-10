-- Add order paper tables for ingestion.

CREATE TABLE IF NOT EXISTS order_papers (
    id TEXT PRIMARY KEY,
    sitting_date DATE NOT NULL,
    order_paper_number TEXT NOT NULL,
    session TEXT,
    sitting_number TEXT,
    raw_text TEXT NOT NULL,
    parsed_json JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    CONSTRAINT uq_order_papers UNIQUE (sitting_date, order_paper_number)
);

CREATE TABLE IF NOT EXISTS order_paper_items (
    id TEXT PRIMARY KEY,
    order_paper_id TEXT NOT NULL,
    sequence INTEGER NOT NULL,
    item_type TEXT NOT NULL,
    title TEXT NOT NULL,
    mover TEXT,
    action TEXT,
    status_text TEXT,
    linked_bill_id TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    CONSTRAINT fk_op_items_order_paper
        FOREIGN KEY (order_paper_id) REFERENCES order_papers(id) ON DELETE CASCADE,
    CONSTRAINT fk_op_items_bill
        FOREIGN KEY (linked_bill_id) REFERENCES bills(id)
);

CREATE INDEX IF NOT EXISTS idx_order_papers_date ON order_papers(sitting_date);
CREATE INDEX IF NOT EXISTS idx_order_papers_number ON order_papers(order_paper_number);
CREATE INDEX IF NOT EXISTS idx_order_paper_items_order_paper ON order_paper_items(order_paper_id);
CREATE INDEX IF NOT EXISTS idx_order_paper_items_type ON order_paper_items(item_type);
