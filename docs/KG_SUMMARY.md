# Knowledge Graph Summary

## Generated Files

- **knowledge_graph.json** (170KB): Structured knowledge graph data
- **knowledge_graph.html** (95KB): Interactive visualization

## Statistics

### Overall
- **Total Nodes**: 285
- **Total Edges**: 262

### Entity Types Distribution
| Type | Count |
|------|-------|
| DATE | 89 |
| ORG | 53 |
| CARDINAL | 37 |
| TIME | 21 |
| PERSON | 18 |
| FAC | 13 |
| GPE | 12 |
| NORP | 5 |
| MONEY | 12 |
| ORDINAL | 3 |
| LAW | 3 |
| QUANTITY | 2 |
| LOC | 3 |
| EVENT | 1 |
| PRODUCT | 2 |
| WORK_OF_ART | 1 |
| SPEAKER | 7 |
| LEGISLATION | 3 |

### Relationship Types Distribution
| Type | Count |
|------|-------|
| DISCUSSES | 199 |
| PROPOSES | 16 |
| MENTIONS | 12 |
| REFERENCES | 13 |
| WORKS_WITH | 10 |
| CRITICIZES | 7 |
| ADVOCATES_FOR | 3 |
| QUESTIONS | 2 |

## Key Insights

1. **Most Common Relationships**: "DISCUSSES" dominates (199 edges), indicating active debate on topics
2. **Legislation Focus**: "PROPOSES" (16 edges) shows active legislative proposals
3. **Entity Diversity**: Strong representation of dates, organizations, and locations
4. **Speaker Entities**: 7 unique speakers identified and added as nodes
5. **Legislation References**: 3 legislation nodes automatically added

## Sample Relationship Examples

- DISCUSSES: The House of Assembly First Session → 2026
- DISCUSSES: this day → this new year
- PROPOSES: Speaker → minutes
- REFERENCES: Speaker → Road Traffic Act

## Next Steps

1. Open `knowledge_graph.html` in browser to explore interactively
2. Use `knowledge_graph.json` for:
   - Graph database import (Neo4j, Memgraph)
   - Analytics and insights
   - ML pipeline integration
3. Extend with custom relationship types if needed
